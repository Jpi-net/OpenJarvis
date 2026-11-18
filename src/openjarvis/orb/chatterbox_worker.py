"""Chatterbox TTS worker: a local process that turns sentences into audio.

Runs in its own venv (``~/.openjarvis/orb/chatterbox-venv``, made by
``python -m openjarvis.orb.voice_setup``) with ``mlx-audio``: Resemble AI's
Chatterbox multilingual model, ported to MLX, on the Apple GPU. The orb
starts it with ``<venv>/bin/python -I chatterbox_worker.py`` and keeps it
alive, so the model is loaded once.

Privacy: nothing leaves the Mac. The worker talks to its parent only over
its stdin/stdout pipes (no port, nothing listening), forces the Hugging Face
offline mode and refuses every network connection: the weights must already
be in the local cache.

Protocol (both directions): frames of a 4-byte big-endian length, then a
UTF-8 JSON header; a reply that carries audio has ``"nbytes": N`` in its
header and is followed by N bytes of float32 little-endian mono PCM.

Requests and replies::

    {"op": "load", "model": ..., "quantize": 8, "flow_steps": 4}
        -> {"ok": true, "sample_rate": 24000, "load_s": 3.1}
    {"op": "voice", "reference": "/path.wav", "exaggeration": 0.4}
        -> {"ok": true, "voice_s": 0.2}
    {"op": "say", "text": "...", "lang": "fr", "cfg_weight": 0.5,
     "exaggeration": 0.4, "temperature": 0.8, "speed": 1.0}
        -> {"ok": true, "sample_rate": 24000, "nbytes": N, "gen_s": 0.9} + PCM
    {"op": "ping"} -> {"ok": true}
    {"op": "quit"} -> (exits)

Any failure replies ``{"ok": false, "error": "..."}`` and the worker keeps
serving. This file must not import ``openjarvis``: it runs in another venv.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import sys
import time
from typing import Any, BinaryIO

DEFAULT_MODEL = "mlx-community/chatterbox-multilingual-v3"
SAMPLE_RATE = 24000
# Cap on speech tokens (25 per second): French is read at ~1.8 tokens per
# character; the cap leaves room for slow voices, not for long babble.
_TOKENS_PER_CHAR = 2.6
_MIN_TOKENS = 50
_MAX_TOKENS = 1000
_LEAD_KEEP_S = 0.02
_TRAIL_KEEP_S = 0.12
_SILENCE_LEVEL = 0.01
_RELATIVE_SILENCE = 0.04  # of the clip's peak
_BABBLE_GAP_S = 0.3
_BABBLE_MAX_S = 0.7
_TARGET_LEVEL = 0.11  # speech level (90th percentile of 10 ms RMS), ~piper's
_CALIBRATION_TEXT = "Bonjour, je suis prêt à vous aider."


# --------------------------------------------------------------------------
# Offline guarantees
# --------------------------------------------------------------------------


def force_offline() -> None:
    """Hugging Face offline mode + no network connection of any kind."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"

    def refuse(*_a: Any, **_k: Any) -> Any:
        raise OSError("network access is disabled in the Chatterbox worker")

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def connect(self: socket.socket, address: Any) -> Any:
        if self.family != getattr(socket, "AF_UNIX", None):
            refuse()
        return real_connect(self, address)

    def connect_ex(self: socket.socket, address: Any) -> Any:
        if self.family != getattr(socket, "AF_UNIX", None):
            refuse()
        return real_connect_ex(self, address)

    socket.socket.connect = connect  # type: ignore[method-assign]
    socket.socket.connect_ex = connect_ex  # type: ignore[method-assign]
    socket.create_connection = refuse  # type: ignore[assignment]
    socket.getaddrinfo = refuse  # type: ignore[assignment]


# --------------------------------------------------------------------------
# Framing
# --------------------------------------------------------------------------


def read_frame(stream: BinaryIO) -> dict[str, Any] | None:
    head = _read_exact(stream, 4)
    if head is None:
        return None
    (size,) = struct.unpack(">I", head)
    body = _read_exact(stream, size)
    if body is None:
        return None
    return json.loads(body.decode("utf-8"))


def write_frame(stream: BinaryIO, header: dict[str, Any], payload: bytes = b"") -> None:
    if payload:
        header = {**header, "nbytes": len(payload)}
    body = json.dumps(header).encode("utf-8")
    stream.write(struct.pack(">I", len(body)) + body)
    if payload:
        stream.write(payload)
    stream.flush()


def _read_exact(stream: BinaryIO, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


# --------------------------------------------------------------------------
# Audio helpers (numpy only)
# --------------------------------------------------------------------------


def time_stretch(audio: Any, rate: float, sr: int = SAMPLE_RATE) -> Any:
    """WSOLA time-stretch: ``rate`` > 1 speaks faster, pitch unchanged."""
    import numpy as np

    if abs(rate - 1.0) < 0.02 or len(audio) < sr // 10:
        return audio
    frame = int(0.030 * sr)
    hop_out = frame // 2
    hop_in = hop_out * rate
    tol = int(0.008 * sr)
    n = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(frame) / frame)  # periodic hann
    window = n.astype(np.float32)
    x = np.concatenate([np.zeros(tol, np.float32), audio, np.zeros(frame + tol * 2)])
    count = int((len(audio) - frame) / hop_in) + 1
    out = np.zeros(count * hop_out + frame, np.float32)
    norm = np.zeros_like(out)
    prev = tol  # position (in x) of the previous chosen frame
    for k in range(max(1, count)):
        nominal = tol + int(round(k * hop_in))
        if k == 0:
            pos = nominal
        else:
            target = x[prev + hop_out : prev + hop_out + frame]
            lo = max(0, nominal - tol)
            region = x[lo : nominal + tol + frame]
            if len(region) < frame or not np.any(target):
                pos = nominal
            else:
                corr = np.correlate(region, target, mode="valid")
                pos = lo + int(np.argmax(corr))
        seg = x[pos : pos + frame]
        if len(seg) < frame:
            break
        o = k * hop_out
        out[o : o + frame] += seg * window
        norm[o : o + frame] += window
        prev = pos
    norm[norm < 1e-3] = 1.0
    return (out / norm)[: int(len(audio) / rate)].astype(np.float32)


def _envelope(audio: Any, sr: int) -> tuple[Any, int, float]:
    """10 ms peak envelope, block size, and the silence threshold for it."""
    import numpy as np

    block = max(1, sr // 100)
    usable = len(audio) // block * block
    env = np.abs(audio[:usable]).reshape(-1, block).max(axis=1)
    peak = float(env.max()) if len(env) else 0.0
    return env, block, max(_SILENCE_LEVEL, _RELATIVE_SILENCE * peak)


def trim_silence(audio: Any, sr: int = SAMPLE_RATE) -> Any:
    """Cut the leading / trailing silence, keeping short margins."""
    import numpy as np

    env, block, threshold = _envelope(audio, sr)
    loud = np.flatnonzero(env > threshold)
    if not loud.size:
        return audio[:0]
    start = max(0, int(loud[0]) * block - int(_LEAD_KEEP_S * sr))
    end = min(len(audio), (int(loud[-1]) + 1) * block + int(_TRAIL_KEEP_S * sr))
    out = audio[start:end].copy()
    fade = min(len(out), int(0.01 * sr))
    if fade:
        out[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return out


def drop_trailing_babble(audio: Any, sr: int = SAMPLE_RATE) -> Any:
    """Remove a short burst after a clear pause at the very end of a sentence.

    Chatterbox sometimes keeps "talking" after the text is said (a syllable
    or two of babble after a pause). One sentence is synthesized at a time,
    so a lone fragment after a long final pause is almost always that.
    """
    import numpy as np

    env, block, threshold = _envelope(audio, sr)
    voiced = env > threshold
    idx = np.flatnonzero(voiced)
    if idx.size < 2:
        return audio
    gaps = np.flatnonzero(np.diff(idx) > 1)
    if not gaps.size:
        return audio
    last_gap = int(gaps[-1])
    gap_start, tail_start = int(idx[last_gap]) + 1, int(idx[last_gap + 1])
    gap_s = (tail_start - gap_start) * block / sr
    tail_s = (int(idx[-1]) + 1 - tail_start) * block / sr
    speech_s = (gap_start - int(idx[0])) * block / sr
    if gap_s >= _BABBLE_GAP_S and tail_s <= _BABBLE_MAX_S and speech_s >= 2 * tail_s:
        return audio[: gap_start * block]
    return audio


def speech_level(audio: Any, sr: int = SAMPLE_RATE) -> float:
    """Loudness of the voiced parts: 90th percentile of the 10 ms RMS."""
    import numpy as np

    block = max(1, sr // 100)
    usable = len(audio) // block * block
    if not usable:
        return 0.0
    rms = np.sqrt((audio[:usable].reshape(-1, block) ** 2).mean(axis=1))
    return float(np.percentile(rms, 90))


def soft_limit(audio: Any, knee: float = 0.85) -> Any:
    """Leave the signal untouched below ``knee``, round off the peaks above."""
    import numpy as np

    mag = np.abs(audio)
    over = mag > knee
    if np.any(over):
        room = 1.0 - knee
        audio = audio.copy()
        audio[over] = np.sign(audio[over]) * (
            knee + room * np.tanh((mag[over] - knee) / room)
        )
    return audio


def max_tokens_for(text: str) -> int:
    return int(max(_MIN_TOKENS, min(_MAX_TOKENS, _TOKENS_PER_CHAR * len(text) + 30)))


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------


class Engine:
    def __init__(self) -> None:
        self.model: Any = None
        self.conds: Any = None
        self.voice_key: tuple[Any, ...] | None = None
        self.gain = 1.0

    def load(self, req: dict[str, Any]) -> dict[str, Any]:
        import io
        from contextlib import redirect_stdout

        import mlx.nn as nn
        from huggingface_hub import snapshot_download
        from mlx_audio.tts.utils import load_model

        t0 = time.perf_counter()
        repo = str(req.get("model") or DEFAULT_MODEL)
        path = (
            repo
            if os.path.isdir(repo)
            else snapshot_download(repo, local_files_only=True)
        )
        with redirect_stdout(io.StringIO()):
            model = load_model(path)
        bits = int(req.get("quantize", 8) or 0)
        if bits in (4, 6, 8):
            # The speech-token LM is memory-bound: 8-bit halves its step time.
            nn.quantize(model.t3.tfmr, group_size=64, bits=bits)
        steps = int(req.get("flow_steps", 4) or 0)
        if steps > 0:
            model.s3gen.flow.n_timesteps = steps
        self.model = model
        return {
            "ok": True,
            "sample_rate": SAMPLE_RATE,
            "load_s": time.perf_counter() - t0,
        }

    def voice(self, req: dict[str, Any]) -> dict[str, Any]:
        import io
        from contextlib import redirect_stdout

        t0 = time.perf_counter()
        ref = os.path.expanduser(str(req.get("reference") or ""))
        if not ref or not os.path.isfile(ref):
            raise FileNotFoundError(f"reference clip not found: {ref or '(none)'}")
        exaggeration = float(req.get("exaggeration", 0.4))
        key = (ref, os.path.getmtime(ref), exaggeration)
        if key != self.voice_key:
            with redirect_stdout(io.StringIO()):
                self.conds = self.model.prepare_conditionals(
                    ref, 0, exaggeration=exaggeration
                )
            self.voice_key = key
            # Each reference comes out at its own loudness: calibrate once.
            self.gain = 1.0
            raw = self._generate(_CALIBRATION_TEXT, {"exaggeration": exaggeration})
            level = speech_level(raw)
            if level > 1e-4:
                self.gain = max(0.3, min(4.0, _TARGET_LEVEL / level))
        return {"ok": True, "voice_s": time.perf_counter() - t0, "gain": self.gain}

    def _generate(self, text: str, req: dict[str, Any]) -> Any:
        import io
        from contextlib import redirect_stdout

        import numpy as np

        with redirect_stdout(io.StringIO()):
            results = list(
                self.model.generate(
                    text,
                    conds=self.conds,
                    lang_code=str(req.get("lang") or "fr"),
                    exaggeration=float(req.get("exaggeration", 0.4)),
                    cfg_weight=float(req.get("cfg_weight", 0.5)),
                    temperature=float(req.get("temperature", 0.8)),
                    max_new_tokens=max_tokens_for(text),
                    verbose=False,
                )
            )
        parts = [np.asarray(r.audio, dtype=np.float32).reshape(-1) for r in results]
        audio = np.concatenate(parts) if parts else np.zeros(0, np.float32)
        return trim_silence(drop_trailing_babble(np.nan_to_num(audio)))

    def say(self, req: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
        if self.model is None or self.conds is None:
            raise RuntimeError("model or voice not loaded")
        text = str(req.get("text") or "").strip()
        if not text:
            return {"ok": True, "sample_rate": SAMPLE_RATE, "gen_s": 0.0}, b""
        t0 = time.perf_counter()
        audio = self._generate(text, req) * self.gain
        audio = time_stretch(audio, float(req.get("speed", 1.0) or 1.0))
        audio = soft_limit(audio)
        header = {"ok": True, "sample_rate": SAMPLE_RATE}
        header["gen_s"] = time.perf_counter() - t0
        return header, audio.astype("<f4").tobytes()


def serve(inp: BinaryIO, out: BinaryIO, engine: Any) -> None:
    while True:
        req = read_frame(inp)
        if req is None:
            return
        op = req.get("op")
        if op == "quit":
            return
        try:
            if op == "load":
                write_frame(out, engine.load(req))
            elif op == "voice":
                write_frame(out, engine.voice(req))
            elif op == "say":
                header, pcm = engine.say(req)
                write_frame(out, header, pcm)
            elif op == "ping":
                write_frame(out, {"ok": True})
            else:
                write_frame(out, {"ok": False, "error": f"unknown op {op!r}"})
        except Exception as exc:  # report, keep serving
            write_frame(
                out, {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
            )


def main() -> None:
    force_offline()
    # The protocol owns the real stdout; anything printed (by libraries too,
    # even from C) goes to stderr instead.
    out = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    inp = os.fdopen(os.dup(0), "rb", buffering=0)
    import warnings

    warnings.filterwarnings("ignore")
    serve(inp, out, Engine())


if __name__ == "__main__":
    main()
