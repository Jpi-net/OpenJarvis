"""Tests for the local Chatterbox voice, with a fake worker (no model, no GPU).

The fake worker is a real subprocess speaking the real protocol: it reuses
``chatterbox_worker.serve`` / the framing helpers with a fake engine that
returns a tone whose length depends on the text.
"""

from __future__ import annotations

import io
import subprocess
import sys
import textwrap
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from openjarvis.orb import chatterbox_worker as W
from openjarvis.orb import voice as V

SR = V.CHATTERBOX_SAMPLE_RATE
WORKER_FILE = Path(W.__file__)

FAKE_WORKER = textwrap.dedent(
    """
    import importlib.util, os, sys, time
    spec = importlib.util.spec_from_file_location("cbw", {worker!r})
    cbw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cbw)
    import numpy as np

    MODE = os.environ.get("FAKE_CBX_MODE", "")

    class Engine:
        def __init__(self):
            self.said = 0

        def load(self, req):
            if MODE == "fail_load":
                raise RuntimeError("no weights")
            return {{"ok": True, "sample_rate": 24000, "load_s": 0.01,
                     "quantize": req.get("quantize"),
                     "flow_steps": req.get("flow_steps")}}

        def voice(self, req):
            if not os.path.isfile(req["reference"]):
                raise FileNotFoundError(req["reference"])
            return {{"ok": True, "voice_s": 0.0, "gain": 1.0}}

        def say(self, req):
            self.said += 1
            if MODE == "die_on_2" and self.said == 2:
                os._exit(3)
            if MODE == "slow":
                time.sleep(5)
            n = int(0.01 * 24000 * len(req["text"]))
            t = np.arange(n) / 24000
            tone = (0.3 * np.sin(2 * np.pi * 220 * t)).astype("<f4")
            tone = cbw.time_stretch(tone, float(req.get("speed", 1.0)))
            header = {{"ok": True, "sample_rate": 24000, "gen_s": 0.0,
                       "echo": {{k: req.get(k) for k in
                                ("lang", "cfg_weight", "exaggeration", "speed")}}}}
            return header, tone.astype("<f4").tobytes()

    out = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)
    cbw.serve(os.fdopen(os.dup(0), "rb", buffering=0), out, Engine())
    """
)


@pytest.fixture
def fake_worker(tmp_path: Path) -> list[str]:
    script = tmp_path / "fake_worker.py"
    script.write_text(FAKE_WORKER.format(worker=str(WORKER_FILE)), encoding="utf-8")
    return [sys.executable, "-I", str(script)]


@pytest.fixture
def voices(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    vdir = tmp_path / "voices"
    vdir.mkdir()
    (vdir / "grave.wav").write_bytes(b"RIFF")
    monkeypatch.setattr(V, "CHATTERBOX_VOICES", vdir)
    monkeypatch.setattr(V, "CHATTERBOX_USER_REFERENCE", vdir / "reference.wav")
    monkeypatch.setattr(V, "CHATTERBOX_LOG", None)
    return vdir


@dataclass
class Cfg:
    engine: str = "chatterbox"
    local_only: bool = True
    piper_voice: str = "fr_FR-tom-medium"
    say_voice: str = "Thomas"
    speed: float = 1.15
    effect: str = "jarvis"
    chatterbox_reference: str = ""
    chatterbox_exaggeration: float = 0.4
    chatterbox_cfg_weight: float = 0.5
    chatterbox_speed: float = 1.0


class WavSink:
    def __init__(self) -> None:
        self.blocks: list[np.ndarray] = []
        self.rate = 0

    def start(self, sample_rate: int) -> None:
        self.rate = sample_rate

    def write(self, block: np.ndarray) -> None:
        self.blocks.append(block.copy())

    def finish(self) -> None:
        pass

    def abort(self) -> None:
        pass

    @property
    def audio(self) -> np.ndarray:
        return np.concatenate(self.blocks) if self.blocks else np.zeros(0)


class FakeChunk:
    def __init__(self, audio: np.ndarray) -> None:
        self.audio_float_array = audio


class FakePiper:
    """Stands in for a loaded PiperVoice: 22.05 kHz tone, 5 ms per char."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    def synthesize(self, text: str, _cfg: object) -> list[FakeChunk]:
        self.texts.append(text)
        n = int(22050 * 0.005 * len(text))
        return [FakeChunk(np.full(n, 0.2, dtype=np.float32))]


def make_voice(
    monkeypatch: pytest.MonkeyPatch, command: list[str], cfg: Cfg | None = None
) -> V.Voice:
    monkeypatch.setattr(V, "chatterbox_command", lambda venv=None: command)
    voice = V.Voice(cfg or Cfg())
    piper = FakePiper()

    def fake_load_piper() -> None:
        voice._piper = piper
        voice._piper_rate = 22050

    monkeypatch.setattr(voice, "_load_piper", fake_load_piper)
    voice._sink = WavSink()
    return voice


# --------------------------------------------------------------------------
# Worker helpers (pure numpy)
# --------------------------------------------------------------------------


def test_frames_round_trip() -> None:
    buf = io.BytesIO()
    W.write_frame(buf, {"op": "say", "text": "Très bien"})
    W.write_frame(buf, {"ok": True}, b"\x00\x00\x80\x3f")
    buf.seek(0)
    assert W.read_frame(buf) == {"op": "say", "text": "Très bien"}
    header = W.read_frame(buf)
    assert header == {"ok": True, "nbytes": 4}
    assert buf.read(4) == b"\x00\x00\x80\x3f"
    assert W.read_frame(buf) is None  # EOF


def test_time_stretch_keeps_pitch() -> None:
    t = np.arange(SR * 2) / SR
    tone = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    fast = W.time_stretch(tone, 1.25)
    assert abs(len(fast) / SR - 2 / 1.25) < 0.02
    spectrum = np.abs(np.fft.rfft(fast[SR // 10 : -SR // 10]))
    peak_hz = np.argmax(spectrum) * SR / (len(fast) - SR // 5)
    assert abs(peak_hz - 200) < 5
    assert W.time_stretch(tone, 1.0) is tone


def test_drop_trailing_babble() -> None:
    def burst(seconds: float) -> np.ndarray:
        n = int(SR * seconds)
        return (0.4 * np.sin(np.arange(n) * 0.3)).astype(np.float32)

    gap = np.zeros(int(SR * 0.45), dtype=np.float32)
    speech = burst(2.0)
    babble = np.concatenate([speech, gap, burst(0.3)])
    assert len(W.drop_trailing_babble(babble)) == pytest.approx(
        len(speech), abs=SR // 50
    )
    two_parts = np.concatenate([speech, gap, burst(1.5)])  # a real second clause
    assert len(W.drop_trailing_babble(two_parts)) == len(two_parts)


def test_trim_silence_and_limits() -> None:
    audio = np.concatenate([np.zeros(SR), 0.5 * np.ones(SR // 2), np.zeros(SR)]).astype(
        np.float32
    )
    trimmed = W.trim_silence(audio)
    assert SR // 2 < len(trimmed) < SR // 2 + int(0.2 * SR)
    assert len(W.trim_silence(np.zeros(SR, dtype=np.float32))) == 0
    loud = np.array([0.5, 0.95, -1.4], dtype=np.float32)
    limited = W.soft_limit(loud)
    assert limited[0] == 0.5 and 0.85 < limited[1] < 1.0 and -1.0 < limited[2] < -0.85
    assert W.max_tokens_for("C'est fait.") >= 50
    assert W.max_tokens_for("x" * 5000) == 1000


def test_worker_refuses_network() -> None:
    code = textwrap.dedent(
        f"""
        import importlib.util, socket
        spec = importlib.util.spec_from_file_location("cbw", {str(WORKER_FILE)!r})
        cbw = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cbw)
        cbw.force_offline()
        import os
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        for attempt in (
            lambda: socket.create_connection(("127.0.0.1", 9), timeout=1),
            lambda: socket.socket(socket.AF_INET).connect(("127.0.0.1", 9)),
            lambda: socket.getaddrinfo("huggingface.co", 443),
        ):
            try:
                attempt()
            except OSError as exc:
                assert "disabled" in str(exc), exc
            else:
                raise SystemExit("network was allowed")
        a, b = socket.socketpair(socket.AF_UNIX)  # local pipes stay usable
        print("blocked")
        """
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert "blocked" in result.stdout


# --------------------------------------------------------------------------
# Client + Voice integration
# --------------------------------------------------------------------------


def test_worker_client_round_trip(fake_worker: list[str], voices: Path) -> None:
    worker = V.ChatterboxWorker(fake_worker, log_path=None)
    info = worker.start(
        {"model": "m", "quantize": 8, "flow_steps": 4},
        {"reference": str(voices / "grave.wav")},
    )
    assert info["ok"] and info["sample_rate"] == SR and info["flow_steps"] == 4
    audio = worker.say("Bonjour Monsieur.", {"lang": "fr"})
    assert audio.dtype == np.float32 and len(audio) == int(0.01 * SR * 17)
    with pytest.raises(V.ChatterboxError, match="FileNotFoundError"):
        worker.request({"op": "voice", "reference": "/nope.wav"}, 5.0)
    assert worker.alive  # an error reply does not kill the worker
    worker.close()
    assert not worker.alive


def test_worker_client_times_out(
    fake_worker: list[str], voices: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_CBX_MODE", "slow")
    worker = V.ChatterboxWorker(fake_worker, log_path=None)
    worker.start({}, {"reference": str(voices / "grave.wav")})
    t0 = time.monotonic()
    with pytest.raises(V.ChatterboxError, match="timed out"):
        worker.say("Bonjour.", {}, timeout=0.5)
    assert time.monotonic() - t0 < 3.0
    assert not worker.alive  # a stuck worker is killed


def test_voice_speaks_with_chatterbox(
    fake_worker: list[str], voices: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = Cfg(chatterbox_cfg_weight=0.3, chatterbox_exaggeration=0.35)
    voice = make_voice(monkeypatch, fake_worker, cfg)
    voice.warm_up()
    assert voice.engine == "chatterbox"  # local: allowed with local_only
    assert voice.sample_rate == SR
    assert voice._fx.name == "none"  # natural voice unless chatterbox_effect
    params = voice._chatterbox_params()
    assert params["cfg_weight"] == 0.3 and params["exaggeration"] == 0.35
    first: list[float] = []
    spoken = voice.speak_stream(
        iter(["Bonjour Monsieur. Tous les ", "systèmes sont opérationnels."]),
        threading.Event(),
        on_first_audio=lambda: first.append(1.0),
    )
    assert spoken == "Bonjour Monsieur. Tous les systèmes sont opérationnels."
    expected = int(0.01 * SR * 17) + int(0.01 * SR * 37)
    gaps = 2 * int(SR * V._SENTENCE_GAP_S)
    assert len(voice._sink.audio) == expected + gaps
    assert first == [1.0]
    voice.close()
    assert voice._cbx is None


def test_voice_falls_back_to_piper_when_worker_dies(
    fake_worker: list[str], voices: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_CBX_MODE", "die_on_2")  # warm-up is say #1
    voice = make_voice(monkeypatch, fake_worker)
    voice.warm_up()
    if voice._fallback_thread is not None:
        voice._fallback_thread.join(5)
    restarts: list[int] = []
    monkeypatch.setattr(voice, "_restart_chatterbox_async", lambda: restarts.append(1))
    spoken = voice.speak_stream(iter(["Très bien, Monsieur."]), threading.Event())
    assert spoken == "Très bien, Monsieur."
    assert voice._piper.texts == ["Très bien, Monsieur."]
    piper_len = int(int(22050 * 0.005 * 20) * SR / 22050)
    assert abs(len(voice._sink.audio) - piper_len - int(SR * V._SENTENCE_GAP_S)) < 5
    middle = voice._sink.audio[200 : piper_len - 200]  # level-matched, resampled
    assert np.median(middle) == pytest.approx(0.2 * V._FALLBACK_GAIN, rel=1e-3)
    assert restarts
    assert voice.engine == "chatterbox"  # it comes back once restarted


def test_voice_uses_piper_when_chatterbox_missing(
    voices: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(venv: Path | None = None) -> list[str]:
        raise V.ChatterboxError("not installed")

    voice = make_voice(monkeypatch, [])
    monkeypatch.setattr(V, "chatterbox_command", missing)
    voice.warm_up()
    assert voice.engine == "piper"


def test_voice_uses_piper_when_worker_fails_to_load(
    fake_worker: list[str], voices: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_CBX_MODE", "fail_load")
    voice = make_voice(monkeypatch, fake_worker)
    voice.warm_up()
    assert voice.engine == "piper"
    assert voice._cbx is None


def test_reference_resolution(voices: Path, tmp_path: Path) -> None:
    assert V.resolve_chatterbox_reference("") == voices / "grave.wav"
    (voices / "reference.wav").write_bytes(b"RIFF")
    assert V.resolve_chatterbox_reference("") == voices / "reference.wav"
    assert V.resolve_chatterbox_reference("pose") == voices / "pose.wav"
    assert V.resolve_chatterbox_reference("tom.wav") == voices / "tom.wav"
    own = tmp_path / "me.wav"
    assert V.resolve_chatterbox_reference(str(own)) == own
    assert V.resolve_chatterbox_reference("~/x.wav") == Path.home() / "x.wav"


def test_missing_reference_falls_back(
    fake_worker: list[str], voices: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    voice = make_voice(monkeypatch, fake_worker, Cfg(chatterbox_reference="absent"))
    voice.warm_up()
    assert voice.engine == "piper"


def test_chatterbox_command_requires_venv(tmp_path: Path) -> None:
    with pytest.raises(V.ChatterboxError, match="voice_setup"):
        V.chatterbox_command(tmp_path / "nope")
    py = tmp_path / "venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("")
    cmd = V.chatterbox_command(tmp_path / "venv")
    assert cmd[:2] == [str(py), "-I"] and cmd[2].endswith("chatterbox_worker.py")
