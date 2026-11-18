"""Microphone input for the orb: wake word, end-of-speech detection and STT.

One continuous 16 kHz mono ``sounddevice`` stream feeds a queue of 80 ms
chunks (1280 samples, the frame size openWakeWord expects). Wake-word
detection and utterance recording both consume that queue, so there is no gap
between "wake word heard" and the start of recording.

The detection / segmentation logic lives in small classes driven by plain
arrays (:class:`WakeWordDetector`, :class:`SpeechProbability`,
:class:`UtteranceSegmenter`), so it can be tested without a microphone.
"""

from __future__ import annotations

import logging
import math
import os
import platform
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from openjarvis.orb.config import DATA_DIR, EarsConfig

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger("openjarvis.orb.ears")

SAMPLE_RATE = 16_000
CHUNK = 1280  # 80 ms, openWakeWord frame size
CHUNK_SECONDS = CHUNK / SAMPLE_RATE

OWW_DIR = DATA_DIR / "openwakeword"
WHISPER_DIR = DATA_DIR / "whisper"

# Silero VAD hysteresis: start speech above SPEECH_ON, keep it above SPEECH_OFF.
SPEECH_ON = 0.5
SPEECH_OFF = 0.35
DEFAULT_START_TIMEOUT = 8.0

# Whisper's classic hallucinations on near-silence (French subtitles corpus).
_HALLUCINATIONS = (
    "sous-titrage",
    "sous-titres réalisés",
    "amara.org",
    "merci d'avoir regardé",
    "abonnez-vous",
)


def _ensure_ssl() -> None:
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception as exc:  # pragma: no cover - best effort
        logger.debug("truststore injection failed: %s", exc)


def chunk_level(chunk: "np.ndarray") -> float:
    """Map an int16 or float32 chunk to a 0..1 loudness for the orb animation."""
    import numpy as np

    x = chunk.astype(np.float32)
    if chunk.dtype == np.int16:
        x /= 32768.0
    rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
    if rms <= 1e-6:
        return 0.0
    db = 20.0 * math.log10(rms)  # about -60 dBFS (quiet room) .. -15 (loud voice)
    return min(1.0, max(0.0, (db + 55.0) / 40.0))


# ---------------------------------------------------------------------------
# Array-driven building blocks
# ---------------------------------------------------------------------------


def ensure_oww_models(wake_word: str, target: Path = OWW_DIR) -> dict[str, str]:
    """Download (once) the openWakeWord feature, VAD and wake-word ONNX models."""
    target.mkdir(parents=True, exist_ok=True)
    custom = Path(wake_word).expanduser()
    names = ["melspectrogram.onnx", "embedding_model.onnx", "silero_vad.onnx"]
    need_download = not all((target / n).exists() for n in names)
    if not custom.suffix:
        need_download = need_download or not list(target.glob(f"{wake_word}*.onnx"))
    if need_download:
        _ensure_ssl()
        from openwakeword.utils import download_models

        logger.info("Downloading openWakeWord models into %s", target)
        download_models(
            [] if custom.suffix else [wake_word], target_directory=str(target)
        )
    if custom.suffix:  # explicit path to a custom .onnx model
        wake_path = custom
    else:
        found = sorted(target.glob(f"{wake_word}*.onnx"))
        if not found:
            raise FileNotFoundError(
                f"openWakeWord model '{wake_word}' not found in {target}"
            )
        wake_path = found[-1]
    return {
        "wake": str(wake_path),
        "melspec": str(target / "melspectrogram.onnx"),
        "embedding": str(target / "embedding_model.onnx"),
        "vad": str(target / "silero_vad.onnx"),
    }


class WakeWordDetector:
    """openWakeWord (ONNX) fed with 80 ms int16 chunks."""

    def __init__(self, paths: dict[str, str], threshold: float = 0.5):
        from openwakeword.model import Model

        self.threshold = threshold
        self.model = Model(
            wakeword_models=[paths["wake"]],
            inference_framework="onnx",
            melspec_model_path=paths["melspec"],
            embedding_model_path=paths["embedding"],
        )

    def score(self, chunk: "np.ndarray") -> float:
        preds = self.model.predict(chunk)
        return float(max(preds.values())) if preds else 0.0

    def process(self, chunk: "np.ndarray") -> bool:
        """Return True when the wake word is detected (and reset the buffers)."""
        s = self.score(chunk)
        if s >= self.threshold:
            logger.debug("wake word score %.2f", s)
            self.reset()
            return True
        return False

    def reset(self) -> None:
        self.model.reset()


class SpeechProbability:
    """Silero VAD (bundled with openWakeWord) on 80 ms int16 chunks."""

    def __init__(self, vad_path: str):
        from openwakeword.vad import VAD

        self.vad = VAD(model_path=vad_path)

    def __call__(self, chunk: "np.ndarray") -> float:
        # Two 40 ms frames per chunk; take the max so short onsets are not lost.
        a = float(self.vad.predict(chunk[: CHUNK // 2], frame_size=CHUNK // 2))
        b = float(self.vad.predict(chunk[CHUNK // 2 :], frame_size=CHUNK // 2))
        return max(a, b)

    def reset(self) -> None:
        self.vad.reset_states()


class UtteranceSegmenter:
    """State machine cutting one utterance out of a stream of chunks.

    Feed ``push(chunk, speech_prob)`` per chunk; it returns ``"done"`` once
    ``silence_seconds`` of silence follow speech (or ``max_seconds`` is hit),
    ``"timeout"`` if speech never started within ``start_timeout``, else None.
    """

    def __init__(
        self,
        silence_seconds: float = 0.8,
        max_seconds: float = 30.0,
        start_timeout: float | None = DEFAULT_START_TIMEOUT,
        pre_roll_seconds: float = 0.32,
        start_chunks: int = 2,
        min_speech_chunks: int = 3,
        tail_seconds: float = 0.2,
    ):
        self.silence_chunks = max(1, round(silence_seconds / CHUNK_SECONDS))
        self.max_chunks = max(1, round(max_seconds / CHUNK_SECONDS))
        self.timeout_chunks = (
            None
            if start_timeout is None
            else max(1, round(start_timeout / CHUNK_SECONDS))
        )
        self.start_chunks = start_chunks
        self.min_speech_chunks = min_speech_chunks
        self.tail_chunks = max(0, round(tail_seconds / CHUNK_SECONDS))
        self.pre_roll: deque = deque(
            maxlen=max(1, round(pre_roll_seconds / CHUNK_SECONDS))
        )
        self.chunks: list = []
        self.started = False
        self.waited = 0  # chunks seen before speech started
        self.onset = 0  # consecutive speech chunks while waiting
        self.voiced = 0  # speech chunks since start
        self.silence = 0  # trailing silent chunks
        self.total = 0  # chunks pushed
        self.speech_end: int | None = None  # stream chunk index where speech ended

    def push(self, chunk: "np.ndarray", prob: float) -> str | None:
        self.total += 1
        if not self.started:
            self.waited += 1
            self.pre_roll.append(chunk)
            self.onset = self.onset + 1 if prob >= SPEECH_ON else 0
            if self.onset >= self.start_chunks:
                self.started = True
                self.chunks = list(self.pre_roll)
                self.pre_roll.clear()
                self.voiced = self.onset
                self.silence = 0
            elif self.timeout_chunks is not None and self.waited >= self.timeout_chunks:
                return "timeout"
            return None

        self.chunks.append(chunk)
        if prob >= SPEECH_OFF:
            self.voiced += 1
            self.silence = 0
        else:
            self.silence += 1
        if self.silence >= self.silence_chunks:
            if self.voiced < self.min_speech_chunks:  # click / cough: keep waiting
                self.pre_roll.extend(self.chunks[-self.pre_roll.maxlen :])
                self.started, self.chunks, self.onset, self.voiced = False, [], 0, 0
                return None
            self._cut()
            return "done"
        if len(self.chunks) >= self.max_chunks:
            self._cut()
            return "done"
        return None

    def _cut(self) -> None:
        self.speech_end = self.total - self.silence
        drop = max(0, self.silence - self.tail_chunks)
        if drop:
            self.chunks = self.chunks[: len(self.chunks) - drop]

    def audio(self) -> "np.ndarray":
        """The utterance as float32 in [-1, 1]."""
        import numpy as np

        if not self.chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.chunks).astype(np.float32) / 32768.0


# ---------------------------------------------------------------------------
# Ears
# ---------------------------------------------------------------------------


class Ears:
    """Wake word + utterance recording + faster-whisper transcription."""

    def __init__(self, cfg: EarsConfig, language: str = "fr"):
        self.cfg = cfg
        self.language = language
        self._queue: queue.Queue = queue.Queue(maxsize=int(60 / CHUNK_SECONDS))
        self._stream = None
        self._stream_lock = threading.Lock()
        self._muted = threading.Event()
        # Barge-in: while muted (assistant speaking), chunks go to a separate
        # queue that only the interrupt detector reads.
        self._barge = threading.Event()
        self._barge_queue: queue.Queue = queue.Queue(maxsize=int(5 / CHUNK_SECONDS))
        self._wake: WakeWordDetector | None = None
        self._vad: SpeechProbability | None = None
        self._whisper = None
        self._load_lock = threading.Lock()

    # -- loading ----------------------------------------------------------

    def _load_wake(self) -> None:
        with self._load_lock:
            if self._wake is None or self._vad is None:
                paths = ensure_oww_models(self.cfg.wake_word)
                self._wake = WakeWordDetector(paths, self.cfg.wake_threshold)
                self._vad = SpeechProbability(paths["vad"])

    def _stt_engine(self) -> str:
        engine = (getattr(self.cfg, "stt_engine", "auto") or "auto").lower()
        if engine != "auto":
            return engine
        if sys.platform == "darwin" and platform.machine() == "arm64":
            try:
                import mlx_whisper  # noqa: F401
            except ImportError:
                return "faster-whisper"
            return "mlx"
        return "faster-whisper"

    def _load_whisper(self) -> None:
        with self._load_lock:
            if self._whisper is not None:
                return
            _ensure_ssl()
            t0 = time.perf_counter()
            model = self.cfg.stt_model
            if self._stt_engine() == "mlx":
                # Whisper on the Apple GPU: ~0.2 s (small) / ~0.5 s (turbo)
                # for a 5 s utterance, versus ~1 s for small on CPU.
                self._mlx_repo = (
                    model
                    if "/" in model
                    else "mlx-community/whisper-large-v3-turbo"
                    if model == "large-v3-turbo"
                    else f"mlx-community/whisper-{model}-mlx"
                )
                self._whisper = "mlx"
                logger.info("STT engine: mlx-whisper %s", self._mlx_repo)
                return
            from faster_whisper import WhisperModel

            if model.startswith("large"):
                logger.warning(
                    "STT model %r is too slow on CPU; using 'small' instead", model
                )
                model = "small"
            WHISPER_DIR.mkdir(parents=True, exist_ok=True)
            self._whisper = WhisperModel(
                model,
                device="cpu",
                compute_type="int8",
                # 4 threads: best latency when other apps load the CPU; more
                # threads gain little on an idle M-series and lose a lot
                # under contention.
                cpu_threads=min(4, os.cpu_count() or 4),
                download_root=str(WHISPER_DIR),
            )
            logger.info(
                "faster-whisper '%s' loaded in %.1fs",
                model,
                time.perf_counter() - t0,
            )

    def warm_up(self) -> None:
        import numpy as np

        self._load_wake()
        self._load_whisper()
        # One dummy pass of each model so the first real request is fast.
        silence = np.zeros(CHUNK, dtype=np.int16)
        assert self._wake is not None and self._vad is not None
        for _ in range(3):
            self._wake.score(silence)
            self._vad(silence)
        self._wake.reset()
        self._vad.reset()
        t = np.arange(SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
        self.transcribe((0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32))
        self._ensure_stream()

    # -- microphone -------------------------------------------------------

    def _resolve_device(self) -> int | None:
        name = (self.cfg.input_device or "").strip().lower()
        if not name:
            return None
        import sounddevice as sd

        for idx, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0 and name in dev["name"].lower():
                return idx
        logger.warning(
            "Input device '%s' not found, using the default", self.cfg.input_device
        )
        return None

    def _ensure_stream(self) -> None:
        with self._stream_lock:
            if self._stream is not None:
                return
            import sounddevice as sd

            def callback(indata, frames, time_info, status) -> None:  # audio thread
                if status:
                    logger.debug("input status: %s", status)
                if self._muted.is_set():
                    if self._barge.is_set():
                        try:
                            self._barge_queue.put_nowait(indata[:, 0].copy())
                        except queue.Full:
                            pass
                    return
                try:
                    self._queue.put_nowait(indata[:, 0].copy())
                except queue.Full:
                    try:
                        self._queue.get_nowait()
                        self._queue.put_nowait(indata[:, 0].copy())
                    except (queue.Empty, queue.Full):
                        pass

            device = self._resolve_device()
            try:
                self._stream = sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="int16",
                    blocksize=CHUNK,
                    device=device,
                    callback=callback,
                )
                self._stream.start()
            except Exception as exc:
                self._stream = None
                raise RuntimeError(
                    f"Cannot open the microphone ({exc}). On macOS, allow microphone "
                    "access for the terminal/app in System Settings > Privacy & "
                    "Security > Microphone."
                ) from exc
            logger.info("Microphone stream open (device=%s)", device)

    def _drain(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def _next_chunk(self, stop: threading.Event) -> "np.ndarray | None":
        """Next 80 ms int16 chunk, or None if stopped. Waits while muted."""
        while not stop.is_set():
            if self._muted.is_set():
                self._drain()
                stop.wait(0.05)
                continue
            try:
                return self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
        return None

    def mute(self, muted: bool) -> None:
        if muted:
            self._muted.set()
            self._drain()
        else:
            self._drain()  # drop anything captured while we were speaking
            self._muted.clear()

    def close(self) -> None:
        with self._stream_lock:
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception as exc:  # pragma: no cover
                    logger.debug("closing stream: %s", exc)
                self._stream = None
        self._drain()

    # -- public API -------------------------------------------------------

    def wait_for_wake_word(
        self,
        stop: threading.Event,
        level_cb: Callable[[float], None] | None = None,
    ) -> bool:
        self._load_wake()
        self._ensure_stream()
        assert self._wake is not None
        self._wake.reset()
        while True:
            chunk = self._next_chunk(stop)
            if chunk is None:
                return False
            if level_cb is not None:
                level_cb(chunk_level(chunk))
            if self._wake.process(chunk):
                logger.info("Wake word detected")
                return True

    def listen_for_interrupt(self, stop: threading.Event, threshold: float) -> bool:
        """While muted (assistant speaking), watch for the wake word.

        Returns True if the user said the wake word (barge-in), False once
        ``stop`` is set. Uses a stricter threshold than normal wake-up because
        the microphone also hears the assistant's own voice.
        """
        self._load_wake()
        assert self._wake is not None
        while True:
            try:
                self._barge_queue.get_nowait()
            except queue.Empty:
                break
        self._wake.reset()
        self._barge.set()
        try:
            while not stop.is_set():
                try:
                    chunk = self._barge_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                score = self._wake.score(chunk)
                if score >= threshold:
                    logger.info("Barge-in: wake word while speaking (%.2f)", score)
                    return True
            return False
        finally:
            self._barge.clear()
            self._wake.reset()

    def record_utterance(
        self,
        stop: threading.Event,
        level_cb: Callable[[float], None] | None = None,
        start_timeout: float | None = None,
    ) -> "np.ndarray | None":
        """Record until ``silence_seconds`` of silence.

        ``start_timeout=None`` uses a default of 8 s, so a false wake-up never
        leaves the assistant listening forever.
        """
        self._load_wake()
        self._ensure_stream()
        assert self._vad is not None
        self._vad.reset()
        seg = UtteranceSegmenter(
            silence_seconds=self.cfg.silence_seconds,
            max_seconds=self.cfg.max_utterance_seconds,
            start_timeout=DEFAULT_START_TIMEOUT
            if start_timeout is None
            else start_timeout,
        )
        while True:
            chunk = self._next_chunk(stop)
            if chunk is None:
                return None
            if level_cb is not None:
                level_cb(chunk_level(chunk))
            result = seg.push(chunk, self._vad(chunk))
            if result == "timeout":
                return None
            if result == "done":
                audio = seg.audio()
                logger.debug("utterance: %.2fs", audio.size / SAMPLE_RATE)
                return audio

    def transcribe(self, audio: "np.ndarray") -> str:
        import numpy as np

        if audio is None or audio.size < SAMPLE_RATE // 10:
            return ""
        self._load_whisper()
        audio = np.asarray(audio, dtype=np.float32)
        t0 = time.perf_counter()
        # np.errstate: numpy + Accelerate raise spurious matmul warnings in
        # faster-whisper's feature extractor on Apple Silicon.
        if self._whisper == "mlx":
            import mlx_whisper

            result = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=self._mlx_repo,
                language=self.language,
                condition_on_previous_text=False,
                temperature=0.0,
                verbose=None,
            )
            text = str(result.get("text", "")).strip()
        else:
            # np.errstate: numpy + Accelerate raise spurious matmul warnings in
            # faster-whisper's feature extractor on Apple Silicon.
            with np.errstate(all="ignore"):
                segments, _info = self._whisper.transcribe(
                    audio,
                    language=self.language,
                    beam_size=1,
                    vad_filter=False,
                    condition_on_previous_text=False,
                    without_timestamps=True,
                    temperature=0.0,
                )
                text = " ".join(s.text.strip() for s in segments).strip()
        logger.info(
            "STT %.2fs audio -> %.2fs: %r",
            audio.size / SAMPLE_RATE,
            time.perf_counter() - t0,
            text,
        )
        if any(h in text.lower() for h in _HALLUCINATIONS):
            return ""
        return text
