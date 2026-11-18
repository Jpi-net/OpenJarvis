"""Text-to-speech for the orb: streamed, sentence-by-sentence, low latency.

``Voice.speak_stream`` consumes LLM text deltas, cuts them into sentences as
soon as possible, synthesizes them on a worker thread and plays them back to
back on a single sounddevice output stream, so the synthesis of sentence N+1
overlaps the playback of sentence N.

Engines:
- ``piper`` (default): local neural TTS. The voice is downloaded once from
  Hugging Face (``rhasspy/piper-voices``) into ``DATA_DIR/piper/``.
- ``say``: macOS built-in voices. Also used automatically when piper fails.

A multi-speaker piper voice takes its speaker after a ``#``:
``fr_FR-upmc-medium#pierre`` (a name of the voice's speaker map) or
``fr_FR-mls-medium#36`` (a numeric speaker id, when it is not a map name).

``cfg.effect`` names a post-processing preset of ``voice_fx`` (``none``,
``jarvis``...), applied to each synthesized chunk with state kept across the
chunks of a reply.
"""

from __future__ import annotations

import logging
import math
import os
import queue
import re
import subprocess
import tempfile
import threading
import time
import unicodedata
import wave
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from openjarvis.orb import voice_fx
from openjarvis.orb.config import DATA_DIR, VoiceConfig

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger("openjarvis.orb.voice")

PIPER_DIR = DATA_DIR / "piper"
_HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
_SAY_SAMPLE_RATE = 22050
_SAY_BASE_RATE = 185  # words per minute of `say` at speed 1.0
_BLOCKS_PER_SECOND = 30  # playback block size -> level_cb cadence
_SENTENCE_GAP_S = 0.06  # short silence appended after each sentence
_OUTPUT_LATENCY_S = 0.05


# --------------------------------------------------------------------------
# Text cleaning
# --------------------------------------------------------------------------

_MD_LINK = re.compile(r"\[([^\]]*)\]\((?:[^)\s]+)\)")
_URL = re.compile(r"(?:https?://|www\.)\S*[^\s.,;:!?…)\]»\"']", re.IGNORECASE)
_CODE_FENCE = re.compile(r"```[^\n]*")
_BULLET = re.compile(r"^\s*(?:[-*+•▪◦‣–—]|#{1,6})\s+", re.MULTILINE)
_MD_SYMBOLS = re.compile(r"[*#`~|>]+|(?<!\w)_+|_+(?!\w)")
_SPACES = re.compile(r"\s+")


def _is_emoji(ch: str) -> bool:
    cp = ord(ch)
    if cp in (0x200D, 0xFE0F, 0xFE0E, 0x20E3):  # joiners / variation selectors
        return True
    if 0x1F000 <= cp <= 0x1FAFF or 0x2600 <= cp <= 0x27BF or 0x1F1E6 <= cp <= 0x1F1FF:
        return True
    if 0x2B00 <= cp <= 0x2BFF or 0xE0000 <= cp <= 0xE007F:
        return True
    return unicodedata.category(ch) == "So"


def clean_text(text: str) -> str:
    """Make LLM output speakable: no markdown, URLs, emojis."""
    text = _CODE_FENCE.sub(" ", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _URL.sub("le lien", text)
    text = _BULLET.sub("", text)
    text = _MD_SYMBOLS.sub("", text)
    text = "".join(ch for ch in text if not _is_emoji(ch))
    text = _SPACES.sub(" ", text).strip()
    # Keep French spacing for ":;!?" but drop stray spaces before ",." left by
    # removed symbols.
    return re.sub(r"\s+([,.…])", r"\1", text)


# --------------------------------------------------------------------------
# Incremental sentence splitter
# --------------------------------------------------------------------------

_HARD_END = ".!?…:;"
_CLOSERS = "\"'»)]”’"
_ABBREVIATIONS = {
    "mm",
    "mme",
    "mmes",
    "mlle",
    "mlles",
    "dr",
    "pr",
    "me",
    "st",
    "ste",
    "etc",
    "cf",
    "ex",
    "p",
    "pp",
    "no",
    "vol",
    "av",
    "bd",
    "env",
    "min",
    "max",
    "approx",
    "c.-à-d",
    "c-à-d",
    "mr",
    "mrs",
    "ms",
    "vs",
    "jr",
    "sr",
    "fig",
    "chap",
    "art",
    "réf",
    "tél",
    "j.-c",
}
_WORD_BEFORE = re.compile(r"([\w.\-]+)$")


def _alpha_count(s: str) -> int:
    return sum(ch.isalpha() for ch in s)


class SentenceSplitter:
    """Cut a stream of text deltas into speakable sentences, as early as possible.

    A sentence ends on ``. ! ? … : ;`` (followed by whitespace) or a newline.
    Decimal numbers (``3.5``), times (``10:30``), URLs and common abbreviations
    (``M.``, ``Dr.``, ``etc.``) are not split. A long clause is flushed on a
    comma so speech can start before a long sentence is complete; the first
    segment uses a lower threshold to minimise time-to-first-audio.
    """

    def __init__(
        self, comma_flush: int = 120, first_comma_flush: int = 60, hard_limit: int = 260
    ) -> None:
        self._buf = ""
        self._comma_flush = comma_flush
        self._first_comma_flush = first_comma_flush
        self._hard_limit = hard_limit
        self._emitted = 0
        self._carry = ""  # fragment too short to be spoken alone

    def feed(self, delta: str) -> list[str]:
        self._buf += delta
        out: list[str] = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            piece, self._buf = self._buf[:cut], self._buf[cut:]
            self._emit(piece, out)
        return out

    def flush(self) -> list[str]:
        out: list[str] = []
        piece, self._buf = self._buf, ""
        self._emit(piece, out, final=True)
        return out

    # -- internals ---------------------------------------------------------

    def _emit(self, piece: str, out: list[str], final: bool = False) -> None:
        text = (self._carry + " " + piece).strip() if self._carry else piece.strip()
        self._carry = ""
        if not text:
            return
        if _alpha_count(text) < 2 and not final:
            self._carry = text  # e.g. "1." of a numbered list: glue to the next
            return
        if _alpha_count(text) == 0:
            return
        out.append(text)
        self._emitted += 1

    def _find_cut(self) -> int | None:
        buf = self._buf
        n = len(buf)
        i = 0
        while i < n:
            ch = buf[i]
            if ch == "\n":
                return i + 1
            if ch in _HARD_END:
                j = i
                while j < n and buf[j] in _HARD_END:
                    j += 1
                while j < n and buf[j] in _CLOSERS:
                    j += 1
                if j >= n:
                    return None  # need the next char to decide ("3." + "5")
                if buf[j].isspace() and self._is_boundary(buf, i, j):
                    # French closing quote after a space: « ... fin. »
                    k = j
                    while k < n and buf[k] == " ":
                        k += 1
                    if k >= n:
                        return None
                    if buf[k] in "»”":
                        return k + 1
                    return j
                i = j
                continue
            i += 1
        return self._soft_cut(buf)

    @staticmethod
    def _is_boundary(buf: str, i: int, j: int) -> bool:
        if buf[i] != ".":
            return True
        if i > 0 and buf[i - 1] == ".":
            return True  # "..." ellipsis
        m = _WORD_BEFORE.search(buf[:i])
        if not m:
            return True
        word = m.group(1).lower()
        if word in _ABBREVIATIONS:
            # "etc." usually ends a sentence when a capital follows.
            nxt = buf[j:].lstrip()
            return word == "etc" and nxt[:1].isupper()
        # Single capital letter: initials ("J. Dupont"), "M. Durand".
        return not (len(word) == 1 and buf[i - 1].isupper())

    def _soft_cut(self, buf: str) -> int | None:
        limit = self._first_comma_flush if self._emitted == 0 else self._comma_flush
        if len(buf) <= limit:
            return None
        # Cut on a ", " (or a dash) with a reasonable clause before it: the
        # last one within the limit, else the first one after it.
        cands = sorted(
            m.start() for m in re.finditer(r", | [—–-] ", buf) if m.start() >= 20
        )
        if cands:
            within = [k for k in cands if k <= limit]
            return (within[-1] if within else cands[0]) + 1
        if len(buf) > self._hard_limit:
            k = buf.rfind(" ")
            if k > 40:
                return k + 1
        return None


# --------------------------------------------------------------------------
# Audio sinks
# --------------------------------------------------------------------------


class _Sink(Protocol):
    def start(self, sample_rate: int) -> None: ...
    def write(self, block: np.ndarray) -> None: ...  # blocks ~ playback time
    def finish(self) -> None: ...  # drain and close
    def abort(self) -> None: ...  # drop buffered audio and close


class _SoundDeviceSink:
    """A sounddevice OutputStream kept open across replies.

    Starting a CoreAudio stream costs 0.1-0.6 s on macOS, so the stream is
    opened once (in ``warm_up``) and reused: between replies nothing is
    written and the device plays silence. Stopping just stops writing, so at
    most the ~50 ms output buffer is still heard.
    """

    def __init__(self) -> None:
        self._stream: Any = None
        self._rate = 0

    def start(self, sample_rate: int) -> None:
        stream = self._stream
        if stream is not None and self._rate == sample_rate and stream.active:
            return
        self.close()
        import sounddevice as sd

        t0 = time.perf_counter()
        stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            latency=_OUTPUT_LATENCY_S,
        )
        stream.start()
        self._stream, self._rate = stream, sample_rate
        logger.debug(
            "Output stream opened in %.0f ms", 1000 * (time.perf_counter() - t0)
        )

    def write(self, block: np.ndarray) -> None:
        if self._stream is None:
            return
        try:
            self._stream.write(block.reshape(-1, 1))
        except Exception:
            logger.warning("Output stream failed, reopening", exc_info=True)
            rate = self._rate
            self.close()
            self.start(rate)
            self._stream.write(block.reshape(-1, 1))

    def finish(self) -> None:
        # Writes return once data is buffered: wait for the buffer to play.
        if self._stream is not None:
            time.sleep(float(self._stream.latency) + 0.02)

    def abort(self) -> None:
        pass  # stop writing; the short buffer drains by itself

    def close(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return

        def run() -> None:  # CoreAudio abort/close blocks ~0.1 s
            try:
                stream.abort()
                stream.close()
            except Exception:
                logger.debug("Closing output stream failed", exc_info=True)

        threading.Thread(target=run, name="orb-voice-close", daemon=True).start()


# --------------------------------------------------------------------------
# Voice
# --------------------------------------------------------------------------


def _voice_url_path(name: str) -> str:
    """``fr_FR-tom-medium`` -> ``fr/fr_FR/tom/medium/fr_FR-tom-medium``."""
    try:
        locale, speaker, quality = name.split("-", 2)
    except ValueError as exc:
        raise ValueError(f"invalid piper voice name: {name!r}") from exc
    lang = locale.split("_")[0]
    return f"{lang}/{locale}/{speaker}/{quality}/{name}"


def split_voice_name(spec: str) -> tuple[str, str | None]:
    """``fr_FR-upmc-medium#pierre`` -> ``("fr_FR-upmc-medium", "pierre")``."""
    name, sep, speaker = (spec or "").strip().partition("#")
    return name.strip(), (speaker.strip() or None) if sep else None


def resolve_speaker(
    speaker: str | None, speaker_id_map: dict[str, int], num_speakers: int
) -> int | None:
    """Speaker id for a multi-speaker voice (map name first, then numeric id)."""
    if num_speakers <= 1:
        if speaker:
            logger.warning("Voice has a single speaker; ignoring %r", speaker)
        return None
    if not speaker:
        return 0
    if speaker in speaker_id_map:
        return int(speaker_id_map[speaker])
    lowered = {k.lower(): v for k, v in speaker_id_map.items()}
    if speaker.lower() in lowered:
        return int(lowered[speaker.lower()])
    if speaker.isdigit() and int(speaker) < num_speakers:
        return int(speaker)
    logger.warning(
        "Unknown speaker %r (known: %s); using speaker 0",
        speaker,
        ", ".join(list(speaker_id_map)[:10]) or f"0..{num_speakers - 1}",
    )
    return 0


def _download(url: str, dest: Path) -> None:
    import httpx
    import truststore

    truststore.inject_into_ssl()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    logger.info("Downloading %s", url)
    with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as fh:
            for data in resp.iter_bytes(1 << 16):
                fh.write(data)
    os.replace(tmp, dest)


class Voice:
    def __init__(self, cfg: VoiceConfig):
        self.cfg = cfg
        self._engine = (cfg.engine or "piper").strip().lower()
        self._piper: Any = None
        self._syn_config: Any = None
        self._sample_rate = _SAY_SAMPLE_RATE
        self._load_lock = threading.Lock()
        self._loaded = False
        self._sink: _Sink | None = None  # test override (else sounddevice)
        self._output: _SoundDeviceSink | None = None
        self._speak_lock = threading.Lock()
        self._speed = max(0.3, min(3.0, float(cfg.speed or 1.0)))
        self._fx = voice_fx.design("none", self._sample_rate)

    # -- loading -----------------------------------------------------------

    @property
    def engine(self) -> str:
        return self._engine

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def _ensure_loaded(self) -> None:
        with self._load_lock:
            if self._loaded:
                return
            if self._engine == "piper":
                try:
                    self._load_piper()
                except Exception:
                    logger.exception("Piper unavailable, falling back to macOS 'say'")
                    self._engine = "say"
            if self._engine != "piper":
                self._engine = "say"
                self._sample_rate = _SAY_SAMPLE_RATE
            self._load_fx()
            self._loaded = True

    def _load_fx(self) -> None:
        try:
            self._fx = voice_fx.design(self.cfg.effect, self._sample_rate)
            self._fx.new()  # import scipy now, not on the first reply
        except Exception:
            logger.exception("Voice effect %r unavailable", self.cfg.effect)
            self._fx = voice_fx.design("none", self._sample_rate)
        if self._piper is not None:
            # The pitch drop stretches the audio: speak faster to compensate.
            self._syn_config.length_scale = self._fx.pitch_ratio / self._speed
        if self._fx.enabled:
            logger.info("Voice effect: %s", self._fx.name)

    def _load_piper(self) -> None:
        from piper import PiperVoice, SynthesisConfig

        name, speaker = split_voice_name(self.cfg.piper_voice)
        model = PIPER_DIR / f"{name}.onnx"
        config = PIPER_DIR / f"{name}.onnx.json"
        base = f"{_HF_BASE}/{_voice_url_path(name)}"
        if not config.exists():
            _download(f"{base}.onnx.json?download=true", config)
        if not model.exists():
            _download(f"{base}.onnx?download=true", model)
        t0 = time.perf_counter()
        self._piper = PiperVoice.load(model, config_path=config)
        pcfg = self._piper.config
        speaker_id = resolve_speaker(
            speaker, dict(pcfg.speaker_id_map or {}), int(pcfg.num_speakers or 1)
        )
        self._syn_config = SynthesisConfig(
            speaker_id=speaker_id, length_scale=1.0 / self._speed
        )
        self._sample_rate = int(pcfg.sample_rate)
        logger.info(
            "Piper voice %s%s loaded in %.2fs (%d Hz)",
            name,
            "" if speaker_id is None else f" (speaker {speaker_id})",
            time.perf_counter() - t0,
            self._sample_rate,
        )

    def warm_up(self) -> None:
        """Load (and download if needed) the voice; prime synthesis and audio."""
        self._ensure_loaded()
        t0 = time.perf_counter()
        for _ in self._synthesize("Bonjour."):
            pass
        if self._sink is None:
            try:
                self._default_sink().start(self._sample_rate)
            except Exception:
                logger.exception("Audio output unavailable")
        logger.info(
            "Voice warm-up (%s) done in %.2fs", self._engine, time.perf_counter() - t0
        )

    # -- synthesis ---------------------------------------------------------

    def _synthesize(self, text: str) -> Iterable[np.ndarray]:
        """Yield float32 mono chunks at ``self.sample_rate`` for ``text``."""
        if self._engine == "piper":
            try:
                for chunk in self._piper.synthesize(text, self._syn_config):
                    yield chunk.audio_float_array
                return
            except Exception:
                logger.exception("Piper synthesis failed for %r", text)
                if self._sample_rate != _SAY_SAMPLE_RATE:
                    return  # stream already open at piper's rate; skip sentence
        yield from self._synthesize_say(text)

    def _synthesize_say(self, text: str) -> Iterable[np.ndarray]:
        import numpy as np

        rate = int(_SAY_BASE_RATE * self._speed / self._fx.pitch_ratio)
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="orb-say-")
        os.close(fd)
        try:
            subprocess.run(
                [
                    "say",
                    "-v",
                    self.cfg.say_voice,
                    "-r",
                    str(rate),
                    "-o",
                    path,
                    f"--data-format=LEI16@{self._sample_rate}",
                    "--",
                    text,
                ],
                check=True,
                capture_output=True,
                timeout=60,
            )
            with wave.open(path, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                channels = wf.getnchannels()
            audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
            if channels > 1:
                audio = audio.reshape(-1, channels).mean(axis=1)
            yield audio
        except Exception:
            logger.exception("'say' synthesis failed for %r", text)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def _default_sink(self) -> _SoundDeviceSink:
        if self._output is None:
            self._output = _SoundDeviceSink()
        return self._output

    def close(self) -> None:
        """Release the audio output (optional; the stream is reopened on demand)."""
        if self._output is not None:
            self._output.close()

    # -- public API ----------------------------------------------------------

    def speak(
        self,
        text: str,
        stop: threading.Event,
        level_cb: Callable[[float], None] | None = None,
    ) -> None:
        self.speak_stream([text], stop, level_cb=level_cb)

    def speak_stream(
        self,
        chunks: Iterable[str],
        stop: threading.Event,
        level_cb: Callable[[float], None] | None = None,
        on_first_audio: Callable[[], None] | None = None,
    ) -> str:
        """Speak streamed text; returns the (cleaned) text actually spoken."""
        self._ensure_loaded()
        with self._speak_lock:  # one reply at a time on the shared stream
            return self._speak_stream(chunks, stop, level_cb, on_first_audio)

    def _speak_stream(
        self,
        chunks: Iterable[str],
        stop: threading.Event,
        level_cb: Callable[[float], None] | None,
        on_first_audio: Callable[[], None] | None,
    ) -> str:
        import numpy as np

        sentences: queue.Queue[str | None] = queue.Queue()
        audio_q: queue.Queue[tuple[str, np.ndarray] | None] = queue.Queue(maxsize=64)
        done = threading.Event()  # consumer finished / aborted

        def put_audio(item: tuple[str, np.ndarray] | None) -> bool:
            while not (stop.is_set() or done.is_set()):
                try:
                    audio_q.put(item, timeout=0.05)
                    return True
                except queue.Full:
                    continue
            return False

        def text_worker() -> None:
            splitter = SentenceSplitter()
            it = iter(chunks)
            try:
                for delta in it:
                    if stop.is_set() or done.is_set():
                        break
                    if delta:
                        for s in splitter.feed(delta):
                            sentences.put(s)
                else:
                    for s in splitter.flush():
                        sentences.put(s)
            except Exception:
                logger.exception("Text stream failed")
                if not stop.is_set():
                    for s in splitter.flush():
                        sentences.put(s)
            finally:
                close = getattr(it, "close", None)
                if close is not None and (stop.is_set() or done.is_set()):
                    try:
                        close()
                    except Exception:
                        pass
                sentences.put(None)

        def synth_worker() -> None:
            gap = np.zeros(int(self._sample_rate * _SENTENCE_GAP_S), dtype=np.float32)
            fx = self._fx.new()  # fresh filter state for each reply
            finished = False
            try:
                while not (stop.is_set() or done.is_set()):
                    try:
                        raw = sentences.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    if raw is None:
                        break
                    text = clean_text(raw)
                    if _alpha_count(text) == 0:
                        continue
                    first = True
                    for audio in self._synthesize(text):
                        if stop.is_set() or done.is_set():
                            return
                        audio = fx.process(audio)
                        if not put_audio((text if first else "", audio)):
                            return
                        first = False
                    if not first and not put_audio(("", fx.process(gap))):
                        return
                finished = True
            finally:
                if finished and fx.enabled:
                    put_audio(("", fx.flush()))
                put_audio(None)

        threads = [
            threading.Thread(target=text_worker, name="orb-voice-text", daemon=True),
            threading.Thread(target=synth_worker, name="orb-voice-synth", daemon=True),
        ]
        for t in threads:
            t.start()

        spoken: list[str] = []
        sink: _Sink = self._sink if self._sink is not None else self._default_sink()
        block = max(64, self._sample_rate // _BLOCKS_PER_SECOND)
        level = 0.0
        started = False
        aborted = False

        def emit_level(value: float) -> None:
            if level_cb is not None:
                try:
                    level_cb(value)
                except Exception:
                    logger.exception("level_cb failed")

        try:
            sink.start(self._sample_rate)
            while True:
                if stop.is_set():
                    aborted = True
                    break
                try:
                    item = audio_q.get(timeout=0.03)
                except queue.Empty:
                    if started and level > 0.0:
                        level = 0.0
                        emit_level(0.0)
                    continue
                if item is None:
                    break
                text, audio = item
                if text:
                    spoken.append(text)
                for pos in range(0, len(audio), block):
                    if stop.is_set():
                        aborted = True
                        break
                    frame = audio[pos : pos + block]
                    if not started:
                        started = True
                        if on_first_audio is not None:
                            try:
                                on_first_audio()
                            except Exception:
                                logger.exception("on_first_audio failed")
                    rms = float(np.sqrt(np.mean(frame * frame))) if len(frame) else 0.0
                    target = min(1.0, math.pow(rms / 0.22, 0.8)) if rms > 1e-4 else 0.0
                    # fast attack, smooth release
                    level = target if target > level else level * 0.6 + target * 0.4
                    emit_level(level)
                    sink.write(frame)
                if aborted:
                    break
        except Exception:
            logger.exception("Audio playback failed")
            aborted = True
        finally:
            done.set()
            try:
                if aborted:
                    sink.abort()
                else:
                    sink.finish()
            except Exception:
                logger.exception("Closing audio output failed")
            emit_level(0.0)
            if not aborted:  # on abort, workers exit on their own (daemon)
                for t in threads:
                    t.join(timeout=0.5)

        return " ".join(spoken)
