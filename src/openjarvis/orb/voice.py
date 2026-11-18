"""Text-to-speech for the orb: streamed, sentence-by-sentence, low latency.

``Voice.speak_stream`` consumes LLM text deltas, cuts them into sentences as
soon as possible, synthesizes them on a worker thread and plays them back to
back on a single sounddevice output stream, so the synthesis of sentence N+1
overlaps the playback of sentence N.

Engines:
- ``piper`` (default): local neural TTS. The voice is downloaded once from
  Hugging Face (``rhasspy/piper-voices``) into ``DATA_DIR/piper/``.
- ``say``: macOS built-in voices. Also used automatically when piper fails.
- ``edge``: Microsoft Edge "Read aloud" neural voices (``edge-tts``; online,
  free, no key). ``cfg.edge_voice`` is a short name such as
  ``fr-FR-RemyMultilingualNeural``. The text is sent to Microsoft.
- ``elevenlabs``: ElevenLabs streaming API (online, paid, best quality). The
  key comes from ``$ELEVENLABS_API_KEY`` or ``~/.openjarvis/elevenlabs.key``;
  ``cfg.elevenlabs_voice`` is a voice id.

The online engines request the next sentences while the current one plays,
trim the long silences the services put around each sentence, and fall back
to piper (then ``say``) when the network fails, without interrupting a reply.

A multi-speaker piper voice takes its speaker after a ``#``:
``fr_FR-upmc-medium#pierre`` (a name of the voice's speaker map) or
``fr_FR-mls-medium#36`` (a numeric speaker id, when it is not a map name).

``cfg.effect`` names a post-processing preset of ``voice_fx`` (``none``,
``jarvis``...) for the local engines, applied to each synthesized chunk with
state kept across the chunks of a reply. The online voices are played as is
unless ``cfg.online_effect`` names a preset.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
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
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
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

_NET_ENGINES = ("edge", "elevenlabs")
_NET_SAMPLE_RATE = 24000  # edge: 24 kHz MP3; elevenlabs: pcm_24000
_NET_LOOKAHEAD = 2  # sentences requested ahead of the one playing
_NET_RETRY_AFTER_S = 30.0  # after a network failure, use the fallback this long
_DEFAULT_EDGE_VOICE = "fr-FR-RemyMultilingualNeural"
_DEFAULT_ELEVENLABS_VOICE = "JBFqnCBsd6RMkjVDRZzb"  # "George", premade, multilingual
_DEFAULT_ELEVENLABS_MODEL = "eleven_flash_v2_5"  # lowest latency
_ELEVENLABS_API = "https://api.elevenlabs.io"
ELEVENLABS_KEY_FILE = Path.home() / ".openjarvis" / "elevenlabs.key"
# The services pad each sentence with ~0.2 s of silence before and ~0.8 s
# after: keep only short margins so the voice keeps a brisk pace.
_LEAD_KEEP_S = 0.03
_TRAIL_KEEP_S = 0.14
_SILENCE_LEVEL = 0.006
# piper peaks near full scale, ~3 dB above the online voices: level-match it
# when it stands in for them, so a fallback sentence does not jump out.
_FALLBACK_GAIN = 0.7
_ONLINE_ENGINES = frozenset({"edge", "elevenlabs"})


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


# --------------------------------------------------------------------------
# Online engines
# --------------------------------------------------------------------------


class EngineError(RuntimeError):
    """An online engine is unavailable or a request failed."""


def speed_to_rate(speed: float) -> str:
    """Speech-rate multiplier -> SSML prosody rate: ``1.15`` -> ``"+15%"``."""
    pct = round((float(speed) - 1.0) * 100)
    return f"{max(-50, min(100, pct)):+d}%"


def normalize_pitch(pitch: object) -> str:
    """``"-2Hz"`` / ``"+0Hz"`` as edge-tts expects; anything else -> ``"+0Hz"``."""
    text = str(pitch or "").strip().replace(" ", "")
    m = re.fullmatch(r"([+-]?)(\d+)(?:hz|Hz|HZ)?", text)
    if not m:
        if text:
            logger.warning("Invalid voice pitch %r (expected e.g. '-3Hz')", pitch)
        return "+0Hz"
    return f"{m.group(1) or '+'}{m.group(2)}Hz"


def _resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst or not len(audio):
        return audio
    import numpy as np
    from scipy.signal import resample_poly

    g = math.gcd(src, dst)
    out = resample_poly(audio, dst // g, src // g).astype("float32")
    return np.clip(out, -1.0, 1.0, out=out)


class SilenceTrimmer:
    """Drop the silence a service pads a sentence with, keeping short margins.

    Streaming-friendly: speech is passed on as soon as it arrives; only a
    silent tail is held back (released if speech resumes, else cut to
    ``trail_keep`` seconds by ``flush``).
    """

    def __init__(
        self,
        sample_rate: int,
        lead_keep: float = _LEAD_KEEP_S,
        trail_keep: float = _TRAIL_KEEP_S,
        threshold: float = _SILENCE_LEVEL,
    ) -> None:
        import numpy as np

        self._np = np
        self._lead = int(sample_rate * lead_keep)
        self._trail = int(sample_rate * trail_keep)
        self._threshold = threshold
        self._held = np.zeros(0, dtype=np.float32)
        self._started = False

    def process(self, audio: np.ndarray) -> np.ndarray:
        np = self._np
        buf = np.concatenate([self._held, audio]) if len(self._held) else audio
        loud = np.flatnonzero(np.abs(buf) > self._threshold)
        if not self._started:
            if not loud.size:
                self._held = buf[-self._lead :] if self._lead else buf[:0]
                return buf[:0]
            self._started = True
            cut = max(0, int(loud[0]) - self._lead)
            buf, loud = buf[cut:], loud - cut
        if not loud.size:
            self._held = buf
            return buf[:0]
        end = int(loud[-1]) + 1
        self._held = buf[end:]
        return buf[:end]

    def flush(self) -> np.ndarray:
        tail = self._held[: self._trail] if self._started else self._held[:0]
        self._held = self._held[:0]
        return tail


_END = object()


class SpeechJob:
    """Audio of one sentence, produced in the background by an online engine."""

    def __init__(self) -> None:
        self._q: queue.Queue[Any] = queue.Queue()
        self.cancelled = threading.Event()
        self._on_cancel: Callable[[], None] | None = None

    # producer side (any thread)
    def put(self, audio: np.ndarray) -> None:
        if len(audio):
            self._q.put(audio)

    def end(self, error: BaseException | None = None) -> None:
        self._q.put(error if error is not None else _END)

    # consumer side
    def chunks(
        self,
        halted: Callable[[], bool],
        idle: Callable[[], None] | None = None,
        timeout: float = 20.0,
    ) -> Iterator[np.ndarray]:
        """Yield float32 chunks; raise ``EngineError`` if the request failed."""
        deadline = time.monotonic() + timeout
        while not halted():
            if idle is not None:
                idle()
            try:
                item = self._q.get(timeout=0.04)
            except queue.Empty:
                if time.monotonic() > deadline:
                    self.cancel()
                    raise EngineError("timed out") from None
                continue
            if item is _END:
                return
            if isinstance(item, BaseException):
                raise item
            deadline = time.monotonic() + timeout
            yield item

    def cancel(self) -> None:
        self.cancelled.set()
        cb, self._on_cancel = self._on_cancel, None
        if cb is not None:
            try:
                cb()
            except Exception:
                logger.debug("Cancelling a speech request failed", exc_info=True)


class _NetEngine(Protocol):
    name: str
    sample_rate: int

    def start(self, text: str) -> SpeechJob: ...
    def prepare(self, connections: int = 1) -> None: ...
    def close(self) -> None: ...


def _pcm16_to_float(data: bytes) -> np.ndarray:
    import numpy as np

    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def _frames_to_float(frames: Iterable[Any]) -> list[np.ndarray]:
    import numpy as np

    out = []
    for frame in frames:
        a = frame.to_ndarray()
        if a.dtype == np.int16:
            a = a.astype(np.float32) / 32768.0
        else:
            a = a.astype(np.float32, copy=False)
        if a.ndim == 2:  # planar (channels, n) or packed (1, n * channels)
            channels = len(frame.layout.channels)
            a = a.mean(axis=0) if a.shape[0] > 1 else a.reshape(-1)
            if a.shape[0] != frame.samples and channels > 1:
                a = a.reshape(-1, channels).mean(axis=1)
        out.append(a)
    return out


class Mp3Decoder:
    """Incremental MP3 -> float32 mono decoder (PyAV), fed arbitrary byte chunks."""

    def __init__(self) -> None:
        import av

        self._codec = av.CodecContext.create("mp3", "r")

    def feed(self, data: bytes) -> list[np.ndarray]:
        frames = []
        for packet in self._codec.parse(data):
            try:
                frames.extend(self._codec.decode(packet))
            except Exception:  # a tag / damaged frame: skip it, keep going
                logger.debug("Skipping an undecodable MP3 packet", exc_info=True)
        return _frames_to_float(frames)

    def flush(self) -> list[np.ndarray]:
        frames = []
        try:
            for packet in self._codec.parse(b""):
                frames.extend(self._codec.decode(packet))
            frames.extend(self._codec.decode(None))
        except Exception:
            logger.debug("MP3 decoder flush failed", exc_info=True)
        return _frames_to_float(frames)


def _tls_context() -> Any:
    """System trust store (the user's network intercepts TLS)."""
    import ssl

    import truststore

    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


@dataclass
class _EdgeConn:
    ws: Any
    last_used: float


class EdgeEngine:
    """Microsoft Edge neural voices through the edge-tts protocol.

    edge-tts opens one websocket (TLS + handshake, ~0.5 s) per text. Here
    the websockets are kept and reused for the following sentences (the
    service closes them after ~30 s idle), and one is opened as soon as a
    reply starts, while the LLM is still writing the first sentence. Every
    request runs on one private asyncio loop thread.
    """

    name = "edge"
    sample_rate = _NET_SAMPLE_RATE
    idle_s = 18.0  # reuse a websocket idle for less than this
    connect_timeout = 5.0
    receive_timeout = 8.0

    def __init__(
        self,
        voice: str = _DEFAULT_EDGE_VOICE,
        rate: str = "+0%",
        pitch: str = "+0Hz",
        connect: Callable[[], Any] | None = None,
    ) -> None:
        from edge_tts.data_classes import TTSConfig

        self.voice = voice
        self._tc = TTSConfig(voice, rate, "+0%", pitch, "SentenceBoundary")
        self._connect_fn = connect  # test hook: async () -> websocket-like
        self._session: Any = None
        self._ssl: Any = None
        self._spares: list[asyncio.Future[_EdgeConn | None]] = []
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="orb-voice-edge", daemon=True
        )
        self._thread.start()

    # -- thread-safe API ---------------------------------------------------

    def start(self, text: str) -> SpeechJob:
        job = SpeechJob()
        fut = asyncio.run_coroutine_threadsafe(self._request(text, job), self._loop)
        job._on_cancel = fut.cancel
        return job

    def prepare(self, connections: int = 1) -> None:
        """Open websockets in advance so the next requests skip the handshake."""
        self._loop.call_soon_threadsafe(self._prefill, connections)

    def close(self) -> None:
        async def shutdown() -> None:
            for fut in self._spares:
                if fut.done() and not fut.cancelled() and fut.result() is not None:
                    await _close_ws(fut.result().ws)
                else:
                    fut.cancel()
            self._spares.clear()
            if self._session is not None:
                await self._session.close()

        if self._loop.is_closed():
            return
        try:
            asyncio.run_coroutine_threadsafe(shutdown(), self._loop).result(2.0)
        except Exception:
            logger.debug("Closing edge-tts connections failed", exc_info=True)
        self._loop.call_soon_threadsafe(self._loop.stop)

    # -- loop side ---------------------------------------------------------

    def _prefill(self, connections: int) -> None:
        now = time.monotonic()
        usable = 0
        for fut in self._spares:
            if not fut.done():
                usable += 1
            elif not fut.cancelled() and fut.result() is not None:
                conn = fut.result()
                usable += int(not conn.ws.closed and now - conn.last_used < self.idle_s)
        for _ in range(max(0, connections - usable)):
            self._spares.append(asyncio.ensure_future(self._try_connect()))

    async def _try_connect(self) -> _EdgeConn | None:
        try:
            return await self._connect()
        except Exception as exc:
            logger.debug("edge-tts pre-connection failed: %s", type(exc).__name__)
            return None

    async def _connect(self) -> _EdgeConn:
        if self._connect_fn is not None:
            ws = await self._connect_fn()
        else:
            ws = await asyncio.wait_for(self._ws_connect(), self.connect_timeout)
        word_boundary = self._tc.boundary == "WordBoundary"
        await ws.send_str(
            f"X-Timestamp:{_edge_date()}\r\n"
            "Content-Type:application/json; charset=utf-8\r\n"
            "Path:speech.config\r\n\r\n"
            '{"context":{"synthesis":{"audio":{"metadataoptions":{'
            f'"sentenceBoundaryEnabled":"{str(not word_boundary).lower()}",'
            f'"wordBoundaryEnabled":"{str(word_boundary).lower()}"'
            '},"outputFormat":"audio-24khz-48kbitrate-mono-mp3"}}}}\r\n'
        )
        return _EdgeConn(ws, time.monotonic())

    async def _ws_connect(self) -> Any:
        import aiohttp
        from edge_tts.communicate import connect_id
        from edge_tts.constants import SEC_MS_GEC_VERSION, WSS_HEADERS, WSS_URL
        from edge_tts.drm import DRM

        if self._session is None or self._session.closed:
            self._ssl = _tls_context()
            self._session = aiohttp.ClientSession(trust_env=True)
        for attempt in range(2):
            try:
                return await self._session.ws_connect(
                    f"{WSS_URL}&ConnectionId={connect_id()}"
                    f"&Sec-MS-GEC={DRM.generate_sec_ms_gec()}"
                    f"&Sec-MS-GEC-Version={SEC_MS_GEC_VERSION}",
                    compress=15,
                    headers=DRM.headers_with_muid(WSS_HEADERS),
                    ssl=self._ssl,
                    heartbeat=None,
                )
            except aiohttp.ClientResponseError as exc:
                if exc.status != 403 or attempt:
                    raise
                DRM.handle_client_response_error(exc)  # clock skew: retry once
        raise EngineError("edge-tts connection refused")

    async def _acquire(self) -> _EdgeConn:
        """A ready connection: an idle one, a pending pre-connection, or a new one."""
        self._spares.sort(key=lambda f: not f.done())  # ready ones first
        while self._spares:
            fut = self._spares.pop(0)
            try:
                conn = await asyncio.shield(fut)
            except asyncio.CancelledError:
                if not fut.cancelled():  # we were cancelled, not the spare
                    self._spares.append(fut)
                    raise
                continue
            if conn is None:
                continue
            if conn.ws.closed or time.monotonic() - conn.last_used > self.idle_s:
                await _close_ws(conn.ws)
                continue
            return conn
        return await self._connect()

    def _release(self, conn: _EdgeConn) -> None:
        conn.last_used = time.monotonic()
        fut: asyncio.Future[_EdgeConn | None] = self._loop.create_future()
        fut.set_result(conn)
        self._spares.append(fut)
        while len(self._spares) > 4:
            old = self._spares.pop(0)
            if old.done() and not old.cancelled() and old.result() is not None:
                asyncio.ensure_future(_close_ws(old.result().ws))

    async def _request(self, text: str, job: SpeechJob) -> None:
        trimmer = SilenceTrimmer(self.sample_rate)
        got = [False]
        try:
            for attempt in range(2):
                conn = await self._acquire()
                try:
                    await self._turn(conn.ws, text, job, trimmer, got)
                except asyncio.CancelledError:
                    await _close_ws(conn.ws)
                    raise
                except Exception:
                    await _close_ws(conn.ws)
                    if got[0] or attempt:
                        raise
                    continue  # a stale reused websocket: retry on a new one
                self._release(conn)
                break
            job.put(trimmer.flush())
            job.end()
        except asyncio.CancelledError:
            job.end()
            raise
        except Exception as exc:
            job.end(EngineError(f"edge-tts: {type(exc).__name__}: {exc}"[:200]))

    async def _turn(
        self,
        ws: Any,
        text: str,
        job: SpeechJob,
        trimmer: SilenceTrimmer,
        got: list[bool],
    ) -> None:
        import aiohttp
        import numpy as np
        from edge_tts.communicate import (
            connect_id,
            mkssml,
            remove_incompatible_characters,
            ssml_headers_plus_data,
        )

        safe = _xml_escape(remove_incompatible_characters(text))
        await ws.send_str(
            ssml_headers_plus_data(connect_id(), _edge_date(), mkssml(self._tc, safe))
        )
        decoder = Mp3Decoder()

        def emit(parts: list[np.ndarray]) -> None:
            if parts:
                out = trimmer.process(np.concatenate(parts))
                if len(out):
                    got[0] = True
                    job.put(out)

        while True:
            msg = await ws.receive(timeout=self.receive_timeout)
            if msg.type == aiohttp.WSMsgType.TEXT:
                head = msg.data.split("\r\n\r\n", 1)[0]
                if "Path:turn.end" in head:
                    break
            elif msg.type == aiohttp.WSMsgType.BINARY:
                data = msg.data
                if len(data) < 2:
                    raise EngineError("malformed audio message")
                header_len = int.from_bytes(data[:2], "big")
                payload = data[2 + header_len :]
                if payload:
                    emit(decoder.feed(payload))
            else:
                raise EngineError(f"websocket {msg.type.name.lower()}")
            if job.cancelled.is_set():
                raise asyncio.CancelledError
        emit(decoder.flush())
        if not got[0]:
            raise EngineError("no audio received")


async def _close_ws(ws: Any) -> None:
    try:
        await asyncio.wait_for(ws.close(), 1.0)
    except Exception:
        pass


def _edge_date() -> str:
    from edge_tts.communicate import date_to_string

    return date_to_string()


def _xml_escape(text: str) -> str:
    from xml.sax.saxutils import escape

    return escape(text)


def load_elevenlabs_key() -> str:
    """``$ELEVENLABS_API_KEY`` or ``~/.openjarvis/elevenlabs.key`` (never logged)."""
    key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if not key:
        try:
            key = ELEVENLABS_KEY_FILE.read_text(encoding="utf-8").strip()
        except OSError:
            key = ""
    if key and any(not 33 <= ord(ch) <= 126 for ch in key):
        logger.warning("The ElevenLabs API key contains invalid characters; ignored")
        return ""
    return key


class ElevenLabsEngine:
    """ElevenLabs HTTP streaming TTS, raw 16-bit PCM at 24 kHz.

    One HTTP/1.1 keep-alive client is shared by up to three request threads,
    so after the first sentence the TLS handshake is not paid again.
    """

    name = "elevenlabs"
    sample_rate = _NET_SAMPLE_RATE

    def __init__(
        self,
        voice_id: str = _DEFAULT_ELEVENLABS_VOICE,
        speed: float = 1.0,
        model: str = _DEFAULT_ELEVENLABS_MODEL,
        language: str = "fr",
        api_key: str | None = None,
        transport: Any = None,
    ) -> None:
        import httpx

        key = api_key if api_key is not None else load_elevenlabs_key()
        if not key:
            raise EngineError(
                f"no ElevenLabs API key ($ELEVENLABS_API_KEY or {ELEVENLABS_KEY_FILE})"
            )
        self.voice_id = voice_id or _DEFAULT_ELEVENLABS_VOICE
        self.model = model or _DEFAULT_ELEVENLABS_MODEL
        self.language = language
        self.speed = max(0.7, min(1.2, float(speed)))  # API limits
        self._client = httpx.Client(
            base_url=_ELEVENLABS_API,
            headers={"xi-api-key": key},
            timeout=httpx.Timeout(10.0, connect=5.0),
            transport=transport,
            verify=_tls_context() if transport is None else True,
        )
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=_NET_LOOKAHEAD + 1, thread_name_prefix="orb-voice-11labs"
        )
        self._previous = ""

    def build_request(self, text: str, previous_text: str = "") -> Any:
        body: dict[str, Any] = {
            "text": text,
            "model_id": self.model,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "speed": self.speed,
            },
        }
        if self.language and ("v2_5" in self.model or "v3" in self.model):
            body["language_code"] = self.language
        if previous_text:
            body["previous_text"] = previous_text[-300:]
        return self._client.build_request(
            "POST",
            f"/v1/text-to-speech/{self.voice_id}/stream",
            params={"output_format": f"pcm_{self.sample_rate}"},
            json=body,
        )

    def start(self, text: str) -> SpeechJob:
        job = SpeechJob()
        request = self.build_request(text, self._previous)
        self._previous = text
        self._pool.submit(self._run, request, job)
        return job

    def prepare(self, connections: int = 1) -> None:
        pass  # httpx keeps the connection alive between requests

    def _run(self, request: Any, job: SpeechJob) -> None:
        trimmer = SilenceTrimmer(self.sample_rate)
        carry = b""
        try:
            response = self._client.send(request, stream=True)
            try:
                if response.status_code != 200:
                    raise EngineError(f"ElevenLabs HTTP {response.status_code}")
                for data in response.iter_bytes(4800):
                    if job.cancelled.is_set():
                        break
                    data = carry + data
                    cut = len(data) - len(data) % 2
                    carry = data[cut:]
                    job.put(trimmer.process(_pcm16_to_float(data[:cut])))
            finally:
                response.close()
            job.put(trimmer.flush())
            job.end()
        except EngineError as exc:
            job.end(exc)
        except Exception as exc:  # never echo provider details (may hold headers)
            job.end(EngineError(f"ElevenLabs request failed ({type(exc).__name__})"))

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._client.close()


class Voice:
    def __init__(self, cfg: VoiceConfig):
        self.cfg = cfg
        self._engine = (cfg.engine or "piper").strip().lower()
        if self._engine in _ONLINE_ENGINES and getattr(cfg, "local_only", True):
            # Privacy guard: the user wants speech synthesis to never leave
            # the Mac. Online engines need `local_only = false` explicitly.
            logger.warning(
                "Voice engine %r sends text to a server and local_only is set; "
                "using piper instead",
                self._engine,
            )
            self._engine = "piper"
        self._piper: Any = None
        self._syn_config: Any = None
        self._piper_rate = 0
        self._sample_rate = _SAY_SAMPLE_RATE
        self._load_lock = threading.Lock()
        self._loaded = False
        self._sink: _Sink | None = None  # test override (else sounddevice)
        self._output: _SoundDeviceSink | None = None
        self._speak_lock = threading.Lock()
        self._speed = max(0.3, min(3.0, float(cfg.speed or 1.0)))
        self._fx = voice_fx.design("none", self._sample_rate)
        self._net: _NetEngine | None = None
        self._net_down_until = 0.0
        self._fallback_thread: threading.Thread | None = None

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
            if self._engine in _NET_ENGINES:
                self._sample_rate = _NET_SAMPLE_RATE
                self._load_fx()
                try:
                    self._net = self._make_net_engine()
                    logger.info(
                        "Voice engine %s (%s)",
                        self._engine,
                        getattr(self._net, "voice", "")
                        or getattr(self._net, "voice_id", ""),
                    )
                except Exception as exc:
                    logger.warning(
                        "Voice engine %r unavailable (%s); falling back to piper",
                        self._engine,
                        exc,
                    )
                    self._engine = "piper"
            if self._engine == "piper":
                try:
                    self._load_piper()
                    self._sample_rate = self._piper_rate
                except Exception:
                    logger.exception("Piper unavailable, falling back to macOS 'say'")
                    self._engine = "say"
            if self._engine not in ("piper", *_NET_ENGINES):
                if self._engine != "say":
                    logger.warning("Unknown voice engine %r; using 'say'", self._engine)
                self._engine = "say"
                self._sample_rate = _SAY_SAMPLE_RATE
            if self._net is None:
                self._load_fx()
            self._loaded = True

    def _make_net_engine(self) -> _NetEngine:
        rate = self._speed / self._fx.pitch_ratio  # a pitch drop slows it down
        if self._engine == "edge":
            return EdgeEngine(
                voice=str(getattr(self.cfg, "edge_voice", "") or _DEFAULT_EDGE_VOICE),
                rate=speed_to_rate(rate),
                pitch=normalize_pitch(getattr(self.cfg, "pitch", "+0Hz")),
            )
        return ElevenLabsEngine(
            voice_id=str(getattr(self.cfg, "elevenlabs_voice", "") or ""),
            model=str(getattr(self.cfg, "elevenlabs_model", "") or ""),
            speed=rate,
        )

    def _effect_name(self) -> str:
        if self._engine in _NET_ENGINES:
            return str(getattr(self.cfg, "online_effect", "none") or "none")
        return self.cfg.effect

    def _load_fx(self) -> None:
        name = self._effect_name()
        try:
            self._fx = voice_fx.design(name, self._sample_rate)
            self._fx.new()  # import scipy now, not on the first reply
        except Exception:
            logger.exception("Voice effect %r unavailable", name)
            self._fx = voice_fx.design("none", self._sample_rate)
        if self._syn_config is not None:
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
        piper = PiperVoice.load(model, config_path=config)
        pcfg = piper.config
        speaker_id = resolve_speaker(
            speaker, dict(pcfg.speaker_id_map or {}), int(pcfg.num_speakers or 1)
        )
        self._syn_config = SynthesisConfig(
            speaker_id=speaker_id, length_scale=self._fx.pitch_ratio / self._speed
        )
        self._piper_rate = int(pcfg.sample_rate)
        self._piper = piper
        logger.info(
            "Piper voice %s%s loaded in %.2fs (%d Hz)",
            name,
            "" if speaker_id is None else f" (speaker {speaker_id})",
            time.perf_counter() - t0,
            self._piper_rate,
        )

    def _load_fallback_async(self) -> None:
        """Online engine: load piper in the background, ready if the network fails."""
        if self._piper is not None or self._fallback_thread is not None:
            return

        def run() -> None:
            try:
                self._load_piper()
            except Exception as exc:
                logger.warning("Fallback piper voice unavailable (%s)", exc)

        self._fallback_thread = threading.Thread(
            target=run, name="orb-voice-fallback", daemon=True
        )
        self._fallback_thread.start()

    def warm_up(self) -> None:
        """Load (and download if needed) the voice; prime synthesis and audio."""
        self._ensure_loaded()
        t0 = time.perf_counter()
        if self._net is not None:
            self._load_fallback_async()
            try:
                job = self._net.start("Bonjour.")
                for _ in job.chunks(lambda: False, timeout=10.0):
                    pass
            except Exception as exc:
                self._mark_net_down(exc)
        else:
            for _ in self._synthesize_local("Bonjour."):
                pass
        if self._sink is None:
            try:
                self._default_sink().start(self._sample_rate)
            except Exception:
                logger.exception("Audio output unavailable")
        logger.info(
            "Voice warm-up (%s) done in %.2fs", self._engine, time.perf_counter() - t0
        )

    def prepare(self) -> None:
        """Hint that a reply is coming soon (e.g. the user stopped talking).

        Online engines open their connection now, so the first sentence does
        not pay the handshake. Cheap and optional.
        """
        if self._loaded and self._net_ok():
            try:
                self._net.prepare(_NET_LOOKAHEAD)  # type: ignore[union-attr]
            except Exception:
                logger.debug("Voice prepare failed", exc_info=True)

    # -- synthesis ---------------------------------------------------------

    def _net_ok(self) -> bool:
        return self._net is not None and time.monotonic() >= self._net_down_until

    def _mark_net_down(self, exc: BaseException) -> None:
        if time.monotonic() >= self._net_down_until:
            logger.warning(
                "Voice engine %s failed (%s); using the local voice for %.0f s",
                self._engine,
                exc,
                _NET_RETRY_AFTER_S,
            )
        self._net_down_until = time.monotonic() + _NET_RETRY_AFTER_S

    def _start_job(self, text: str) -> SpeechJob | None:
        """Start an online request for ``text`` (None = synthesize locally)."""
        if not self._net_ok():
            return None
        try:
            return self._net.start(text)  # type: ignore[union-attr]
        except Exception as exc:
            self._mark_net_down(exc)
            return None

    def _job_audio(
        self,
        text: str,
        job: SpeechJob | None,
        halted: Callable[[], bool],
        idle: Callable[[], None] | None = None,
    ) -> Iterable[np.ndarray]:
        if job is not None:
            got = False
            try:
                for audio in job.chunks(halted, idle):
                    got = True
                    yield audio
                return
            except Exception as exc:
                self._mark_net_down(exc)
                if got:
                    return  # half the sentence was said: do not repeat it
        if not halted():
            yield from self._synthesize_local(text)

    def _synthesize(self, text: str) -> Iterable[np.ndarray]:
        """Yield float32 mono chunks at ``self.sample_rate`` for ``text``."""
        yield from self._job_audio(text, self._start_job(text), lambda: False)

    def _synthesize_local(self, text: str) -> Iterable[np.ndarray]:
        """piper (resampled to the output rate if needed), else ``say``."""
        if self._piper is not None:
            said = False
            try:
                for chunk in self._piper.synthesize(text, self._syn_config):
                    said = True
                    audio = _resample(
                        chunk.audio_float_array, self._piper_rate, self._sample_rate
                    )
                    yield audio * _FALLBACK_GAIN if self._net is not None else audio
                return
            except Exception:
                logger.exception("Piper synthesis failed for %r", text)
                if said:
                    return  # do not repeat part of the sentence with another voice
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
        """Release the audio output and the online engine (at shutdown)."""
        if self._output is not None:
            self._output.close()
        net, self._net = self._net, None
        if net is not None:
            net.close()

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

        self.prepare()  # connect while the LLM writes the first sentence
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

        def halted() -> bool:
            return stop.is_set() or done.is_set()

        # Online engines: sentences queued behind the one playing are
        # requested right away (up to _NET_LOOKAHEAD), so their audio is
        # ready when needed. Local engines synthesize one sentence at a time.
        pending: deque[tuple[str, SpeechJob | None]] = deque()
        eof = [False]

        def take(raw: str | None) -> None:
            if raw is None:
                eof[0] = True
                return
            text = clean_text(raw)
            if _alpha_count(text):
                pending.append((text, self._start_job(text)))

        def pull() -> None:
            lookahead = _NET_LOOKAHEAD + 1 if self._net_ok() else 1
            while not eof[0] and len(pending) < lookahead:
                try:
                    take(sentences.get_nowait())
                except queue.Empty:
                    return

        def synth_worker() -> None:
            gap = np.zeros(int(self._sample_rate * _SENTENCE_GAP_S), dtype=np.float32)
            fx = self._fx.new()  # fresh filter state for each reply
            finished = False
            try:
                while not halted():
                    pull()
                    if not pending:
                        if eof[0]:
                            break
                        try:
                            take(sentences.get(timeout=0.05))
                        except queue.Empty:
                            pass
                        continue
                    text, job = pending.popleft()
                    first = True
                    for audio in self._job_audio(text, job, halted, idle=pull):
                        if halted():
                            return
                        audio = fx.process(audio)
                        if not put_audio((text if first else "", audio)):
                            return
                        first = False
                    if job is not None and not self._net_ok():
                        # The network failed: say the queued sentences locally.
                        for i, (t, j) in enumerate(pending):
                            if j is not None:
                                j.cancel()
                                pending[i] = (t, None)
                    if not first and not put_audio(("", fx.process(gap))):
                        return
                finished = True
            finally:
                for _, j in pending:
                    if j is not None:
                        j.cancel()
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
