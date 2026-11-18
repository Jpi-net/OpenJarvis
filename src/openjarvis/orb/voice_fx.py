"""Voice post-processing for the orb: an original, composed "AI butler" timbre.

The chain is applied to every synthesized chunk of a reply, in order:

1. pitch      small pitch drop by resampling (the caller asks the TTS to speak
              ``pitch_ratio`` faster, so the duration is unchanged);
2. EQ         high-pass, low-shelf warmth, mud cut, presence, air shelf;
3. room       a few short early reflections ("small room / helmet" resonance);
4. sheen      a slow, shallow chorus (and optionally a faint ring modulator)
              mixed low, for a clean synthetic gloss;
5. output     make-up gain and a soft-knee limiter (ceiling -1 dBFS).

Every stage keeps its state between ``process`` calls, so feeding a reply
chunk by chunk gives the same samples as processing it in one buffer (no
clicks at chunk boundaries). Use one ``VoiceFx`` per reply
(``FxDesign.new()``) and call ``flush()`` at the end to get the tail.

Cost: ~1-2 ms per second of 44.1 kHz audio on Apple silicon.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("openjarvis.orb.voice_fx")

_CEILING = 10 ** (-1.0 / 20)  # -1 dBFS
_KNEE = 10 ** (-7.0 / 20)  # the limiter starts bending here


@dataclass(frozen=True)
class Preset:
    description: str
    pitch_semitones: float = 0.0
    highpass_hz: float = 0.0
    low_shelf: tuple[float, float] = (0.0, 0.0)  # (Hz, dB)
    peaks: tuple[tuple[float, float, float], ...] = ()  # (Hz, dB, Q)
    air_shelf: tuple[float, float] = (0.0, 0.0)  # (Hz, dB)
    reflections: tuple[tuple[float, float], ...] = ()  # (ms, gain)
    chorus: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    # chorus = (mix, base delay ms, depth ms, rate Hz)
    ring: tuple[float, float] = (0.0, 0.0)  # (mix, carrier Hz)
    gain_db: float = 0.0


PRESETS: dict[str, Preset | None] = {
    "none": None,
    # Recommended: deeper and warmer than the raw voice, a touch of air and a
    # very small room, with a barely audible synthetic gloss.
    "jarvis": Preset(
        description="calme, grave et posé ; légère aura synthétique",
        pitch_semitones=-1.0,
        highpass_hz=85.0,
        low_shelf=(170.0, 2.0),
        peaks=((420.0, -1.5, 1.0), (3200.0, 1.5, 0.9)),
        air_shelf=(9000.0, 3.0),
        reflections=((4.3, 0.16), (7.9, -0.11), (12.1, 0.08), (17.3, -0.05)),
        chorus=(0.12, 2.6, 0.5, 0.23),
        gain_db=-1.6,
    ),
    # Lower and fuller, more "presence in the room".
    "jarvis_deep": Preset(
        description="plus grave et plus ample",
        pitch_semitones=-2.0,
        highpass_hz=75.0,
        low_shelf=(150.0, 3.0),
        peaks=((450.0, -2.0, 1.0), (2800.0, 2.0, 0.9)),
        air_shelf=(9000.0, 2.5),
        reflections=((5.1, 0.17), (9.3, -0.12), (14.2, 0.08), (21.7, -0.05)),
        chorus=(0.10, 3.0, 0.5, 0.2),
        gain_db=-2.2,
    ),
    # Lighter, crisper, more "hologram": brighter top, a little more shimmer.
    "jarvis_air": Preset(
        description="plus clair et aérien, brillance « hologramme »",
        pitch_semitones=-0.7,
        highpass_hz=100.0,
        low_shelf=(170.0, 1.0),
        peaks=((400.0, -1.5, 1.0), (3500.0, 2.0, 0.8)),
        air_shelf=(8000.0, 5.0),
        reflections=((3.7, 0.13), (6.9, -0.09), (10.7, 0.06)),
        chorus=(0.16, 2.2, 0.7, 0.37),
        gain_db=-2.4,
    ),
    # More processed: boxy band resonance, denser reflections, faint ring-mod.
    "helmet": Preset(
        description="plus traité : résonance de casque, reflets métalliques",
        pitch_semitones=-1.5,
        highpass_hz=120.0,
        low_shelf=(200.0, 1.0),
        peaks=((1150.0, 4.0, 1.4), (2600.0, 2.0, 2.0), (450.0, -2.0, 1.0)),
        air_shelf=(9000.0, 1.0),
        reflections=(
            (2.9, 0.30),
            (5.3, -0.24),
            (7.7, 0.19),
            (10.9, -0.14),
            (14.2, 0.10),
            (19.0, -0.06),
        ),
        chorus=(0.20, 1.6, 0.6, 0.5),
        ring=(0.05, 230.0),
        gain_db=-1.4,
    ),
}


def preset_names() -> list[str]:
    return list(PRESETS)


# --------------------------------------------------------------------------
# Filter design (RBJ audio-EQ cookbook biquads, as scipy "sos" rows)
# --------------------------------------------------------------------------


def _sos_row(b: list[float], a: list[float]) -> list[float]:
    a0 = a[0]
    return [b[0] / a0, b[1] / a0, b[2] / a0, 1.0, a[1] / a0, a[2] / a0]


def _peaking(f: float, db: float, q: float, sr: int) -> list[float]:
    big_a = 10 ** (db / 40)
    w = 2 * math.pi * f / sr
    alpha = math.sin(w) / (2 * q)
    c = math.cos(w)
    return _sos_row(
        [1 + alpha * big_a, -2 * c, 1 - alpha * big_a],
        [1 + alpha / big_a, -2 * c, 1 - alpha / big_a],
    )


def _shelf(f: float, db: float, sr: int, high: bool) -> list[float]:
    big_a = 10 ** (db / 40)
    w = 2 * math.pi * f / sr
    c, s = math.cos(w), math.sin(w)
    alpha = s / 2 * math.sqrt(2)  # shelf slope S = 1
    sq = 2 * math.sqrt(big_a) * alpha
    if high:
        b = [
            big_a * ((big_a + 1) + (big_a - 1) * c + sq),
            -2 * big_a * ((big_a - 1) + (big_a + 1) * c),
            big_a * ((big_a + 1) + (big_a - 1) * c - sq),
        ]
        a = [
            (big_a + 1) - (big_a - 1) * c + sq,
            2 * ((big_a - 1) - (big_a + 1) * c),
            (big_a + 1) - (big_a - 1) * c - sq,
        ]
    else:
        b = [
            big_a * ((big_a + 1) - (big_a - 1) * c + sq),
            2 * big_a * ((big_a - 1) - (big_a + 1) * c),
            big_a * ((big_a + 1) - (big_a - 1) * c - sq),
        ]
        a = [
            (big_a + 1) + (big_a - 1) * c + sq,
            -2 * ((big_a - 1) + (big_a + 1) * c),
            (big_a + 1) + (big_a - 1) * c - sq,
        ]
    return _sos_row(b, a)


def _highpass(f: float, sr: int) -> list[float]:
    w = 2 * math.pi * f / sr
    c = math.cos(w)
    alpha = math.sin(w) / (2 * (1 / math.sqrt(2)))  # Butterworth Q
    b = [(1 + c) / 2, -(1 + c), (1 + c) / 2]
    return _sos_row(b, [1 + alpha, -2 * c, 1 - alpha])


# --------------------------------------------------------------------------
# Design (per preset and sample rate) and per-reply processor
# --------------------------------------------------------------------------


@dataclass
class FxDesign:
    name: str
    sample_rate: int
    preset: Preset | None
    sos: np.ndarray | None = None
    taps: list[tuple[int, float]] = field(default_factory=list)

    @property
    def pitch_ratio(self) -> float:
        """Read-speed ratio of the resampler (< 1 lowers the pitch).

        Resampling stretches the audio by ``1 / pitch_ratio``: the TTS should
        multiply its length scale by ``pitch_ratio`` (speak that much faster)
        so the final duration is unchanged.
        """
        if self.preset is None:
            return 1.0
        return 2 ** (self.preset.pitch_semitones / 12)

    @property
    def enabled(self) -> bool:
        return self.preset is not None

    def new(self) -> VoiceFx:
        return VoiceFx(self)


def design(name: str | None, sample_rate: int) -> FxDesign:
    """Build the filters of preset ``name`` (unknown -> warning, no effect)."""
    key = (name or "none").strip().lower()
    if key not in PRESETS:
        logger.warning(
            "Unknown voice effect %r (choose among %s); using 'none'",
            name,
            ", ".join(PRESETS),
        )
        key = "none"
    preset = PRESETS[key]
    d = FxDesign(name=key, sample_rate=sample_rate, preset=preset)
    if preset is None:
        return d
    sr = sample_rate
    nyq = sr / 2
    rows: list[list[float]] = []
    if preset.highpass_hz > 0:
        rows.append(_highpass(preset.highpass_hz, sr))
    f, db = preset.low_shelf
    if f > 0 and db:
        rows.append(_shelf(f, db, sr, high=False))
    for f, db, q in preset.peaks:
        if db and f < 0.45 * sr:
            rows.append(_peaking(f, db, q, sr))
    f, db = preset.air_shelf
    if f > 0 and db:
        rows.append(_shelf(min(f, 0.62 * nyq), db, sr, high=True))
    d.sos = np.asarray(rows, dtype=np.float64) if rows else None
    d.taps = [(max(1, round(ms * sr / 1000)), g) for ms, g in preset.reflections]
    return d


class _History:
    """Keep the last ``size`` input samples to read delayed values."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.buf = np.zeros(size, dtype=np.float64)

    def extend(self, x: np.ndarray) -> np.ndarray:
        ext = np.concatenate([self.buf, x])
        self.buf = ext[-self.size :] if self.size else self.buf
        return ext


class _PitchDrop:
    """Streaming cubic (Catmull-Rom) resampler reading the input at ``ratio``."""

    def __init__(self, ratio: float) -> None:
        self.ratio = ratio
        self.buf = np.zeros(1, dtype=np.float64)  # x[-1] = 0 (silence before)
        self.buf_start = -1  # global input index of buf[0]
        self.out_count = 0  # global output index of the next sample

    def process(self, x: np.ndarray) -> np.ndarray:
        buf = np.concatenate([self.buf, x])
        end = self.buf_start + len(buf)  # global index past the last input
        # output m reads input position t = m * ratio; needs t + 2 < end
        last = math.floor((end - 3) / self.ratio) if end >= 3 else -1
        n = last - self.out_count + 1
        if n > 0:
            m = np.arange(self.out_count, last + 1, dtype=np.float64)
            t = m * self.ratio - self.buf_start
            i = np.floor(t).astype(np.int64)
            fr = t - i
            p0, p1, p2, p3 = buf[i - 1], buf[i], buf[i + 1], buf[i + 2]
            out = p1 + 0.5 * fr * (
                p2
                - p0
                + fr * (2 * p0 - 5 * p1 + 4 * p2 - p3 + fr * (3 * (p1 - p2) + p3 - p0))
            )
            self.out_count = last + 1
        else:
            out = np.zeros(0, dtype=np.float64)
        keep_from = math.floor(self.out_count * self.ratio) - 1 - self.buf_start
        keep_from = max(0, min(keep_from, len(buf)))
        self.buf = buf[keep_from:]
        self.buf_start += keep_from
        return out

    def flush(self) -> np.ndarray:
        return self.process(np.zeros(3, dtype=np.float64))


class VoiceFx:
    """Stateful processor for one reply."""

    def __init__(self, d: FxDesign) -> None:
        self.d = d
        p = d.preset
        self.enabled = p is not None
        if p is None:
            return
        from scipy.signal import sosfilt

        self._sosfilt = sosfilt
        self._pitch = _PitchDrop(d.pitch_ratio) if p.pitch_semitones else None
        self._zi = np.zeros((len(d.sos), 2)) if d.sos is not None else None
        sr = d.sample_rate
        mix, base_ms, depth_ms, rate = p.chorus
        self._chorus = (
            (mix, base_ms * sr / 1000, depth_ms * sr / 1000, 2 * math.pi * rate / sr)
            if mix > 0
            else None
        )
        max_tap = max((t for t, _ in d.taps), default=0)
        self._room_hist = _History(max_tap)
        ch_max = 0
        if self._chorus is not None:
            ch_max = math.ceil(self._chorus[1] + self._chorus[2]) + 2
        self._chorus_hist = _History(ch_max)
        ring_mix, ring_hz = p.ring
        self._ring = (ring_mix, 2 * math.pi * ring_hz / sr) if ring_mix > 0 else None
        self._gain = 10 ** (p.gain_db / 20)
        self._n = 0  # samples output so far (LFO / carrier phase)

    def process(self, audio: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return audio
        x = np.asarray(audio, dtype=np.float64)
        if self._pitch is not None:
            x = self._pitch.process(x)
        return self._post(x)

    def flush(self) -> np.ndarray:
        """Tail of the reply (resampler leftovers + reflections)."""
        if not self.enabled:
            return np.zeros(0, dtype=np.float32)
        x = self._pitch.flush() if self._pitch is not None else np.zeros(0)
        tail = max(self._room_hist.size, self._chorus_hist.size)
        return self._post(np.concatenate([x, np.zeros(tail)]))

    def _post(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        if n == 0:
            return x.astype(np.float32)
        d = self.d
        if d.sos is not None:
            x, self._zi = self._sosfilt(d.sos, x, zi=self._zi)
        if d.taps:
            h = self._room_hist.size
            ext = self._room_hist.extend(x)
            y = x.copy()
            for delay, g in d.taps:
                y += g * ext[h - delay : h - delay + n]
            x = y
        idx = None
        if self._chorus is not None or self._ring is not None:
            idx = np.arange(self._n, self._n + n, dtype=np.float64)
        if self._chorus is not None:
            mix, base, depth, w = self._chorus
            h = self._chorus_hist.size
            ext = self._chorus_hist.extend(x)
            delay = base + depth * np.sin(w * idx)
            pos = np.arange(h, h + n, dtype=np.float64) - delay
            i = np.floor(pos).astype(np.int64)
            fr = pos - i
            wet = ext[i] * (1 - fr) + ext[i + 1] * fr
            x = x + mix * wet
        if self._ring is not None:
            mix, w = self._ring
            x = x * (1 - mix) + mix * x * np.cos(w * idx) * 2
        self._n += n
        x = x * self._gain
        return _soft_limit(x).astype(np.float32)


def _soft_limit(x: np.ndarray) -> np.ndarray:
    """Stateless soft knee: linear below -7 dBFS, smoothly capped at -1 dBFS."""
    a = np.abs(x)
    over = a > _KNEE
    if not over.any():
        return x
    room = _CEILING - _KNEE
    y = x.copy()
    y[over] = np.sign(x[over]) * (_KNEE + room * np.tanh((a[over] - _KNEE) / room))
    return y
