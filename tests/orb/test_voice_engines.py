"""Tests for the orb's online voice engines (edge, elevenlabs), without network."""

from __future__ import annotations

import io
import json
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from openjarvis.orb import voice as V

SR = V._NET_SAMPLE_RATE


@dataclass
class Cfg:
    engine: str = "edge"
    piper_voice: str = "fr_FR-tom-medium"
    say_voice: str = "Thomas"
    speed: float = 1.15
    effect: str = "jarvis"


class WavSink:
    def __init__(self, realtime: float = 0.0) -> None:
        self.blocks: list[np.ndarray] = []
        self.rate = 0
        self.realtime = realtime
        self.aborted = False

    def start(self, sample_rate: int) -> None:
        self.rate = sample_rate

    def write(self, block: np.ndarray) -> None:
        self.blocks.append(block.copy())
        if self.realtime:
            time.sleep(self.realtime * len(block) / self.rate)

    def finish(self) -> None:
        pass

    def abort(self) -> None:
        self.aborted = True

    @property
    def audio(self) -> np.ndarray:
        return np.concatenate(self.blocks) if self.blocks else np.zeros(0)


def tone(seconds: float, freq: float = 220.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SR * seconds), dtype=np.float32)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("speed", "rate"),
    [(1.15, "+15%"), (1.0, "+0%"), (0.9, "-10%"), (1.25, "+25%"), (9.0, "+100%")],
)
def test_speed_to_rate(speed: float, rate: str) -> None:
    assert V.speed_to_rate(speed) == rate


@pytest.mark.parametrize(
    ("pitch", "out"),
    [("+0Hz", "+0Hz"), ("-3Hz", "-3Hz"), ("2", "+2Hz"), ("", "+0Hz"), ("low", "+0Hz")],
)
def test_normalize_pitch(pitch: str, out: str) -> None:
    assert V.normalize_pitch(pitch) == out


def test_silence_trimmer_keeps_short_margins_and_inner_pauses() -> None:
    speech = np.concatenate([tone(0.3), silence(0.2), tone(0.3)])
    padded = np.concatenate([silence(0.25), speech, silence(0.8)])
    trimmer = V.SilenceTrimmer(SR, lead_keep=0.03, trail_keep=0.1)
    # Fed in small chunks, like a network stream.
    out = [trimmer.process(padded[i : i + 700]) for i in range(0, len(padded), 700)]
    out.append(trimmer.flush())
    result = np.concatenate(out)
    lead = np.flatnonzero(np.abs(result) > 0.006)[0] / SR
    assert lead == pytest.approx(0.03, abs=0.002)
    # inner 0.2 s pause kept, 0.8 s tail cut to 0.1 s
    assert len(result) / SR == pytest.approx(0.03 + len(speech) / SR + 0.1, abs=0.01)


def test_silence_trimmer_all_silent() -> None:
    trimmer = V.SilenceTrimmer(SR)
    assert len(trimmer.process(silence(0.5))) == 0
    assert len(trimmer.flush()) == 0


def test_mp3_decoder_decodes_a_real_stream_in_pieces() -> None:
    av = pytest.importorskip("av")
    signal = np.concatenate([tone(0.5, amp=0.4), silence(0.1)])
    buf = io.BytesIO()
    with av.open(buf, "w", format="mp3") as out:
        stream = out.add_stream("libmp3lame", rate=SR, layout="mono")
        frame = av.AudioFrame.from_ndarray(
            (signal * 32767).astype(np.int16).reshape(1, -1),
            format="s16",
            layout="mono",
        )
        frame.sample_rate = SR
        for packet in stream.encode(frame):
            out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
    data = buf.getvalue()
    dec = V.Mp3Decoder()
    parts = [p for i in range(0, len(data), 720) for p in dec.feed(data[i : i + 720])]
    parts += dec.flush()
    audio = np.concatenate(parts)
    assert audio.dtype == np.float32
    assert len(audio) / SR == pytest.approx(len(signal) / SR, abs=0.12)
    assert 0.3 < np.abs(audio).max() < 0.5


# --------------------------------------------------------------------------
# Edge engine with a fake websocket
# --------------------------------------------------------------------------


class FakeDecoder:
    """Payload bytes are raw float32 samples (stands in for MP3)."""

    def feed(self, data: bytes) -> list[np.ndarray]:
        return [np.frombuffer(data, dtype=np.float32).copy()]

    def flush(self) -> list[np.ndarray]:
        return []


def _audio_msg(samples: np.ndarray) -> Any:
    import aiohttp

    header = b"X-RequestId:x\r\nContent-Type:audio/mpeg\r\nPath:audio\r\n"
    data = len(header).to_bytes(2, "big") + header + samples.tobytes()
    return SimpleNamespace(type=aiohttp.WSMsgType.BINARY, data=data)


def _text_msg(path: str) -> Any:
    import aiohttp

    return SimpleNamespace(
        type=aiohttp.WSMsgType.TEXT, data=f"X-RequestId:x\r\nPath:{path}\r\n\r\n{{}}"
    )


class FakeWS:
    def __init__(self, turns: int = 10, broken: bool = False) -> None:
        self.sent: list[str] = []
        self.closed = False
        self.broken = broken
        self._inbox: list[Any] = []
        self.turns = turns

    async def send_str(self, data: str) -> None:
        if self.broken:
            raise ConnectionResetError("closing transport")
        self.sent.append(data)
        if "Path:ssml" in data:
            self._inbox = [
                _text_msg("turn.start"),
                _audio_msg(silence(0.2)),
                _audio_msg(tone(0.2)),
                _audio_msg(tone(0.2)),
                _audio_msg(silence(0.8)),
                _text_msg("turn.end"),
            ]

    async def receive(self, timeout: float | None = None) -> Any:
        import asyncio

        while not self._inbox:
            await asyncio.sleep(0.005)
        return self._inbox.pop(0)

    async def close(self) -> None:
        self.closed = True


@pytest.fixture()
def fake_decoder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(V, "Mp3Decoder", FakeDecoder)


def _collect(job: V.SpeechJob) -> np.ndarray:
    return np.concatenate(list(job.chunks(lambda: False, timeout=5)))


def test_edge_engine_streams_trims_and_reuses_connection(fake_decoder: None) -> None:
    sockets: list[FakeWS] = []

    async def connect() -> FakeWS:
        sockets.append(FakeWS())
        return sockets[-1]

    eng = V.EdgeEngine("fr-FR-HenriNeural", rate="+15%", pitch="-2Hz", connect=connect)
    try:
        a1 = _collect(eng.start("Bonjour <Monsieur> & co."))
        a2 = _collect(eng.start("Deuxième phrase."))
    finally:
        eng.close()
    assert len(sockets) == 1  # second request reused the websocket
    sent = sockets[0].sent
    assert "Path:speech.config" in sent[0]
    assert "audio-24khz-48kbitrate-mono-mp3" in sent[0]
    ssml = sent[1]
    assert "(fr-FR, HenriNeural)" in ssml
    assert "rate='+15%'" in ssml and "pitch='-2Hz'" in ssml
    assert "&lt;Monsieur&gt; &amp; co." in ssml
    # 0.4 s of speech + 30 ms lead + 140 ms tail instead of 0.2 + 0.8 s padding
    for audio in (a1, a2):
        expected = 0.4 + V._LEAD_KEEP_S + V._TRAIL_KEEP_S
        assert len(audio) / SR == pytest.approx(expected, abs=0.01)


def test_edge_engine_retries_a_stale_connection(fake_decoder: None) -> None:
    sockets: list[FakeWS] = []

    async def connect() -> FakeWS:
        sockets.append(FakeWS())
        return sockets[-1]

    eng = V.EdgeEngine("fr-FR-HenriNeural", connect=connect)
    try:
        _collect(eng.start("Un."))
        sockets[0].broken = True  # the service closed the idle websocket
        audio = _collect(eng.start("Deux."))
    finally:
        eng.close()
    assert len(sockets) == 2
    assert sockets[0].closed
    assert len(audio) > 0


def test_edge_engine_failure_raises_engine_error(fake_decoder: None) -> None:
    async def connect() -> FakeWS:
        raise OSError("network is unreachable")

    eng = V.EdgeEngine("fr-FR-HenriNeural", connect=connect)
    try:
        with pytest.raises(V.EngineError, match="OSError"):
            _collect(eng.start("Bonjour."))
    finally:
        eng.close()


def test_edge_engine_rejects_invalid_voice() -> None:
    with pytest.raises(ValueError):
        V.EdgeEngine("not a voice", rate="+15%")


# --------------------------------------------------------------------------
# ElevenLabs with a fake transport
# --------------------------------------------------------------------------

KEY = "sk_test_0123456789abcdef"


def test_elevenlabs_request_building() -> None:
    import httpx

    eng = V.ElevenLabsEngine(
        voice_id="voice123",
        speed=1.15,
        api_key=KEY,
        transport=httpx.MockTransport(None),
    )
    try:
        req = eng.build_request("Bonjour Monsieur.", previous_text="Avant.")
    finally:
        eng.close()
    assert req.method == "POST"
    assert req.url.path == "/v1/text-to-speech/voice123/stream"
    assert req.url.params["output_format"] == "pcm_24000"
    assert req.headers["xi-api-key"] == KEY
    body = json.loads(req.content)
    assert body["text"] == "Bonjour Monsieur."
    assert body["model_id"] == "eleven_flash_v2_5"
    assert body["language_code"] == "fr"
    assert body["previous_text"] == "Avant."
    assert body["voice_settings"]["speed"] == pytest.approx(1.15)


def test_elevenlabs_speed_is_clamped() -> None:
    import httpx

    eng = V.ElevenLabsEngine(
        speed=1.6, api_key=KEY, transport=httpx.MockTransport(None)
    )
    eng.close()
    assert eng.speed == 1.2
    assert eng.voice_id == V._DEFAULT_ELEVENLABS_VOICE


def test_elevenlabs_streams_pcm_split_on_odd_bytes() -> None:
    import httpx

    signal = np.concatenate([silence(0.1), tone(0.3), silence(0.5)])
    pcm = (signal * 32767).astype("<i2").tobytes()

    def chunks() -> Any:
        for i in range(0, len(pcm), 1001):  # odd sizes: samples split in two
            yield pcm[i : i + 1001]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=chunks())

    eng = V.ElevenLabsEngine(api_key=KEY, transport=httpx.MockTransport(handler))
    try:
        audio = _collect(eng.start("Bonjour."))
    finally:
        eng.close()
    expected = 0.3 + V._LEAD_KEEP_S + V._TRAIL_KEEP_S
    assert len(audio) / SR == pytest.approx(expected, abs=0.01)
    assert np.abs(audio).max() == pytest.approx(0.3, abs=0.01)


def test_elevenlabs_http_error_does_not_leak_key() -> None:
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": f"bad key {KEY}"})

    eng = V.ElevenLabsEngine(api_key=KEY, transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(V.EngineError) as info:
            _collect(eng.start("Bonjour."))
    finally:
        eng.close()
    assert "401" in str(info.value)
    assert KEY not in str(info.value)


def test_elevenlabs_without_key_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.setattr(V, "ELEVENLABS_KEY_FILE", tmp_path / "missing.key")
    with pytest.raises(V.EngineError):
        V.ElevenLabsEngine()


def test_elevenlabs_key_from_env_or_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    key_file = tmp_path / "elevenlabs.key"
    key_file.write_text(f"  {KEY}\n", encoding="utf-8")
    monkeypatch.setattr(V, "ELEVENLABS_KEY_FILE", key_file)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    assert V.load_elevenlabs_key() == KEY
    monkeypatch.setenv("ELEVENLABS_API_KEY", "from-env")
    assert V.load_elevenlabs_key() == "from-env"


# --------------------------------------------------------------------------
# Voice: engine selection, fallback chain, streaming pipeline
# --------------------------------------------------------------------------


class FakeNet:
    """Online engine stand-in: each sentence is 0.25 s of tone, after `delay`."""

    name = "edge"
    sample_rate = SR

    def __init__(self, fail: bool = False, delay: float = 0.05) -> None:
        self.fail = fail
        self.delay = delay
        self.started: list[tuple[str, float]] = []
        self.prepared = 0
        self.closed = False

    def start(self, text: str) -> V.SpeechJob:
        job = V.SpeechJob()
        self.started.append((text, time.monotonic()))

        def run() -> None:
            time.sleep(self.delay)
            if self.fail:
                job.end(V.EngineError("network down"))
                return
            for _ in range(5):
                if job.cancelled.is_set():
                    break
                job.put(tone(0.05))
            job.end()

        threading.Thread(target=run, daemon=True).start()
        return job

    def prepare(self, connections: int = 1) -> None:
        self.prepared += 1

    def close(self) -> None:
        self.closed = True


def _voice(monkeypatch: pytest.MonkeyPatch, net: Any, **cfg: Any) -> V.Voice:
    """A Voice whose local fallback says each sentence as 0.1 s of a 440 Hz tone."""
    v = V.Voice(Cfg(**cfg))  # type: ignore[arg-type]
    v.local_calls = []  # type: ignore[attr-defined]

    def make() -> Any:
        if isinstance(net, Exception):
            raise net
        return net

    def load_piper() -> None:
        raise RuntimeError("no piper in tests")

    def say(text: str) -> Any:
        v.local_calls.append(text)  # type: ignore[attr-defined]
        yield tone(0.1, freq=440.0)

    monkeypatch.setattr(v, "_make_net_engine", make)
    monkeypatch.setattr(v, "_load_piper", load_piper)
    monkeypatch.setattr(v, "_synthesize_say", say)
    v._sink = WavSink()
    return v


def test_online_engine_plays_unprocessed_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v = _voice(monkeypatch, FakeNet(), effect="jarvis")
    v._ensure_loaded()
    assert v.engine == "edge"
    assert v.sample_rate == SR
    assert not v._fx.enabled  # cfg.effect is for the local voices only


def test_online_effect_can_be_opted_in(monkeypatch: pytest.MonkeyPatch) -> None:
    v = _voice(monkeypatch, FakeNet())
    v.cfg.online_effect = "jarvis"  # type: ignore[attr-defined]
    v._ensure_loaded()
    assert v._fx.enabled


def test_unavailable_online_engine_falls_back_to_piper_then_say(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v = _voice(monkeypatch, V.EngineError("no key"), engine="elevenlabs")
    v._ensure_loaded()
    assert v.engine == "say"
    assert v.sample_rate == V._SAY_SAMPLE_RATE
    assert v._fx.enabled  # local voice: cfg.effect applies again


def test_make_net_engine_reads_optional_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = Cfg(speed=1.25)
    cfg.edge_voice = "fr-BE-GerardNeural"  # type: ignore[attr-defined]
    cfg.pitch = "-2Hz"  # type: ignore[attr-defined]
    seen: dict[str, Any] = {}

    class Spy:
        def __init__(self, **kw: Any) -> None:
            seen.update(kw)

    monkeypatch.setattr(V, "EdgeEngine", Spy)
    v = V.Voice(cfg)  # type: ignore[arg-type]
    v._make_net_engine()
    assert seen == {"voice": "fr-BE-GerardNeural", "rate": "+25%", "pitch": "-2Hz"}
    seen.clear()
    V.Voice(Cfg())._make_net_engine()  # type: ignore[arg-type]
    assert seen["voice"] == V._DEFAULT_EDGE_VOICE and seen["pitch"] == "+0Hz"


TEXT = ["Bonjour Monsieur. ", "Tous les systèmes ", "sont opérationnels. ", "Fin."]


def test_speak_stream_online_requests_ahead(monkeypatch: pytest.MonkeyPatch) -> None:
    net = FakeNet(delay=0.1)
    v = _voice(monkeypatch, net)
    levels: list[float] = []
    first: list[bool] = []
    spoken = v.speak_stream(
        TEXT,
        threading.Event(),
        level_cb=levels.append,
        on_first_audio=lambda: first.append(True),
    )
    assert spoken == "Bonjour Monsieur. Tous les systèmes sont opérationnels. Fin."
    assert [t for t, _ in net.started] == [
        "Bonjour Monsieur.",
        "Tous les systèmes sont opérationnels.",
        "Fin.",
    ]
    # All three requested at once, not one after the other.
    assert net.started[-1][1] - net.started[0][1] < 0.08
    assert net.prepared >= 1 and first == [True]
    assert max(levels) > 0.5 and levels[-1] == 0.0
    assert v.local_calls == []  # type: ignore[attr-defined]
    sink = v._sink
    assert isinstance(sink, WavSink)
    gap = int(SR * V._SENTENCE_GAP_S)
    assert len(sink.audio) == 3 * (int(SR * 0.05) * 5 + gap)


def test_network_failure_mid_reply_falls_back_locally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    net = FakeNet(fail=True)
    v = _voice(monkeypatch, net)
    spoken = v.speak_stream(TEXT, threading.Event())
    assert spoken == "Bonjour Monsieur. Tous les systèmes sont opérationnels. Fin."
    assert v.local_calls == [  # type: ignore[attr-defined]
        "Bonjour Monsieur.",
        "Tous les systèmes sont opérationnels.",
        "Fin.",
    ]
    # Engine marked down: the next reply goes straight to the local voice.
    n = len(net.started)
    v.speak_stream(["Encore une phrase."], threading.Event())
    assert len(net.started) == n
    assert v.local_calls[-1] == "Encore une phrase."  # type: ignore[attr-defined]


def test_stop_mid_reply_returns_quickly(monkeypatch: pytest.MonkeyPatch) -> None:
    v = _voice(monkeypatch, FakeNet(delay=0.02))
    v._sink = WavSink(realtime=1.0)
    stop = threading.Event()
    many = [f"Phrase numéro {i}. " for i in range(30)]
    th = threading.Thread(target=v.speak_stream, args=(many, stop))
    th.start()
    time.sleep(0.4)
    t0 = time.monotonic()
    stop.set()
    th.join(2.0)
    assert not th.is_alive()
    assert time.monotonic() - t0 < 0.15
    assert v._sink.aborted  # type: ignore[union-attr]


def test_close_closes_online_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    net = FakeNet()
    v = _voice(monkeypatch, net)
    v._ensure_loaded()
    v.close()
    assert net.closed
