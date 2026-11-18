# Orb module contract

Shared interfaces for `src/openjarvis/orb/`. Each module is owned by one
implementer; `app.py` (the orchestrator) only relies on what is written here.

General rules
- Python 3.11+, `from __future__ import annotations`, type hints, no print():
  use `logging.getLogger("openjarvis.orb.<module>")`.
- Read settings only from `openjarvis.orb.config` (`JarvisConfig` sections).
  Store downloaded models / state under `config.DATA_DIR` (`~/.openjarvis/orb/`).
- Network downloads: call `truststore.inject_into_ssl()` first (the user's
  network intercepts TLS; system certificates are required).
- Heavy imports (numpy, faster_whisper, piper, openwakeword, webview) happen
  inside functions/constructors, so `import openjarvis.orb.<module>` stays cheap.
- Never block forever: every blocking call takes a `threading.Event` `stop`
  and returns promptly once it is set.
- Do not edit `pyproject.toml`, `config.py`, `app.py`, or other modules' files.
  Do not commit. Dependencies are already installed in `.venv`
  (pywebview, openwakeword, piper-tts, faster-whisper, sounddevice, numpy,
  onnxruntime, httpx, truststore).

States (strings): "idle", "listening", "thinking", "speaking".

## ui.py — `OrbWindow`
```python
class OrbWindow:
    def __init__(self, cfg: OrbConfig, title: str = "Jarvis"): ...
    def run(self, worker: Callable[[], None]) -> None:
        """Create the window and block on the GUI loop (must be called on the
        main thread, macOS requirement). Starts `worker` in a daemon thread
        once the window is shown. Returns when the window closes."""

    def set_state(self, state: str) -> None: ...  # thread-safe
    def set_level(
        self, level: float
    ) -> None: ...  # 0..1 audio amplitude, thread-safe, called ~30x/s
    def close(self) -> None: ...
```

## ears.py — `Ears`
```python
class Ears:
    def __init__(self, cfg: EarsConfig, language: str = "fr"): ...
    def warm_up(self) -> None: ...  # load wake-word + STT models (download if needed)
    def wait_for_wake_word(
        self, stop: threading.Event, level_cb: Callable[[float], None] | None = None
    ) -> bool:
        """Block until the wake word is heard (True) or stop is set (False)."""

    def record_utterance(
        self,
        stop: threading.Event,
        level_cb: Callable[[float], None] | None = None,
        start_timeout: float | None = None,
    ) -> "np.ndarray | None":
        """Record 16 kHz mono float32 until `cfg.silence_seconds` of silence.
        Returns None if no speech starts within `start_timeout` seconds."""

    def transcribe(self, audio: "np.ndarray") -> str: ...
    def mute(self, muted: bool) -> None:
        """While True, ignore the microphone (assistant is speaking)."""
```

## voice.py — `Voice`
```python
class Voice:
    def __init__(self, cfg: VoiceConfig): ...
    def warm_up(self) -> None: ...  # load / download the TTS voice
    def speak_stream(
        self,
        chunks: Iterable[str],
        stop: threading.Event,
        level_cb: Callable[[float], None] | None = None,
        on_first_audio: Callable[[], None] | None = None,
    ) -> str:
        """Consume text deltas (as streamed by the LLM), cut them into
        sentences, synthesize sentence N+1 while sentence N plays.
        Returns the full text spoken. Stops quickly when `stop` is set."""

    def speak(self, text: str, stop: threading.Event, level_cb=None) -> None: ...
```

## brain.py — `Brain`
```python
class Brain:
    def __init__(self, cfg: JarvisConfig): ...
    def warm_up(self) -> None: ...  # preload the model in Ollama
    def stream_reply(self, user_text: str, stop: threading.Event) -> Iterator[str]:
        """Yield text deltas of the reply (no reasoning/thinking text, no
        markdown, spoken-style French). Records the exchange in history and
        in the persistent journal."""

    def greeting(
        self,
    ) -> str: ...  # short spoken line on startup, aware of time / last session
    def shutdown(self) -> None: ...  # write end-of-session note to the journal
```

## v2 additions

- `toolkit.py`: `Tool` (schema + callable, `confirm=True` = ask out loud
  first), `ToolBox`, `Notifier` (background work speaks up on its own).
- `actions.py`: `Reflexes(cfg, notifier).tools()/.close()` — instant local
  actions (apps, URLs, web search, weather, timers/reminders, music, volume,
  notes, clipboard).
- `hands.py`: `Hands(cfg, notifier).tools()/.close()` — long tasks delegated
  to the Claude Code CLI in a sandboxed per-task folder under `hands.workspace`.
- `Brain(cfg, toolbox)`: Ollama tool calling, confirmation flow,
  `record_spoken(text)` for announcements.
- `Ears.listen_for_interrupt(stop, threshold)`: barge-in while speaking.
- `voice_fx.py`: voice post-processing presets (`voice.effect`).
