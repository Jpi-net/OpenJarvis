"""The orb's main loop: wake word → listen → think → speak, forever.

Threading model: the GUI loop owns the main thread (macOS requirement); the
whole conversation loop runs in one worker thread started by ``OrbWindow``.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx

from openjarvis.orb.config import DATA_DIR, JarvisConfig

log = logging.getLogger("openjarvis.orb.app")


def ensure_ollama(host: str, timeout: float = 15.0) -> bool:
    """Make sure the Ollama server answers; start it in the background if not."""

    def alive() -> bool:
        try:
            return httpx.get(f"{host}/api/version", timeout=1.0).status_code == 200
        except httpx.HTTPError:
            return False

    if alive():
        return True
    if "127.0.0.1" not in host and "localhost" not in host:
        return False
    exe = shutil.which("ollama")
    if exe is None:
        return False
    log.info("Starting Ollama in the background")
    with open(DATA_DIR / "ollama.log", "ab") as out:
        subprocess.Popen(  # noqa: S603 - fixed local binary
            [exe, "serve"], stdout=out, stderr=out, start_new_session=True
        )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if alive():
            return True
        time.sleep(0.3)
    return False


class _NullUI:
    """Stand-in for the orb window in headless mode."""

    def set_state(self, state: str) -> None:
        log.info("state: %s", state)

    def set_level(self, level: float) -> None:
        pass

    def close(self) -> None:
        pass


class OrbApp:
    def __init__(
        self, cfg: JarvisConfig, *, headless: bool = False, text_input: bool = False
    ):
        self.cfg = cfg
        self.headless = headless
        self.text_input = text_input
        self.stop = threading.Event()
        self.ui: Any = _NullUI()
        self.ears: Any = None
        self.voice: Any = None
        self.brain: Any = None

    # -- setup -------------------------------------------------------------

    def _warm_up(self) -> None:
        from openjarvis.orb.brain import Brain
        from openjarvis.orb.voice import Voice

        self.brain = Brain(self.cfg)
        self.voice = Voice(self.cfg.voice)
        parts: list[Any] = [self.brain, self.voice]
        if not self.text_input:
            from openjarvis.orb.ears import Ears

            self.ears = Ears(self.cfg.ears, language=self.cfg.assistant.language)
            parts.append(self.ears)

        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=len(parts)) as pool:
            for future in [pool.submit(p.warm_up) for p in parts]:
                future.result()
        log.info("Warm-up done in %.1fs", time.monotonic() - started)

    # -- conversation ------------------------------------------------------

    def _say(self, chunks: Any) -> None:
        """Speak a reply (iterable of text deltas) with the orb in sync."""
        if self.ears is not None:
            self.ears.mute(True)
        try:
            self.voice.speak_stream(
                chunks,
                self.stop,
                level_cb=self.ui.set_level,
                on_first_audio=lambda: self.ui.set_state("speaking"),
            )
        finally:
            self.ui.set_level(0.0)
            if self.ears is not None:
                self.ears.mute(False)

    def _answer(self, text: str) -> None:
        log.info("user: %s", text)
        self.ui.set_state("thinking")
        self._say(self.brain.stream_reply(text, self.stop))

    def _voice_loop(self) -> None:
        follow_up = self.cfg.ears.follow_up_seconds
        while not self.stop.is_set():
            self.ui.set_state("idle")
            if not self.ears.wait_for_wake_word(self.stop):
                return
            start_timeout = max(follow_up, 5.0)
            # Conversation: keep listening without the wake word while the
            # user keeps talking within the follow-up window.
            while not self.stop.is_set():
                self.ui.set_state("listening")
                audio = self.ears.record_utterance(
                    self.stop, level_cb=self.ui.set_level, start_timeout=start_timeout
                )
                self.ui.set_level(0.0)
                if audio is None:
                    break
                self.ui.set_state("thinking")
                text = self.ears.transcribe(audio).strip()
                if not text:
                    break
                self._answer(text)
                start_timeout = follow_up

    def _text_loop(self) -> None:
        while not self.stop.is_set():
            self.ui.set_state("idle")
            try:
                text = input("vous> ").strip()
            except EOFError:
                return
            if text:
                self._answer(text)

    def worker(self) -> None:
        try:
            self.ui.set_state("thinking")
            if not ensure_ollama(self.cfg.brain.host):
                log.error("Ollama is not reachable at %s", self.cfg.brain.host)
            self._warm_up()
            greeting = self.brain.greeting()
            if greeting:
                self.ui.set_state("thinking")
                self._say(iter([greeting]))
            if self.text_input:
                self._text_loop()
            else:
                self._voice_loop()
        except Exception:
            log.exception("Orb worker crashed")
        finally:
            self.shutdown()

    # -- lifecycle ---------------------------------------------------------

    def shutdown(self) -> None:
        if getattr(self, "_shut", False):
            return
        self._shut = True
        self.stop.set()
        if self.brain is not None:
            try:
                self.brain.shutdown()
            except Exception:
                log.exception("Brain shutdown failed")
        for part in (self.ears, self.voice):
            if part is not None and hasattr(part, "close"):
                try:
                    part.close()
                except Exception:
                    log.exception("Closing %s failed", type(part).__name__)
        self.ui.close()

    def run(self) -> None:
        if self.headless:
            try:
                self.worker()
            except KeyboardInterrupt:
                self.shutdown()
            return
        from openjarvis.orb.ui import OrbWindow

        self.ui = OrbWindow(self.cfg.orb, title=self.cfg.assistant.name)
        try:
            self.ui.run(self.worker)
        finally:
            # Window closed by the user (or Ctrl+C): stop everything.
            self.stop.set()
            self.shutdown()
