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
from openjarvis.orb.toolkit import Notifier, ToolBox

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
        self.notifier = Notifier()
        self._closables: list[Any] = []  # reflexes / hands background threads
        self._speak_lock = threading.Lock()

    # -- setup -------------------------------------------------------------

    def _build_toolbox(self) -> ToolBox:
        """Reflexes (instant local actions) + hands (delegated long tasks)."""
        box = ToolBox()
        if self.cfg.actions.enabled:
            try:
                from openjarvis.orb.actions import Reflexes

                reflexes = Reflexes(self.cfg, self.notifier)
                box.add(*reflexes.tools())
                self._closables.append(reflexes)
            except Exception:
                log.exception("Reflexes unavailable")
        if self.cfg.hands.enabled:
            try:
                from openjarvis.orb.hands import Hands

                hands = Hands(self.cfg, self.notifier)
                box.add(*hands.tools())
                self._closables.append(hands)
            except Exception:
                log.exception("Hands unavailable")
        log.info("Abilities: %s", ", ".join(box.tools) or "none")
        return box

    def _warm_up(self) -> None:
        from openjarvis.orb.brain import Brain
        from openjarvis.orb.voice import Voice

        self.brain = Brain(self.cfg, self._build_toolbox())
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

    def _say(self, chunks: Any) -> bool:
        """Speak a reply (iterable of text deltas) with the orb in sync.

        Returns True if the user interrupted by saying the wake word.
        """
        interrupt = threading.Event()  # stops the voice
        done = threading.Event()  # stops the helper threads
        barged = threading.Event()

        def relay_stop() -> None:  # app shutdown also stops the voice
            while not done.is_set():
                if self.stop.wait(0.1):
                    interrupt.set()
                    return

        def watch_barge_in() -> None:
            if self.ears.listen_for_interrupt(done, self.cfg.ears.barge_in_threshold):
                barged.set()
                interrupt.set()

        with self._speak_lock:
            if self.ears is not None:
                self.ears.mute(True)
            helpers = [threading.Thread(target=relay_stop, daemon=True)]
            if self.ears is not None and self.cfg.ears.barge_in:
                helpers.append(threading.Thread(target=watch_barge_in, daemon=True))
            for t in helpers:
                t.start()
            try:
                self.voice.speak_stream(
                    chunks,
                    interrupt,
                    level_cb=self.ui.set_level,
                    on_first_audio=lambda: self.ui.set_state("speaking"),
                )
            finally:
                done.set()
                for t in helpers:
                    t.join(timeout=1.0)
                self.ui.set_level(0.0)
                if self.ears is not None:
                    self.ears.mute(False)
        return barged.is_set()

    def _announce_pending(self) -> bool:
        """Say what background work wants to say. True if something was said."""
        items = self.notifier.drain()
        if not items:
            return False
        text = " ".join(a.text for a in items)
        self.ui.set_state("thinking")
        self._say(iter([text]))
        if hasattr(self.brain, "record_spoken"):
            self.brain.record_spoken(text)
        return True

    def _answer(self, text: str) -> bool:
        log.info("user: %s", text)
        self.ui.set_state("thinking")
        return self._say(self.brain.stream_reply(text, self.stop))

    def _wait_wake_or_news(self) -> str:
        """Block until the wake word ("wake"), an announcement ("news") or
        shutdown ("stop")."""
        wake_stop = threading.Event()

        def relay() -> None:
            while not wake_stop.is_set():
                if self.stop.is_set() or self.notifier.pending.is_set():
                    wake_stop.set()
                    return
                time.sleep(0.05)

        threading.Thread(target=relay, daemon=True).start()
        try:
            heard = self.ears.wait_for_wake_word(wake_stop)
        finally:
            wake_stop.set()
        if heard:
            return "wake"
        return "stop" if self.stop.is_set() else "news"

    def _voice_loop(self) -> None:
        follow_up = self.cfg.ears.follow_up_seconds
        while not self.stop.is_set():
            self.ui.set_state("idle")
            event = self._wait_wake_or_news()
            if event == "stop":
                return
            if event == "news":
                if not self._announce_pending():
                    continue
                start_timeout = follow_up  # let the user react to the news
            else:
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
                self._answer(text)  # barge-in or not, listen right after
                self._announce_pending()
                start_timeout = follow_up

    def _text_loop(self) -> None:
        def announcer() -> None:
            while not self.stop.is_set():
                if self.notifier.pending.wait(0.2):
                    self._announce_pending()
                    self.ui.set_state("idle")

        threading.Thread(target=announcer, daemon=True).start()
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
        for part in (*self._closables, self.ears, self.voice):
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
