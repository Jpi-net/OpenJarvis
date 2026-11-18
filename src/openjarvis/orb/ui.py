"""Floating animated orb: the only UI of the voice assistant.

A frameless, transparent, always-on-top pywebview window rendering
``orb.html`` (a 2D-canvas animation). Other threads drive it through
:meth:`OrbWindow.set_state` / :meth:`OrbWindow.set_level`; those calls only
store values, and a single pusher thread forwards changes to JavaScript at
most ~40 times per second, so callers never block on the GUI thread.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from openjarvis.orb.config import OrbConfig

logger = logging.getLogger("openjarvis.orb.ui")

_HTML_PATH = Path(__file__).with_name("orb.html")
_STATES = ("idle", "listening", "thinking", "speaking")
_DEFAULT_COLOR = "#4fc3ff"
_MARGIN = 24  # px between the orb window and the screen edges
_PUSH_INTERVAL = 1 / 40  # max JS bridge update rate (s)
_LEVEL_EPSILON = 0.01  # ignore level changes smaller than this


def _normalize_color(color: str) -> str:
    c = (color or "").strip()
    if re.fullmatch(r"#[0-9a-fA-F]{6}", c):
        return c.lower()
    if re.fullmatch(r"#[0-9a-fA-F]{3}", c):
        return "#" + "".join(ch * 2 for ch in c[1:]).lower()
    logger.warning("Invalid orb color %r, using %s", color, _DEFAULT_COLOR)
    return _DEFAULT_COLOR


def _screen_geometry() -> tuple[float, float, float, float, float, float] | None:
    """Return (full_w, full_h, vis_x, vis_top, vis_w, vis_h) of the primary
    screen in pywebview coordinates (origin top-left of the screen), where the
    visible area excludes the menu bar and the Dock. None if unavailable."""
    try:
        import AppKit  # type: ignore[import-not-found]

        screens = AppKit.NSScreen.screens()
        if not screens:
            return None
        scr = screens[0]
        full, vis = scr.frame(), scr.visibleFrame()
        fw, fh = full.size.width, full.size.height
        vis_x = vis.origin.x - full.origin.x
        vis_top = fh - ((vis.origin.y - full.origin.y) + vis.size.height)
        return fw, fh, vis_x, vis_top, vis.size.width, vis.size.height
    except Exception:  # noqa: BLE001 - non-macOS or AppKit unavailable
        pass
    try:
        import webview

        s = webview.screens[0]
        return s.width, s.height, 0, 0, s.width, s.height
    except Exception:  # noqa: BLE001
        logger.debug("Could not determine screen geometry", exc_info=True)
        return None


def _initial_position(position: str, size: int) -> tuple[int, int] | None:
    geo = _screen_geometry()
    if geo is None:
        return None
    _, _, vx, vy, vw, vh = geo
    pos = (position or "bottom-right").strip().lower()
    if pos == "center":
        return int(vx + (vw - size) / 2), int(vy + (vh - size) / 2)
    left = vx + _MARGIN
    right = vx + vw - size - _MARGIN
    top = vy + _MARGIN
    bottom = vy + vh - size - _MARGIN
    table = {
        "top-left": (left, top),
        "top-right": (right, top),
        "bottom-left": (left, bottom),
        "bottom-right": (right, bottom),
    }
    if pos not in table:
        logger.warning("Unknown orb position %r, using bottom-right", position)
    x, y = table.get(pos, table["bottom-right"])
    return int(x), int(y)


class OrbWindow:
    """The floating orb window. See CONTRACT.md (section ``ui.py``)."""

    def __init__(self, cfg: OrbConfig, title: str = "Jarvis") -> None:
        self._cfg = cfg
        self._title = title
        self._size = max(64, int(cfg.size))
        self._window: Any = None
        self._ready = threading.Event()
        self._closed = threading.Event()
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self._state = "idle"
        self._sent_state: str | None = None
        self._level = 0.0
        self._sent_level = -1.0

    # ------------------------------------------------------------------ API
    def run(self, worker: Callable[[], None]) -> None:
        """Create the window and block on the GUI loop (main thread only).

        ``worker`` is started in a daemon thread once the page is loaded.
        Returns when the window is closed.
        """
        import webview

        if self._closed.is_set():
            return
        html = _HTML_PATH.read_text(encoding="utf-8").replace(
            "__ORB_COLOR__", _normalize_color(self._cfg.color)
        )
        pos = _initial_position(self._cfg.position, self._size)
        x, y = pos if pos else (None, None)
        # Pin the window to the primary screen: by default pywebview uses the
        # screen that has keyboard focus, which may be another monitor.
        try:
            screen = webview.screens[0]
        except Exception:  # noqa: BLE001
            screen = None

        window = webview.create_window(
            self._title,
            html=html,
            width=self._size,
            height=self._size,
            x=x,
            y=y,
            screen=screen,
            resizable=False,
            frameless=True,
            easy_drag=True,
            shadow=False,
            on_top=True,
            transparent=True,
            background_color="#000000",
            focus=False,
            text_select=False,
        )
        self._window = window
        window.events.closed += self._on_closed
        webview.start(self._bootstrap, args=(worker,), debug=False)
        self._closed.set()
        self._wake.set()

    def set_state(self, state: str) -> None:
        """Switch the orb animation (thread-safe, non-blocking)."""
        if state not in _STATES:
            logger.warning("Unknown orb state %r ignored", state)
            return
        with self._lock:
            self._state = state
        self._wake.set()

    def set_level(self, level: float) -> None:
        """Feed the current audio amplitude 0..1 (thread-safe, non-blocking)."""
        try:
            v = float(level)
        except (TypeError, ValueError):
            return
        if not v > 0.0:  # also catches NaN
            v = 0.0
        elif v > 1.0:
            v = 1.0
        with self._lock:
            self._level = v
        self._wake.set()

    def close(self) -> None:
        """Destroy the window; :meth:`run` then returns."""
        if self._closed.is_set():
            return
        self._closed.set()
        self._wake.set()
        window = self._window
        if window is not None:
            try:
                window.destroy()
            except Exception:  # noqa: BLE001 - already gone
                logger.debug("Window destroy failed", exc_info=True)

    # ------------------------------------------------------------ internals
    def _on_closed(self) -> None:
        self._closed.set()
        self._wake.set()

    def _bootstrap(self, worker: Callable[[], None]) -> None:
        """Runs in a pywebview thread right after the GUI loop starts."""
        window = self._window
        window.events.loaded.wait()
        if self._closed.is_set():
            # close() was called before the window existed/loaded.
            try:
                window.destroy()
            except Exception:  # noqa: BLE001
                pass
            return
        self._hide_from_dock()
        self._ready.set()
        threading.Thread(target=self._pusher, name="orb-ui-pusher", daemon=True).start()
        threading.Thread(
            target=self._run_worker, args=(worker,), name="orb-worker", daemon=True
        ).start()

    @staticmethod
    def _run_worker(worker: Callable[[], None]) -> None:
        try:
            worker()
        except Exception:  # noqa: BLE001
            logger.exception("Orb worker crashed")

    @staticmethod
    def _hide_from_dock() -> None:
        """Make the app an accessory (no Dock icon / app-switcher entry)."""
        try:
            import AppKit  # type: ignore[import-not-found]
            from PyObjCTools import AppHelper  # type: ignore[import-not-found]

            def _apply() -> None:
                AppKit.NSApplication.sharedApplication().setActivationPolicy_(
                    AppKit.NSApplicationActivationPolicyAccessory
                )

            AppHelper.callAfter(_apply)
        except Exception:  # noqa: BLE001 - optional nicety
            logger.debug("Could not hide the orb from the Dock", exc_info=True)

    def _pusher(self) -> None:
        """Forward state/level changes to JS, coalesced and rate-limited."""
        while not self._closed.is_set():
            self._wake.wait()
            self._wake.clear()
            if self._closed.is_set():
                break
            with self._lock:
                state, level = self._state, self._level
            parts = []
            if state != self._sent_state:
                parts.append(f'orb.setState("{state}")')
                self._sent_state = state
            if abs(level - self._sent_level) >= _LEVEL_EPSILON or (
                level == 0.0 and self._sent_level != 0.0
            ):
                parts.append(f"orb.setLevel({level:.3f})")
                self._sent_level = level
            if parts:
                try:
                    self._window.run_js("window.orb&&(" + ",".join(parts) + ");")
                except Exception:  # noqa: BLE001 - window closing
                    logger.debug("run_js failed", exc_info=True)
            self._closed.wait(_PUSH_INTERVAL)
