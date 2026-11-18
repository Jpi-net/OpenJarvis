"""Shared plumbing for the orb's abilities.

- ``Tool``: one ability the brain can call (Ollama/OpenAI function-calling
  schema + a Python callable returning a short French result string).
- ``ToolBox``: the registry handed to the brain.
- ``Notifier``: lets background work (reminders, finished tasks) speak up on
  its own; the app loop drains it whenever the assistant is not talking.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger("openjarvis.orb.toolkit")


@dataclass
class Tool:
    name: str
    description: str  # French or English; tells the model WHEN to use it
    parameters: dict[str, Any]  # JSON schema ({"type": "object", ...})
    func: Callable[..., str]  # returns a short result for the model
    # True = side effects the user must approve out loud before running
    # (the brain asks "je lance … ?" and runs it only after a yes).
    confirm: bool = False
    # Optional short spoken line while the tool runs ("Je regarde…").
    progress: str = ""

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def run(self, **kwargs: Any) -> str:
        try:
            return str(self.func(**kwargs))
        except Exception as exc:  # a tool must never crash the brain
            log.exception("Tool %s failed", self.name)
            return f"Erreur de l'outil {self.name} : {exc}"


@dataclass
class ToolBox:
    tools: dict[str, Tool] = field(default_factory=dict)

    def add(self, *tools: Tool) -> None:
        for tool in tools:
            self.tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self.tools.values()]

    def __bool__(self) -> bool:
        return bool(self.tools)


@dataclass
class Announcement:
    text: str
    urgent: bool = False  # True = may interrupt (e.g. a timer ringing)


class Notifier:
    """Thread-safe queue of things the assistant should say on its own."""

    def __init__(self) -> None:
        self._queue: queue.Queue[Announcement] = queue.Queue()
        self.pending = threading.Event()  # set while something waits

    def announce(self, text: str, *, urgent: bool = False) -> None:
        text = text.strip()
        if not text:
            return
        log.info("announce%s: %s", " (urgent)" if urgent else "", text)
        self._queue.put(Announcement(text, urgent))
        self.pending.set()

    def drain(self) -> list[Announcement]:
        items: list[Announcement] = []
        while True:
            try:
                items.append(self._queue.get_nowait())
            except queue.Empty:
                break
        self.pending.clear()
        if not self._queue.empty():  # raced with a producer
            self.pending.set()
        return items
