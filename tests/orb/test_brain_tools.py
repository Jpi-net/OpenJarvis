"""Tool-calling loop of the orb's brain, against a fake Ollama transport."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from openjarvis.orb import brain as brain_mod
from openjarvis.orb import config as orb_config
from openjarvis.orb.brain import Brain, classify_confirmation
from openjarvis.orb.config import JarvisConfig
from openjarvis.orb.toolkit import Tool, ToolBox


@pytest.fixture(autouse=True)
def _tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orb_config, "DATA_DIR", tmp_path / "orb")


class FakeOllama:
    """Scripted /api/chat: each entry is the list of stream chunks of one call."""

    def __init__(self, script: list[list[dict[str, Any]]]) -> None:
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        chunks = self.script.pop(0)
        lines = b"\n".join(json.dumps(c).encode() for c in chunks)
        return httpx.Response(200, content=lines)


def text(*parts: str) -> list[dict[str, Any]]:
    chunks = [{"message": {"role": "assistant", "content": p}} for p in parts]
    return chunks + [{"message": {"role": "assistant", "content": ""}, "done": True}]


def call(tool_name: str, cid: str = "call_1", **args: Any) -> list[dict[str, Any]]:
    tc = {"id": cid, "function": {"index": 0, "name": tool_name, "arguments": args}}
    return [
        {"message": {"role": "assistant", "content": "", "tool_calls": [tc]}},
        {"message": {"role": "assistant", "content": ""}, "done": True},
    ]


def make_brain(
    script: list[list[dict[str, Any]]], *tools: Tool
) -> tuple[Brain, FakeOllama]:
    fake = FakeOllama(script)
    box = ToolBox()
    box.add(*tools)
    b = Brain(
        JarvisConfig(), box if tools else None, transport=httpx.MockTransport(fake)
    )
    return b, fake


def reply(b: Brain, said: str) -> str:
    return "".join(b.stream_reply(said, threading.Event()))


def weather_tool(log: list[dict[str, Any]], delay: float = 0.0) -> Tool:
    def weather(place: str = "", when: str = "today") -> str:
        log.append({"place": place, "when": when})
        time.sleep(delay)
        return "Bruxelles demain : pluie faible, 4 à 9 °C."

    return Tool(
        "weather",
        "Météo",
        {"type": "object", "properties": {}},
        weather,
        progress="Je regarde la météo.",
    )


# ------------------------------------------------------------------ tests


def test_plain_chat_streams_and_sends_tools() -> None:
    b, fake = make_brain([text("Bonjour", " à vous.")], weather_tool([]))
    assert reply(b, "salut") == "Bonjour à vous."
    req = fake.requests[0]
    assert req["tools"][0]["function"]["name"] == "weather"
    assert req["think"] is False and req["stream"] is True


def test_no_toolbox_sends_no_tools() -> None:
    b, fake = make_brain([text("Salut.")])
    assert reply(b, "salut") == "Salut."
    assert "tools" not in fake.requests[0]


def test_tool_round_trip_messages() -> None:
    calls: list[dict[str, Any]] = []
    b, fake = make_brain(
        [call("weather", when="tomorrow"), text("Demain, pluie faible.")],
        weather_tool(calls),
    )
    assert reply(b, "quel temps demain ?") == "Demain, pluie faible."
    assert calls == [{"place": "", "when": "tomorrow"}]
    follow = fake.requests[1]["messages"]
    assert follow[-2]["tool_calls"][0]["function"]["name"] == "weather"
    assert follow[-1] == {
        "role": "tool",
        "content": "Bruxelles demain : pluie faible, 4 à 9 °C.",
        "tool_name": "weather",
        "tool_call_id": "call_1",
    }
    # history keeps the tool exchange, append-only
    b2_msgs = b._messages("suite")
    roles = [m["role"] for m in b2_msgs]
    assert roles[-5:] == ["user", "assistant", "tool", "assistant", "user"]
    # the next request extends the previous one exactly (prefix cache)
    assert b2_msgs[: len(follow)] == follow


def test_slow_tool_yields_progress_first() -> None:
    b, _ = make_brain([call("weather"), text("Il pleut.")], weather_tool([], delay=0.8))
    out = list(b.stream_reply("météo ?", threading.Event()))
    assert out[0] == "Je regarde la météo."
    assert "".join(out) == "Je regarde la météo. Il pleut."


def test_parallel_tool_calls_and_unknown_tool() -> None:
    calls: list[dict[str, Any]] = []
    tcs = [
        {"id": "a", "function": {"name": "weather", "arguments": {"place": "Paris"}}},
        {"id": "b", "function": {"name": "nope", "arguments": "{}"}},
    ]
    script = [
        [{"message": {"content": "", "tool_calls": tcs}, "done": True}],
        text("Voilà."),
    ]
    b, fake = make_brain(script, weather_tool(calls))
    assert reply(b, "météo à Paris") == "Voilà."
    tool_msgs = [m for m in fake.requests[1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["a", "b"]
    assert "inconnu" in tool_msgs[1]["content"]


def test_loop_is_bounded() -> None:
    script = [call("weather", cid=f"c{i}") for i in range(3)] + [text("Fini.")]
    b, fake = make_brain(script, weather_tool([]))
    assert reply(b, "boucle") == "Fini."
    assert len(fake.requests) == brain_mod.MAX_TOOL_ROUNDS
    # tools stay attached (stable cached prefix) but the last round can't act
    assert all("tools" in r for r in fake.requests)


def test_loop_gives_up_politely() -> None:
    script = [call("weather", cid=f"c{i}") for i in range(4)]
    ran: list[dict[str, Any]] = []
    b, fake = make_brain(script, weather_tool(ran))
    assert "pas réussi" in reply(b, "boucle")
    assert len(ran) == brain_mod.MAX_TOOL_ROUNDS - 1


def _quit_tool(ran: list[str]) -> Tool:
    return Tool(
        "quit_app",
        "Quitte une app",
        {"type": "object", "properties": {"name": {"type": "string"}}},
        lambda name: ran.append(name) or f"{name} est fermé.",
        confirm=True,
    )


def test_confirmation_yes_runs_tool() -> None:
    ran: list[str] = []
    b, fake = make_brain(
        [
            call("quit_app", name="Safari"),
            text("Je ferme Safari, je confirme ?"),
            text("Safari est fermé."),
        ],
        _quit_tool(ran),
    )
    assert reply(b, "ferme Safari") == "Je ferme Safari, je confirme ?"
    assert ran == []
    assert fake.requests[1]["tools"] == fake.requests[0]["tools"]  # stable prefix
    assert reply(b, "oui vas-y") == "Safari est fermé."
    assert ran == ["Safari"]
    assert len(fake.requests) == 3  # no extra model call to classify "oui"
    msgs = fake.requests[2]["messages"]
    assert msgs[-2]["tool_calls"][0]["function"]["arguments"] == {"name": "Safari"}
    assert msgs[-1]["content"] == "Safari est fermé."


def test_confirmation_no_cancels_without_model() -> None:
    ran: list[str] = []
    b, fake = make_brain(
        [call("quit_app", name="Mail"), text("Je ferme Mail, d'accord ?")],
        _quit_tool(ran),
    )
    reply(b, "ferme Mail")
    assert reply(b, "non laisse tomber") == "D'accord, j'annule."
    assert ran == [] and len(fake.requests) == 2


def test_confirmation_round_ignores_new_tool_calls() -> None:
    ran: list[str] = []
    b, _ = make_brain(
        [call("quit_app", name="Mail"), call("quit_app", cid="x", name="Mail")],
        _quit_tool(ran),
    )
    out = reply(b, "ferme Mail")
    assert out.endswith("je confirme ?") and ran == []
    assert b._pending is not None


def test_confirmation_template_when_model_forgets_to_ask() -> None:
    b, _ = make_brain(
        [call("quit_app", name="Mail"), text("D'accord.")], _quit_tool([])
    )
    out = reply(b, "ferme Mail")
    assert out.endswith("je confirme ?") and "Mail" in out


def test_other_answer_drops_pending() -> None:
    ran: list[str] = []
    b, fake = make_brain(
        [call("quit_app", name="Mail"), text("Je confirme ?"), text("Il est midi.")],
        _quit_tool(ran),
    )
    reply(b, "ferme Mail")
    assert reply(b, "quelle heure est-il ?") == "Il est midi."
    assert b._pending is None and ran == []


def test_pending_expires(monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[str] = []
    b, _ = make_brain(
        [call("quit_app", name="Mail"), text("Je confirme ?"), text("Oui quoi ?")],
        _quit_tool(ran),
    )
    reply(b, "ferme Mail")
    b._pending.expires = time.monotonic() - 1  # type: ignore[union-attr]
    assert reply(b, "oui") == "Oui quoi ?"
    assert ran == []


@pytest.mark.parametrize(
    ("said", "verdict"),
    [
        ("Oui.", "yes"),
        ("oui vas-y", "yes"),
        ("Euh, d'accord.", "yes"),
        ("go", "yes"),
        ("Je confirme !", "yes"),
        ("ok merci", "yes"),
        ("vas-y lance-le", "yes"),
        ("Non.", "no"),
        ("annule", "no"),
        ("Non, laisse tomber.", "no"),
        ("pas maintenant", "no"),
        ("oui mais plutôt demain à dix heures", None),
        ("non, mets-le à 18 h", None),
        ("quelle heure est-il ?", None),
        ("", None),
    ],
)
def test_classify_confirmation(said: str, verdict: str | None) -> None:
    assert classify_confirmation(said) == verdict


def test_record_spoken_is_append_only() -> None:
    b, fake = make_brain([text("Bonjour."), text("Bien noté.")])
    reply(b, "salut")
    before = b._messages("x")[:-1]
    b.record_spoken("Rappel : boire de l'eau.")
    after = b._messages("x")[:-1]
    assert after[: len(before)] == before
    assert after[-1] == {"role": "assistant", "content": "Rappel : boire de l'eau."}
    journal = (orb_config.DATA_DIR / "journal.jsonl").read_text().splitlines()
    assert json.loads(journal[-1])["type"] == "spoken"


def test_tool_args_are_filtered_and_errors_reported() -> None:
    def boom(place: str = "") -> str:
        raise RuntimeError("réseau coupé")

    tool = Tool("weather", "Météo", {"type": "object", "properties": {}}, boom)
    b, fake = make_brain(
        [call("weather", place="Liège", bogus=1), text("La météo a échoué.")], tool
    )
    assert reply(b, "météo") == "La météo a échoué."
    tool_msg = fake.requests[1]["messages"][-1]
    assert "réseau coupé" in tool_msg["content"]


def test_action_request_without_tool_is_nudged_once() -> None:
    ran: list[dict[str, Any]] = []
    b, fake = make_brain(
        [
            text("Je vérifie la météo."),
            call("weather", when="tomorrow"),
            text("Pluie."),
        ],
        weather_tool(ran),
    )
    assert reply(b, "Quelle météo demain ?") == "Pluie."
    assert ran == [{"place": "", "when": "tomorrow"}]
    assert fake.requests[1]["messages"][-1]["content"] == brain_mod.ACTION_NUDGE
    # the nudge is not kept in history (only the successful call is)
    assert all(m["content"] != brain_mod.ACTION_NUDGE for m in b._messages("x"))


def test_action_request_still_talking_after_nudge_keeps_first_text() -> None:
    b, fake = make_brain(
        [text("Je ne peux pas faire ça."), text("Toujours non.")], weather_tool([])
    )
    assert reply(b, "ouvre la porte du garage") == "Je ne peux pas faire ça."
    assert len(fake.requests) == 2


def test_plain_chat_is_not_held() -> None:
    b, fake = make_brain([text("Très bien.")], weather_tool([]))
    assert reply(b, "comment vas-tu ?") == "Très bien."
    assert len(fake.requests) == 1
