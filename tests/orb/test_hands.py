"""Tests for the orb's hands (Claude Code delegation), using a fake `claude`."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from openjarvis.orb import hands as hands_mod
from openjarvis.orb.config import JarvisConfig
from openjarvis.orb.hands import Hands, Task, slugify, spoken_duration
from openjarvis.orb.toolkit import Notifier

FAKE_CLAUDE = r"""#!{python}
import json, os, sys, time

if "--help" in sys.argv:
    print("--output-format --verbose --permission-mode --settings "
          "--append-system-prompt --permission-prompts --setting-sources "
          "--strict-mcp-config --model --max-budget-usd --allowedTools")
    sys.exit(0)

mode = os.environ.get("FAKE_CLAUDE_MODE", "ok")
prompt = sys.stdin.read()
with open(os.environ["FAKE_CLAUDE_ARGS"], "a") as f:
    f.write(json.dumps({"argv": sys.argv[1:], "prompt": prompt,
                        "cwd": os.getcwd()}) + "\n")

def emit(event):
    print(json.dumps(event), flush=True)

cwd = os.getcwd()
emit({"type": "system", "subtype": "init", "session_id": "sess-1"})
emit({"type": "assistant", "message": {"content": [
    {"type": "text", "text": "Je commence."},
    {"type": "tool_use", "name": "TodoWrite", "input": {"todos": [
        {"content": "Ecrire la page", "activeForm": "Écriture de la page",
         "status": "in_progress"}]}}]}})
emit({"type": "assistant", "message": {"content": [
    {"type": "tool_use", "name": "Write",
     "input": {"file_path": "/Users/nobody/outside.txt", "content": "x"}}]}})
if mode == "slow":
    time.sleep(60)
with open(os.path.join(cwd, "index.html"), "w") as f:
    f.write("<h1>12:00</h1>")
emit({"type": "assistant", "message": {"content": [
    {"type": "tool_use", "name": "Write",
     "input": {"file_path": os.path.join(cwd, "index.html"), "content": "x"}}]}})
emit({"type": "user", "message": {"content": [
    {"type": "tool_result", "content": "ok"}]}})
if mode == "hold":
    time.sleep(float(os.environ.get("FAKE_CLAUDE_HOLD", "1")))
if mode == "fail":
    print("Error: something broke", file=sys.stderr)
    sys.exit(1)
if mode == "budget":
    emit({"type": "result", "subtype": "error_max_budget_usd", "is_error": True,
          "total_cost_usd": 0.5})
    sys.exit(1)
emit({"type": "result", "subtype": "success", "is_error": False,
      "total_cost_usd": 0.1234, "duration_ms": 1500,
      "result": "J'ai créé une **page** avec l'heure en grand, dans " + cwd + "."})
"""


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake = tmp_path / "claude"
    fake.write_text(FAKE_CLAUDE.replace("{python}", sys.executable))
    fake.chmod(0o755)
    args_file = tmp_path / "args.jsonl"
    monkeypatch.setenv("FAKE_CLAUDE_ARGS", str(args_file))
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(fake))  # skip keychain export
    cfg = JarvisConfig()
    cfg.hands.workspace = str(tmp_path / "ws")
    cfg.hands.max_parallel = 2
    cfg.hands.max_budget_usd = 0.5
    data = tmp_path / "data"
    notifier = Notifier()

    def make() -> Hands:
        return Hands(cfg, notifier, claude_path=str(fake), data_dir=data)

    return {
        "cfg": cfg,
        "make": make,
        "notifier": notifier,
        "args": args_file,
        "data": data,
        "ws": tmp_path / "ws",
    }


def _wait(cond, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return
        time.sleep(0.02)
    raise AssertionError("condition not met in time")


def _wait_announce(notifier: Notifier, timeout: float = 10.0) -> list[str]:
    assert notifier.pending.wait(timeout)
    return [a.text for a in notifier.drain()]


def test_slugify_and_durations() -> None:
    assert slugify("Site de la Boulangerie !") == "site-de-la-boulangerie"
    assert slugify("Écran d'accueil — été") == "ecran-d-accueil-ete"
    assert slugify("!!!") == "tache"
    assert len(slugify("mot " * 30)) <= 40
    assert spoken_duration(20) == "moins d'une minute"
    assert spoken_duration(185) == "3 minutes"
    assert spoken_duration(3900) == "1 heure 5"


def test_tools_schema(env) -> None:
    tools = {t.name: t for t in env["make"]().tools()}
    assert set(tools) == {"delegate_task", "task_status", "cancel_task", "task_result"}
    assert tools["delegate_task"].confirm is True
    assert tools["cancel_task"].confirm is True
    assert tools["delegate_task"].parameters["required"] == ["goal", "title"]


def test_full_run_announces_and_persists(env) -> None:
    hands = env["make"]()
    ack = hands.delegate_task("Crée une page qui affiche l'heure", "Horloge géante")
    assert ack.startswith("C'est lancé.")
    texts = _wait_announce(env["notifier"])
    assert len(texts) == 1
    spoken = texts[0]
    assert "Horloge géante" in spoken and "terminée" in spoken
    assert "**" not in spoken
    assert "ton dossier ws, sous-dossier horloge geante" in spoken
    assert str(env["ws"]) not in spoken

    folders = list(env["ws"].iterdir())
    assert len(folders) == 1
    folder = folders[0]
    assert folder.name.endswith("-horloge-geante")
    assert folder.name[:4].isdigit() and folder.name[10] == "-"
    assert (folder / "index.html").is_file()

    call = json.loads(env["args"].read_text().splitlines()[0])
    argv = call["argv"]
    assert Path(call["cwd"]).resolve() == folder.resolve()
    assert "Crée une page" in call["prompt"]
    assert argv[argv.index("--output-format") + 1] == "stream-json"
    assert argv[argv.index("--permission-mode") + 1] == "acceptEdits"
    assert argv[argv.index("--max-budget-usd") + 1] == "0.5"
    assert argv[argv.index("--permission-prompts") + 1] == "none"
    settings = json.loads(argv[argv.index("--settings") + 1])
    assert settings["sandbox"]["enabled"] is True
    assert settings["sandbox"]["allowUnsandboxedCommands"] is False
    allowed = argv[argv.index("--allowedTools") + 1].split(",")
    assert "Write" not in allowed and "Edit" not in allowed
    assert "Bash" not in allowed
    assert "--model" not in argv

    _wait(lambda: hands.task_status().find("terminée") >= 0)
    status = hands.task_status()
    assert "index.html" in status
    result = hands.task_result()
    assert "heure en grand" in result
    assert "file://" in result and "index.html" in result

    saved = json.loads((env["data"] / "tasks.json").read_text())
    assert saved[0]["status"] == "done"
    assert saved[0]["cost_usd"] == pytest.approx(0.1234)
    assert saved[0]["files"] == ["index.html"]
    assert saved[0]["title"] == "Horloge géante"
    assert hands.active_summary() == ""


def test_progress_while_running(env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "hold")
    monkeypatch.setenv("FAKE_CLAUDE_HOLD", "1.5")
    env["cfg"].hands.model = "sonnet"
    hands = env["make"]()
    hands.delegate_task("Fais un site", "site boulangerie")
    _wait(lambda: "index.html" in hands.task_status())
    status = hands.task_status("boulangerie")
    assert "en cours depuis moins d'une minute" in status
    assert "il écrit index.html" in status
    assert hands.active_summary() == "1 tâche en cours : site boulangerie, 0 min"
    _wait_announce(env["notifier"])
    argv = json.loads(env["args"].read_text().splitlines()[0])["argv"]
    assert argv[argv.index("--model") + 1] == "sonnet"


def test_cancel_kills_job(env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "slow")
    hands = env["make"]()
    hands.delegate_task("Long travail", "long travail")
    task = hands._running()[0]
    _wait(lambda: task.tool_uses >= 2)
    start = time.monotonic()
    msg = hands.cancel_task("")
    assert "J'ai arrêté « long travail »" in msg
    assert time.monotonic() - start < 6
    assert task.proc.poll() is not None
    assert hands.cancel_task("") == "Aucune tâche n'est en cours."
    time.sleep(0.3)
    assert not env["notifier"].pending.is_set()  # no announcement on cancel
    saved = json.loads((env["data"] / "tasks.json").read_text())
    assert saved[0]["status"] == "cancelled"


def test_max_parallel_and_close(env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "slow")
    env["cfg"].hands.max_parallel = 1
    hands = env["make"]()
    assert hands.delegate_task("A", "premier").startswith("C'est lancé")
    refused = hands.delegate_task("B", "second")
    assert "déjà 1 tâche en cours" in refused and "premier" in refused
    assert len(list(env["ws"].iterdir())) == 1
    procs = [t.proc for t in hands._running()]
    hands.close()
    assert all(p.poll() is not None for p in procs)
    saved = json.loads((env["data"] / "tasks.json").read_text())
    assert saved[0]["status"] == "interrupted"


def test_running_task_found_at_startup_is_interrupted(env) -> None:
    data: Path = env["data"]
    data.mkdir(parents=True)
    task = Task(id="abc", title="vieux", goal="g", folder="/x/2026-01-01-vieux")
    (data / "tasks.json").write_text(json.dumps([task.to_json()]))
    hands = env["make"]()
    assert "interrompue" in hands.task_status("vieux")
    saved = json.loads((data / "tasks.json").read_text())
    assert saved[0]["status"] == "interrupted" and saved[0]["ended"]


def test_failures_are_announced(env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "budget")
    hands = env["make"]()
    hands.delegate_task("X", "budget")
    text = _wait_announce(env["notifier"])[0]
    assert "n'a pas abouti" in text and "budget de 0.5 dollars" in text

    monkeypatch.setenv("FAKE_CLAUDE_MODE", "fail")
    hands.delegate_task("Y", "panne")
    text = _wait_announce(env["notifier"])[0]
    assert "panne" in text and "code 1" in text


def test_same_title_twice_gets_new_folder(env) -> None:
    hands = env["make"]()
    hands.delegate_task("A", "même nom")
    _wait_announce(env["notifier"])
    hands.delegate_task("B", "même nom")
    _wait_announce(env["notifier"])
    names = sorted(p.name for p in env["ws"].iterdir())
    assert names[0].endswith("-meme-nom") and names[1].endswith("-meme-nom-2")


def test_missing_claude(env, monkeypatch: pytest.MonkeyPatch) -> None:
    hands = Hands(
        env["cfg"],
        env["notifier"],
        claude_path="/nonexistent/claude",
        data_dir=env["data"],
    )
    assert "Claude Code" in hands.delegate_task("x", "y")
    assert hands_mod.find_claude("/nonexistent/claude") is None
