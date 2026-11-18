"""Unit tests for the PreToolUse guard of delegated Claude Code jobs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from openjarvis.orb import hands_guard as guard

GUARD = Path(guard.__file__)


@pytest.fixture
def roots(tmp_path: Path) -> list[str]:
    task = tmp_path / "task"
    tmp = tmp_path / "tmp"
    task.mkdir()
    tmp.mkdir()
    return [str(task), str(tmp)]


def _run(payload: object, roots: list[str]) -> subprocess.CompletedProcess[str]:
    args = [a for r in roots for a in ("--root", r)]
    return subprocess.run(
        [sys.executable, "-I", str(GUARD), *args],
        input=payload if isinstance(payload, str) else json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=20,
    )


def _write(path: str, tool: str = "Write") -> dict:
    key = "notebook_path" if tool == "NotebookEdit" else "file_path"
    return {
        "hook_event_name": "PreToolUse",
        "tool_name": tool,
        "tool_input": {key: path, "content": "x"},
    }


@pytest.mark.parametrize("tool", ["Write", "Edit", "MultiEdit", "NotebookEdit"])
def test_writes_inside_the_task_folder_or_tmp_pass(roots, tool) -> None:
    assert guard.check(_write(f"{roots[0]}/src/app.py", tool), roots) is None
    assert guard.check(_write(f"{roots[1]}/scratch.txt", tool), roots) is None
    assert guard.check(_write("relative/file.txt", tool), roots) is None


@pytest.mark.parametrize("tool", ["Write", "Edit", "MultiEdit", "NotebookEdit"])
def test_writes_outside_are_denied(roots, tool) -> None:
    for path in (
        "~/jarvis_guard_probe.txt",
        "/etc/hosts",
        f"{roots[0]}/../escape.txt",
        f"{roots[0]}-sibling/x.txt",  # prefix trick
        "../outside.txt",
        str(Path.home() / ".zshrc"),
    ):
        reason = guard.check(_write(path, tool), roots)
        assert reason and "hors du dossier" in reason, path


def test_symlink_out_of_the_folder_is_denied(roots, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (Path(roots[0]) / "link").symlink_to(outside)
    assert guard.check(_write(f"{roots[0]}/link/x.txt"), roots)


def test_dot_claude_in_the_task_folder_is_protected(roots) -> None:
    reason = guard.check(_write(f"{roots[0]}/.claude/settings.json"), roots)
    assert reason and ".claude" in reason
    assert guard.check(_write(f"{roots[0]}/.mcp.json"), roots) is None


def test_secret_reads_are_denied(roots) -> None:
    home = Path.home()
    for tool, args in (
        ("Read", {"file_path": str(home / ".ssh" / "id_ed25519")}),
        ("Read", {"file_path": "~/.aws/credentials"}),
        ("Read", {"file_path": "~/.npmrc"}),
        ("Grep", {"pattern": "token", "path": "~/.config/gh"}),
        ("Glob", {"pattern": "~/.ssh/*"}),
        ("Glob", {"pattern": "*", "path": str(home / "Library" / "Keychains")}),
    ):
        payload = {"tool_name": tool, "tool_input": args}
        reason = guard.check(payload, roots)
        assert reason and "secrets" in reason, (tool, args)


def test_ordinary_reads_pass(roots) -> None:
    for tool, args in (
        ("Read", {"file_path": "/etc/hosts"}),
        ("Read", {"file_path": f"{roots[0]}/README.md"}),
        ("Grep", {"pattern": "def", "path": str(Path.home() / "Documents")}),
        ("Glob", {"pattern": "**/*.py"}),
    ):
        assert guard.check({"tool_name": tool, "tool_input": args}, roots) is None


def test_bash_must_stay_sandboxed(roots) -> None:
    ok = {"tool_name": "Bash", "tool_input": {"command": "npm install"}}
    assert guard.check(ok, roots) is None
    bad = {
        "tool_name": "Bash",
        "tool_input": {"command": "rm -rf ~", "dangerouslyDisableSandbox": True},
    }
    assert "bac à sable" in guard.check(bad, roots)


def test_other_tools_pass(roots) -> None:
    payload = {
        "tool_name": "mcp__claude_ai_Notion__notion-search",
        "tool_input": {"query": "x"},
    }
    assert guard.check(payload, roots) is None


def test_hook_protocol_exit_codes(roots) -> None:
    allowed = _run(_write(f"{roots[0]}/ok.txt"), roots)
    assert allowed.returncode == 0 and allowed.stderr == ""
    denied = _run(_write("~/jarvis_guard_probe.txt"), roots)
    assert denied.returncode == 2 and "hors du dossier" in denied.stderr
    broken = _run("not json", roots)  # fails closed
    assert broken.returncode == 2 and "indisponible" in broken.stderr
    assert not (Path.home() / "jarvis_guard_probe.txt").exists()


def test_session_start_token() -> None:
    out = subprocess.run(
        [sys.executable, "-I", str(GUARD), "--session-start"],
        input="{}",
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert out.returncode == 0 and guard.GUARD_TOKEN in out.stdout
