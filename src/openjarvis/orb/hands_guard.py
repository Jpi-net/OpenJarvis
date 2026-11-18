"""PreToolUse guard for the orb's delegated Claude Code jobs.

Delegated jobs run with ``--permission-mode bypassPermissions`` so they can
use Bash freely; the OS sandbox confines Bash itself (writes to the task
folder, secrets unreadable, network allow-list). The file tools
(Write/Edit/MultiEdit/NotebookEdit, Read/Glob/Grep) do NOT go through that
sandbox, so this hook is their fence. Claude Code runs it before each tool
call (``hooks.PreToolUse`` in the ``--settings`` JSON built by ``hands.py``)::

    python -I hands_guard.py --root <task folder> --root <task temp dir>

Protocol (Claude Code hooks): the tool call arrives as JSON on stdin
(``tool_name``, ``tool_input``…). Exit 0 = no objection; exit 2 = block the
call, stderr is shown to the model as the reason. The guard fails closed: an
unreadable payload or an internal error also exits 2.

``--session-start`` mode only prints ``GUARD_TOKEN``: ``hands.py`` checks
that this SessionStart hook ran, proving the ``--settings`` JSON (sandbox + hooks)
was accepted (Claude Code silently ignores invalid settings in ``-p`` mode).

Standard library only, so ``python -I`` starts it fast and the job cannot
plant a module next to it.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

GUARD_TOKEN = "jarvis-guard: actif"
# Credentials the job must never read (file tools here, Bash via the sandbox).
SECRET_PATHS = [
    "~/.ssh",
    "~/.aws",
    "~/.gnupg",
    "~/.config/gh",
    "~/.config/gcloud",
    "~/.azure",
    "~/.kube",
    "~/.docker/config.json",
    "~/.netrc",
    "~/.git-credentials",
    "~/.npmrc",
    "~/.pypirc",
    "~/Library/Keychains",
    "~/Library/Cookies",
]
WRITE_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit"}
READ_TOOLS = {"Read", "Glob", "Grep"}
# Inside the task folder these would be loaded by Claude Code itself (hooks,
# settings) and run outside the sandbox: never writable by the job.
PROTECTED_NAMES = {".claude"}


def _real(path: str, base: str) -> str:
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(base, path)
    return os.path.realpath(path)


def _inside(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def check(payload: dict[str, Any], roots: list[str]) -> str | None:
    """Return a refusal reason, or None when the call may proceed."""
    tool = str(payload.get("tool_name") or "")
    args = payload.get("tool_input") or {}
    if not isinstance(args, dict):
        args = {}
    real_roots = [os.path.realpath(os.path.expanduser(r)) for r in roots if r]
    base = real_roots[0] if real_roots else os.getcwd()

    if tool == "Bash" and args.get("dangerouslyDisableSandbox"):
        return (
            "Refusé : les commandes doivent rester dans le bac à sable. "
            "Relance la commande sans dangerouslyDisableSandbox."
        )

    if tool in WRITE_TOOLS:
        raw = str(args.get("file_path") or args.get("notebook_path") or "")
        if not raw:
            return None  # the tool will reject the call itself
        target = _real(raw, base)
        root = next((r for r in real_roots if _inside(target, r)), None)
        if root is None:
            return (
                f"Refusé : {raw} est hors du dossier de la tâche. N'écris que "
                f"dans {base} (ou le dossier temporaire $TMPDIR)."
            )
        rel = os.path.relpath(target, root).split(os.sep)
        if PROTECTED_NAMES & set(rel):
            return "Refusé : le dossier .claude ne doit pas être modifié."
        return None

    if tool in READ_TOOLS:
        secrets = [os.path.realpath(os.path.expanduser(s)) for s in SECRET_PATHS]
        candidates = [args.get("file_path"), args.get("path")]
        pattern = str(args.get("pattern") or "")
        if tool == "Glob" and pattern.startswith(("/", "~")):
            candidates.append(pattern.split("*", 1)[0] or "/")
        # Searching a parent such as ~ stays allowed: the ``Read(...)`` deny
        # rules of the settings still hide the secret files from Glob/Grep.
        for raw in candidates:
            if raw and any(_inside(_real(str(raw), base), s) for s in secrets):
                return f"Refusé : {raw} contient des secrets, accès interdit."
        return None
    return None


def main(argv: list[str]) -> int:
    if "--session-start" in argv:
        print(GUARD_TOKEN)  # seen by hands.py in the hook_response event
        return 0
    roots = [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == "--root"]
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        if not isinstance(payload, dict):
            raise ValueError("payload is not an object")
        reason = check(payload, roots)
    except Exception as exc:  # noqa: BLE001 - fail closed
        print(f"Refusé : garde de sécurité indisponible ({exc}).", file=sys.stderr)
        return 2
    if reason:
        print(reason, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
