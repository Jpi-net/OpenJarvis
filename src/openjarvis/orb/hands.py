"""The orb's "hands": long tasks delegated to Claude Code in the background.

The local model can talk, but it cannot reliably build software or wire up a
service. When the user asks for that, the brain calls ``delegate_task``; we
start ``claude -p`` (headless Claude Code) in a fresh folder under the
workspace (``~/Jarvis/<YYYY-MM-DD>-<slug>/``), return at once, follow its
``stream-json`` events in a reader thread and speak up through the
``Notifier`` when the job is done.

Safety model (verified on macOS with Claude Code 2.1.327):

- ``--permission-mode acceptEdits``: the Write/Edit tools are auto-approved
  only inside the working directory (the task folder). Write/Edit are *not*
  put in ``--allowedTools`` on purpose: an allow rule would approve them for
  any path on disk.
- Built-in sandbox (``--settings`` ``sandbox.enabled``, macOS Seatbelt) for
  every Bash command: writes confined to the task folder (+ temp dirs),
  secrets unreadable, network limited to a developer allow-list (package
  registries, GitHub, CDNs…). ``autoAllowBashIfSandboxed`` lets commands run
  without prompts; ``allowUnsandboxedCommands: false`` removes the escape
  hatch; ``failIfUnavailable`` refuses to start rather than run unsandboxed.
- ``--permission-prompts none``: anything that would need a human approval is
  denied automatically, so a job never hangs waiting for an answer.
- ``--setting-sources project``: the user's personal Claude Code settings
  (hooks waiting on a GUI, plugins, permissive allow rules) do not apply.
- WebFetch / WebSearch are allowed so the job can read documentation.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from openjarvis.orb import config as orb_config
from openjarvis.orb.config import JarvisConfig
from openjarvis.orb.toolkit import Notifier, Tool

log = logging.getLogger("openjarvis.orb.hands")

CLAUDE_PATH_ENV = "JARVIS_CLAUDE_PATH"
MAX_TASK_SECONDS = 60 * 60  # hard wall-clock limit per job
KEEP_TASKS = 50  # entries kept in tasks.json

# Hosts sandboxed shell commands may reach (WebFetch is not concerned).
ALLOWED_DOMAINS = [
    "localhost",
    "github.com",
    "*.github.com",
    "*.githubusercontent.com",
    "ghcr.io",
    "gitlab.com",
    "bitbucket.org",
    "registry.npmjs.org",
    "*.npmjs.org",
    "registry.yarnpkg.com",
    "*.yarnpkg.com",
    "nodejs.org",
    "bun.sh",
    "pypi.org",
    "*.pypi.org",
    "files.pythonhosted.org",
    "*.pythonhosted.org",
    "astral.sh",
    "*.astral.sh",
    "crates.io",
    "*.crates.io",
    "proxy.golang.org",
    "sum.golang.org",
    "formulae.brew.sh",
    "cdn.jsdelivr.net",
    "unpkg.com",
    "cdnjs.cloudflare.com",
    "fonts.googleapis.com",
    "fonts.gstatic.com",
    "huggingface.co",
    "*.huggingface.co",
    "*.hf.co",
]
SECRET_PATHS = ["~/.ssh", "~/.aws", "~/.gnupg", "~/.config/gh", "~/Library/Keychains"]
# Tools approved without a prompt. Write/Edit deliberately absent (see above).
ALLOWED_TOOLS = ["Read", "Glob", "Grep", "WebFetch", "WebSearch", "TodoWrite"]
EDIT_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit"}

STATUS_FR = {
    "running": "en cours",
    "done": "terminée",
    "failed": "échouée",
    "cancelled": "annulée",
    "interrupted": "interrompue",
}

_MD_RE = re.compile(r"[*`#>|]+")
_EMPH_RE = re.compile(r"(?<!\w)_{1,2}([^_\n]+?)_{1,2}(?!\w)")


# -- small helpers ---------------------------------------------------------


def slugify(text: str, max_len: int = 40) -> str:
    """'Site de la boulangerie !' -> 'site-de-la-boulangerie'."""
    ascii_text = (
        unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    )
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")
    if len(slug) > max_len:
        cut = slug[:max_len]
        slug = cut.rsplit("-", 1)[0] if "-" in cut else cut
    return slug or "tache"


def spoken_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    if minutes < 1:
        return "moins d'une minute"
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes > 1 else ''}"
    hours, rest = divmod(minutes, 60)
    return f"{hours} heure{'s' if hours > 1 else ''}" + (f" {rest}" if rest else "")


def _short_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes} min" if minutes < 60 else f"{minutes // 60} h {minutes % 60:02d}"


def _clean_spoken(text: str) -> str:
    """Strip markdown and whitespace noise so the text can be read aloud."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [label](url)
    text = _MD_RE.sub("", text)
    text = _EMPH_RE.sub(r"\1", text)
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.M)
    return re.sub(r"\s+", " ", text).strip()


def find_claude(explicit: str | None = None) -> str | None:
    for candidate in (explicit, os.environ.get(CLAUDE_PATH_ENV)):
        if candidate:
            return candidate if Path(candidate).exists() else None
    found = shutil.which("claude")
    if found:
        return found
    fallback = Path.home() / ".local" / "bin" / "claude"
    return str(fallback) if fallback.exists() else None


def _ca_bundle(data_dir: Path) -> str | None:
    """A CA bundle trusted by the user's network (it intercepts TLS).

    Inside the sandbox, tools cannot ask the macOS keychain, so curl/pip/npm
    need a PEM file: reuse one already configured, else build one from
    certifi + the system keychains."""
    for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "NODE_EXTRA_CA_CERTS"):
        value = os.environ.get(var)
        if value and Path(value).is_file():
            return value
    bundle = data_dir / "hands" / "ca-bundle.pem"
    if bundle.is_file() and time.time() - bundle.stat().st_mtime < 7 * 86400:
        return str(bundle)
    parts: list[str] = []
    try:
        import certifi

        parts.append(Path(certifi.where()).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - optional
        pass
    if shutil.which("security"):
        for keychain in (
            "/System/Library/Keychains/SystemRootCertificates.keychain",
            "/Library/Keychains/System.keychain",
        ):
            try:
                out = subprocess.run(  # noqa: S603 - fixed system binary
                    ["security", "find-certificate", "-a", "-p", keychain],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                parts.append(out.stdout)
            except (OSError, subprocess.SubprocessError):
                pass
    if not any("BEGIN CERTIFICATE" in p for p in parts):
        return None
    bundle.parent.mkdir(parents=True, exist_ok=True)
    bundle.write_text("\n".join(parts), encoding="utf-8")
    return str(bundle)


# -- task record -----------------------------------------------------------


@dataclass
class Task:
    id: str
    title: str
    goal: str
    folder: str
    status: str = "running"
    started: float = field(default_factory=time.time)
    ended: float | None = None
    summary: str = ""
    cost_usd: float | None = None
    files: list[str] = field(default_factory=list)
    note: str = ""  # what it is doing right now (spoken style)
    session_id: str = ""
    error: str = ""
    # runtime only (not persisted)
    proc: Any = field(default=None, repr=False, compare=False)
    last_text: str = field(default="", repr=False, compare=False)
    tool_uses: int = field(default=0, repr=False, compare=False)
    timer: Any = field(default=None, repr=False, compare=False)

    PERSISTED = (
        "id title goal folder status started ended summary cost_usd files note "
        "session_id error"
    ).split()

    def to_json(self) -> dict[str, Any]:
        data = {k: getattr(self, k) for k in self.PERSISTED}
        data["files"] = list(self.files)
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Task:
        return cls(**{k: data[k] for k in cls.PERSISTED if k in data})

    @property
    def elapsed(self) -> float:
        return (self.ended or time.time()) - self.started

    @property
    def slug(self) -> str:
        return re.sub(r"^\d{4}-\d{2}-\d{2}-", "", Path(self.folder).name)


# -- the hands -------------------------------------------------------------


class Hands:
    def __init__(
        self,
        cfg: JarvisConfig,
        notifier: Notifier,
        claude_path: str | None = None,
        data_dir: Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.hcfg = cfg.hands
        self.notifier = notifier
        self._claude_arg = claude_path
        self.data_dir = Path(data_dir or orb_config.DATA_DIR)
        self.tasks_file = self.data_dir / "tasks.json"
        self.logs_dir = self.data_dir / "tasks"
        self.workspace = Path(self.hcfg.workspace).expanduser()
        self._lock = threading.RLock()
        self._tasks: list[Task] = []
        self._flags: set[str] | None = None
        self._load()

    # -- persistence -------------------------------------------------------

    def _load(self) -> None:
        try:
            raw = json.loads(self.tasks_file.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError):
            log.warning("Unreadable %s, starting fresh", self.tasks_file)
            return
        changed = False
        for item in raw if isinstance(raw, list) else []:
            try:
                task = Task.from_json(item)
            except (TypeError, KeyError):
                continue
            if task.status == "running":  # the app died while it ran
                task.status = "interrupted"
                task.ended = task.ended or time.time()
                changed = True
            self._tasks.append(task)
        if changed:
            self._save()

    def _save(self) -> None:
        with self._lock:
            data = [t.to_json() for t in self._tasks[-KEEP_TASKS:]]
        try:
            self.tasks_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.tasks_file.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            tmp.replace(self.tasks_file)
        except OSError:
            log.exception("Could not save %s", self.tasks_file)

    # -- public API ----------------------------------------------------------

    def tools(self) -> list[Tool]:
        which = {
            "type": "object",
            "properties": {
                "which": {
                    "type": "string",
                    "description": "Quelques mots du titre de la tâche ; vide = "
                    "la tâche en cours ou la plus récente.",
                }
            },
            "required": [],
        }
        return [
            Tool(
                name="delegate_task",
                description=(
                    "Confie un vrai travail sur l'ordinateur à un agent autonome "
                    "(Claude Code) qui tourne en arrière-plan : construire, coder, "
                    "créer un site, une appli ou un script, installer un outil, "
                    "automatiser quelque chose, se connecter à un service ou une "
                    "API, tout travail en plusieurs étapes. À utiliser dès que la "
                    "demande dépasse une action instantanée. Rend la main tout de "
                    "suite ; je préviens l'utilisateur à voix haute quand c'est fini."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "goal": {
                            "type": "string",
                            "description": "La demande complète et détaillée de "
                            "l'utilisateur, avec toutes les précisions données.",
                        },
                        "title": {
                            "type": "string",
                            "description": "Titre court, 2 à 5 mots "
                            "(ex. « site boulangerie »).",
                        },
                    },
                    "required": ["goal", "title"],
                },
                func=self.delegate_task,
                confirm=bool(self.hcfg.confirm),
                progress="Je lance ça.",
            ),
            Tool(
                name="task_status",
                description=(
                    "Où en est une tâche déléguée (en cours ou récente) : ce qu'elle "
                    "fait maintenant, depuis combien de temps, les fichiers créés. "
                    "À utiliser quand l'utilisateur demande « ça avance ? », « où "
                    "en est … ? »."
                ),
                parameters=which,
                func=self.task_status,
            ),
            Tool(
                name="cancel_task",
                description=(
                    "Arrête une tâche déléguée en cours, quand l'utilisateur veut "
                    "l'annuler ou l'arrêter."
                ),
                parameters=which,
                func=self.cancel_task,
                confirm=True,
            ),
            Tool(
                name="task_result",
                description=(
                    "Résultat d'une tâche déléguée terminée : résumé, où se trouve "
                    "le travail et comment l'ouvrir ou le lancer. À utiliser quand "
                    "l'utilisateur demande ce qu'elle a donné ou veut l'ouvrir ; "
                    "propose ensuite de l'ouvrir (open_app / open_url avec le "
                    "chemin donné)."
                ),
                parameters=which,
                func=self.task_result,
            ),
        ]

    def active_summary(self) -> str:
        running = self._running()
        if not running:
            return ""
        n = len(running)
        head = f"{n} tâche{'s' if n > 1 else ''} en cours : "
        if n == 1:
            t = running[0]
            return head + f"{t.title}, {_short_duration(t.elapsed)}"
        return head + ", ".join(
            f"{t.title} ({_short_duration(t.elapsed)})" for t in running
        )

    def close(self) -> None:
        for task in self._running():
            self._stop(task, "interrupted")
        self._save()

    # -- tools -----------------------------------------------------------------

    def delegate_task(self, goal: str, title: str = "") -> str:
        goal = (goal or "").strip()
        if not goal:
            return "Il me faut une description de ce qu'il faut faire."
        title = (title or "").strip() or " ".join(goal.split()[:5])
        claude = find_claude(self._claude_arg)
        if claude is None:
            return (
                "Je ne trouve pas Claude Code sur cet ordinateur, je ne peux pas "
                "lancer ce travail."
            )
        running = self._running()
        if len(running) >= max(1, int(self.hcfg.max_parallel)):
            names = ", ".join(f"« {t.title} »" for t in running)
            return (
                f"J'ai déjà {len(running)} tâche{'s' if len(running) > 1 else ''} "
                f"en cours ({names}). Attends qu'une se termine ou annules-en une."
            )
        folder = self._new_folder(title)
        task = Task(id=uuid.uuid4().hex[:8], title=title, goal=goal, folder=str(folder))
        task.note = "il démarre"
        try:
            self._spawn(task, claude)
        except OSError as exc:
            log.exception("Could not start Claude Code")
            return f"Je n'ai pas pu lancer la tâche : {exc}"
        with self._lock:
            self._tasks.append(task)
        self._save()
        log.info("Task %s started in %s", task.id, folder)
        return (
            "C'est lancé. Je te préviens quand c'est prêt. "
            f"(Tâche « {title} », dans {self._spoken_folder(task)}.)"
        )

    def task_status(self, which: str = "") -> str:
        task = self._find(which, prefer_running=True)
        if task is None:
            return "Aucune tâche déléguée pour le moment."
        if task.status == "running":
            parts = [
                f"« {task.title} » est en cours depuis {spoken_duration(task.elapsed)}."
            ]
            if task.note:
                parts.append(f"En ce moment, {task.note}.")
            if self._files_phrase(task):
                parts.append(f"Fichiers créés : {self._files_phrase(task)}.")
            others = [t for t in self._running() if t is not task]
            if others and not which:
                parts.append(
                    "Aussi en cours : " + ", ".join(t.title for t in others) + "."
                )
            return " ".join(parts)
        status = STATUS_FR.get(task.status, task.status)
        text = f"« {task.title} » est {status}, après {spoken_duration(task.elapsed)}."
        if self._files_phrase(task):
            text += f" Fichiers : {self._files_phrase(task)}."
        if task.status == "done":
            text += " Demande-moi le résultat pour le détail."
        elif task.error:
            text += f" {task.error}"
        return text

    def cancel_task(self, which: str = "") -> str:
        running = self._running()
        if not running:
            return "Aucune tâche n'est en cours."
        if which:
            task = self._find(which, prefer_running=True)
            if task is None or task.status != "running":
                names = ", ".join(t.title for t in running)
                return f"Je ne vois pas cette tâche parmi celles en cours : {names}."
        elif len(running) == 1:
            task = running[0]
        else:
            names = " ou ".join(f"« {t.title} »" for t in running)
            return f"Laquelle dois-je arrêter : {names} ?"
        self._stop(task, "cancelled")
        self._save()
        return (
            f"J'ai arrêté « {task.title} ». Ce qui a déjà été fait reste dans "
            f"{self._spoken_folder(task)}."
        )

    def task_result(self, which: str = "") -> str:
        task = self._find(which, prefer_running=False)
        if task is None:
            return "Aucune tâche déléguée pour le moment."
        if task.status == "running":
            return (
                f"« {task.title} » n'est pas encore finie "
                f"(en cours depuis {spoken_duration(task.elapsed)})."
            )
        folder = Path(task.folder)
        status = STATUS_FR.get(task.status, task.status)
        parts = [f"« {task.title} » ({status})."]
        if task.summary:
            parts.append(task.summary)
        elif task.error:
            parts.append(task.error)
        parts.append(f"Le travail est dans {self._spoken_folder(task)}.")
        if (folder / "README.md").is_file():
            parts.append("Le fichier README explique comment l'utiliser.")
        target = self._open_target(folder)
        if target is not None:
            parts.append(
                "Propose de l'ouvrir. [Pour toi, ne pas lire à voix haute : "
                f"open_url file://{target}  ou dossier : {folder}]"
            )
        else:
            parts.append(f"[Pour toi, ne pas lire à voix haute : dossier {folder}]")
        return " ".join(parts)

    # -- process management ------------------------------------------------

    def _running(self) -> list[Task]:
        with self._lock:
            return [t for t in self._tasks if t.status == "running"]

    def _new_folder(self, title: str) -> Path:
        base = f"{datetime.now():%Y-%m-%d}-{slugify(title)}"
        folder = self.workspace / base
        n = 2
        while folder.exists():
            folder = self.workspace / f"{base}-{n}"
            n += 1
        folder.mkdir(parents=True)
        return folder

    def _supported_flags(self, claude: str) -> set[str]:
        if self._flags is None:
            try:
                out = subprocess.run(  # noqa: S603 - the user's claude binary
                    [claude, "--help"], capture_output=True, text=True, timeout=20
                ).stdout
            except (OSError, subprocess.SubprocessError):
                out = ""
            self._flags = set(re.findall(r"--[a-z][a-z-]+", out))
        return self._flags

    def _settings(self) -> dict[str, Any]:
        return {
            "sandbox": {
                "enabled": True,
                "failIfUnavailable": True,
                "autoAllowBashIfSandboxed": True,
                "allowUnsandboxedCommands": False,
                "network": {
                    "allowedDomains": ALLOWED_DOMAINS,
                    "allowLocalBinding": True,  # dev servers on localhost
                },
                "filesystem": {"denyRead": SECRET_PATHS},
            },
            "permissions": {
                "deny": [f"Read({p}/**)" for p in SECRET_PATHS],
            },
        }

    def _system_prompt(self, task: Task) -> str:
        spoken = self._spoken_folder(task)
        return (
            "Tu travailles pour Jarvis, un assistant vocal : l'utilisateur ne voit "
            "pas ton écran et ne peut pas te répondre pendant que tu travailles.\n"
            "- Termine la tâche en autonomie, sans jamais poser de question : fais "
            "des choix raisonnables et note tes hypothèses dans le README.\n"
            f"- Travaille uniquement dans le dossier courant ({task.folder}) ; ne "
            "modifie rien ailleurs. Installe les dépendances localement (venv, "
            "node_modules dans ce dossier), jamais globalement ni avec sudo.\n"
            "- Les commandes shell tournent dans un bac à sable : écriture limitée "
            "à ce dossier, réseau limité aux registres de paquets et à GitHub. Si "
            "quelque chose est bloqué, trouve un autre moyen (WebFetch pour lire "
            "de la doc, alternative locale) au lieu d'abandonner.\n"
            "- Garde chaque commande shell simple, une par appel (pas de $?, de "
            "$(...) ni d'enchaînements compliqués) : les commandes jugées "
            "complexes sont refusées automatiquement. Si une commande est "
            "refusée, reformule-la plus simplement ou écris un petit script dans "
            "le dossier puis exécute-le.\n"
            "- Vérifie que le résultat fonctionne (lance-le ou teste-le si possible).\n"
            "- Écris un README.md dans le dossier : ce que c'est, comment l'ouvrir "
            "ou le lancer, tes hypothèses.\n"
            "- Ton tout dernier message doit être UNIQUEMENT un résumé en français "
            "de 2 ou 3 phrases, écrit pour être LU À VOIX HAUTE : pas de markdown, "
            "pas de liste, pas de chemin complet ni de code ; pour l'emplacement "
            f"dis « {spoken} ». Dis ce qui a été fait et comment l'utiliser."
        )

    def _child_env(self, folder: Path) -> dict[str, str]:
        env = {
            k: v
            for k, v in os.environ.items()
            if not (k.startswith("CLAUDE") and k != "CLAUDE_CONFIG_DIR")
        }
        cache = folder / ".cache"
        env.setdefault("npm_config_cache", str(cache / "npm"))
        env.setdefault("PIP_CACHE_DIR", str(cache / "pip"))
        env.setdefault("UV_CACHE_DIR", str(cache / "uv"))
        env.setdefault("YARN_CACHE_FOLDER", str(cache / "yarn"))
        bundle = _ca_bundle(self.data_dir)
        if bundle:
            for var in (
                "SSL_CERT_FILE",
                "REQUESTS_CA_BUNDLE",
                "CURL_CA_BUNDLE",
                "PIP_CERT",
                "NODE_EXTRA_CA_CERTS",
                "GIT_SSL_CAINFO",
            ):
                env.setdefault(var, bundle)
        return env

    def _command(self, task: Task, claude: str) -> list[str]:
        flags = self._supported_flags(claude)
        cmd = [
            claude,
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--permission-mode",
            "acceptEdits",
            "--settings",
            json.dumps(self._settings()),
            "--append-system-prompt",
            self._system_prompt(task),
        ]
        optional: list[tuple[str, list[str]]] = [
            ("--permission-prompts", ["none"]),
            ("--setting-sources", ["project"]),
            ("--strict-mcp-config", []),
        ]
        if self.hcfg.model:
            optional.append(("--model", [self.hcfg.model]))
        if self.hcfg.max_budget_usd and self.hcfg.max_budget_usd > 0:
            optional.append(("--max-budget-usd", [f"{self.hcfg.max_budget_usd:g}"]))
        for flag, values in optional:
            if flag in flags:
                cmd += [flag, *values]
            else:
                log.warning("Claude Code has no %s flag, skipping it", flag)
        # Variadic option last; the prompt goes through stdin.
        cmd += ["--allowedTools", ",".join(ALLOWED_TOOLS)]
        return cmd

    def _spawn(self, task: Task, claude: str) -> None:
        folder = Path(task.folder)
        cmd = self._command(task, claude)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        stderr = open(self.logs_dir / f"{task.id}.err", "wb")  # noqa: SIM115
        try:
            proc = subprocess.Popen(  # noqa: S603 - the user's claude binary
                cmd,
                cwd=folder,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr,
                env=self._child_env(folder),
                start_new_session=True,
            )
        finally:
            stderr.close()
        task.proc = proc
        prompt = f"{task.goal}\n\n(Titre de la tâche : {task.title})"
        try:
            assert proc.stdin is not None
            proc.stdin.write(prompt.encode("utf-8"))
            proc.stdin.close()
        except OSError:
            log.exception("Could not send the prompt to Claude Code")
        threading.Thread(
            target=self._reader, args=(task,), name=f"hands-{task.id}", daemon=True
        ).start()
        task.timer = threading.Timer(MAX_TASK_SECONDS, self._timeout, args=(task,))
        task.timer.daemon = True
        task.timer.start()

    def _timeout(self, task: Task) -> None:
        if task.status == "running":
            log.warning("Task %s hit the time limit", task.id)
            task.error = "Elle a dépassé la durée maximale, je l'ai arrêtée."
            self._stop(task, "failed")
            self.notifier.announce(
                f"La tâche « {task.title} » prenait trop de temps, je l'ai arrêtée. "
                f"Ce qui a été fait est dans {self._spoken_folder(task)}."
            )
            self._save()

    def _stop(self, task: Task, status: str) -> None:
        """Terminate the job's whole process group (SIGTERM, then SIGKILL)."""
        with self._lock:
            if task.status != "running":
                return
            task.status = status
            task.ended = time.time()
        proc = task.proc
        if proc is None or proc.poll() is not None:
            return
        try:
            pgid = os.getpgid(proc.pid)
        except OSError:
            return
        for sig, wait in ((signal.SIGTERM, 5.0), (signal.SIGKILL, 2.0)):
            try:
                os.killpg(pgid, sig)
            except OSError:
                return
            try:
                proc.wait(timeout=wait)
                return
            except subprocess.TimeoutExpired:
                continue

    # -- event stream ------------------------------------------------------

    def _reader(self, task: Task) -> None:
        proc = task.proc
        result: dict[str, Any] | None = None
        log_path = self.logs_dir / f"{task.id}.jsonl"
        try:
            with open(log_path, "a", encoding="utf-8") as raw_log:
                assert proc.stdout is not None
                for raw in proc.stdout:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line:
                        continue
                    raw_log.write(line + "\n")
                    raw_log.flush()
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    if event.get("type") == "result":
                        result = event
                    self._on_event(task, event)
        except Exception:  # noqa: BLE001 - never let the thread die silently
            log.exception("Reader for task %s failed", task.id)
        code = proc.wait()
        self._finish(task, result, code)

    def _on_event(self, task: Task, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "system" and event.get("subtype") == "init":
            task.session_id = str(event.get("session_id") or "")
            task.note = "il analyse la demande"
            return
        if kind != "assistant":
            return
        content = (event.get("message") or {}).get("content") or []
        for block in content if isinstance(content, list) else []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and block.get("text", "").strip():
                task.last_text = block["text"].strip()
            elif block.get("type") == "tool_use":
                task.tool_uses += 1
                note = self._describe_tool(task, block)
                if note:
                    task.note = note

    def _describe_tool(self, task: Task, block: dict[str, Any]) -> str:
        name = str(block.get("name") or "")
        args = block.get("input") or {}
        if not isinstance(args, dict):
            args = {}
        if name in EDIT_TOOLS:
            path = str(args.get("file_path") or args.get("notebook_path") or "")
            rel = self._relative(task, path)
            if rel and rel not in task.files:
                task.files.append(rel)
                self._save()
            return f"il écrit {rel or 'un fichier'}"
        if name == "Bash":
            what = str(args.get("description") or "").strip()
            if what:
                return f"il exécute une commande : {what[:80]}"
            return "il exécute une commande"
        if name in ("WebFetch", "WebSearch"):
            return "il consulte de la documentation en ligne"
        if name in ("Read", "Glob", "Grep"):
            return "il examine des fichiers"
        if name == "TodoWrite":
            for todo in args.get("todos") or []:
                if isinstance(todo, dict) and todo.get("status") == "in_progress":
                    step = todo.get("activeForm") or todo.get("content") or ""
                    if step:
                        return f"étape en cours : {str(step)[:80]}"
            return ""
        if name in ("Task", "Agent"):
            return "il a lancé un sous-agent"
        return f"il utilise l'outil {name}" if name else ""

    @staticmethod
    def _relative(task: Task, path: str) -> str:
        if not path:
            return ""
        try:
            return str(Path(path).resolve().relative_to(Path(task.folder).resolve()))
        except (ValueError, OSError):
            return ""  # outside the task folder (and refused by Claude Code)

    def _finish(self, task: Task, result: dict[str, Any] | None, code: int) -> None:
        if task.timer is not None:
            task.timer.cancel()
        cost = result.get("total_cost_usd") if result else None
        if isinstance(cost, (int, float)):
            task.cost_usd = float(cost)
        ok = (
            result is not None
            and not result.get("is_error")
            and result.get("subtype") == "success"
        )
        with self._lock:
            if task.status != "running":  # cancelled / timed out / closed
                self._save()
                return
            task.status = "done" if ok else "failed"
            task.ended = time.time()
            task.note = ""
        where = self._spoken_folder(task)
        if ok:
            assert result is not None
            raw = str(result.get("result") or task.last_text)
            summary = _clean_spoken(self._unpath(task, raw))
            task.summary = summary or f"Le résultat est dans {where}."
            self.notifier.announce(
                f"La tâche « {task.title} » est terminée. {task.summary}"
            )
        else:
            task.error = self._failure_reason(task, result, code)
            self.notifier.announce(
                f"La tâche « {task.title} » n'a pas abouti : {task.error} "
                f"Ce qui a été fait est dans {where}."
            )
        log.info(
            "Task %s %s in %.0fs (cost %s)",
            task.id,
            task.status,
            task.elapsed,
            task.cost_usd,
        )
        self._save()

    def _failure_reason(
        self, task: Task, result: dict[str, Any] | None, code: int
    ) -> str:
        if result is not None:
            subtype = str(result.get("subtype") or "")
            if "budget" in subtype:
                return f"le budget de {self.hcfg.max_budget_usd:g} dollars est atteint."
            if "turns" in subtype:
                return "elle a atteint le nombre maximal d'étapes."
            text = _clean_spoken(str(result.get("result") or ""))
            if text:
                return text[:300]
        err = ""
        try:
            lines = (
                (self.logs_dir / f"{task.id}.err")
                .read_text(encoding="utf-8", errors="replace")
                .strip()
                .splitlines()
            )
            err = lines[-1] if lines else ""
        except OSError:
            pass
        low = err.lower()
        if "log in" in low or "login" in low or "auth" in low:
            return "Claude Code n'est pas connecté à ton compte."
        if "sandbox" in low:
            return "le bac à sable de sécurité n'a pas pu démarrer."
        return f"Claude Code s'est arrêté avec une erreur (code {code})."

    # -- wording helpers ---------------------------------------------------

    def _spoken_folder(self, task: Task) -> str:
        name = self.workspace.name or "Jarvis"
        return f"ton dossier {name}, sous-dossier {task.slug.replace('-', ' ')}"

    def _unpath(self, task: Task, text: str) -> str:
        """Replace spelled-out folder paths by a phrase that reads well."""
        where = self._spoken_folder(task)
        folder = Path(task.folder)
        variants = {str(folder), str(folder.resolve())}
        try:
            variants.add("~/" + str(folder.relative_to(Path.home())))
        except ValueError:
            pass
        for variant in sorted(variants, key=len, reverse=True):
            text = text.replace(variant + "/", where + ", fichier ")
            text = text.replace(variant, where)
        return text

    @staticmethod
    def _files_phrase(task: Task, limit: int = 4) -> str:
        files = [f for f in task.files if (Path(task.folder) / f).exists()]
        names = files[:limit]
        more = len(files) - len(names)
        text = ", ".join(names)
        return text + (f" et {more} autre{'s' if more > 1 else ''}" if more else "")

    @staticmethod
    def _open_target(folder: Path) -> Path | None:
        for name in ("index.html", "dist/index.html", "build/index.html"):
            if (folder / name).is_file():
                return folder / name
        return None

    def _find(self, which: str, prefer_running: bool) -> Task | None:
        with self._lock:
            tasks = list(self._tasks)
        if not tasks:
            return None
        which = (which or "").strip()
        if which:
            words = set(slugify(which, 200).split("-")) - {"la", "le", "les", "de"}
            score: Callable[[Task], int] = lambda t: len(  # noqa: E731
                words & set(slugify(f"{t.title} {t.slug} {t.id}", 200).split("-"))
            )
            best = max(reversed(tasks), key=score)
            if score(best) > 0:
                return best
            return None
        if prefer_running:
            running = [t for t in tasks if t.status == "running"]
            if running:
                return running[-1]
        else:
            finished = [t for t in tasks if t.status != "running"]
            if finished:
                return finished[-1]
        return tasks[-1]
