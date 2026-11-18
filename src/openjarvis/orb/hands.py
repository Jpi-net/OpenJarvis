"""The orb's "hands": long tasks delegated to Claude Code in the background.

The local model can talk, but it cannot reliably build software or wire up a
service. When the user asks for that, the brain calls ``delegate_task``; we
start ``claude -p`` (headless Claude Code) in a fresh folder under the
workspace (``~/Jarvis/<YYYY-MM-DD>-<slug>/``), return at once, follow its
``stream-json`` events in a reader thread and speak up through the
``Notifier`` when the job is done.

Safety model (verified on macOS with Claude Code 2.1.327):

- ``--permission-mode bypassPermissions``: no tool ever waits for an
  approval, so Bash really runs (npm, python, git…). What keeps it safe:
- Built-in OS sandbox (``--settings`` ``sandbox``, macOS Seatbelt) around
  every Bash command: writes confined to the task folder and a per-task temp
  dir, secrets unreadable, outbound network limited to a broad allow-list
  (``ALLOWED_DOMAINS`` + ``hands.extra_domains``). ``failIfUnavailable``
  refuses to start without it, ``allowUnsandboxedCommands: false`` removes
  the escape hatch.
- A PreToolUse hook (``hands_guard.py``) fences the file tools, which the
  sandbox does not cover: Write/Edit outside the task folder (or into its
  ``.claude``) and Read of secrets are denied, as is any Bash call asking to
  leave the sandbox. A SessionStart hook proves the settings were accepted
  (``-p`` mode silently drops invalid settings): no proof, no job.
- ``--setting-sources project``: the user's personal settings (a GUI
  PermissionRequest hook with a one-day timeout, plugins, work MCP servers)
  do not apply. The claude.ai connectors (Notion, Drive, Gmail…) come from
  the account and still load; ``MCP_CONNECTION_NONBLOCKING=false`` makes
  them ready before the first turn. Playwright (browser) is passed with
  ``--mcp-config``.
- Cron / remote triggers are disallowed: nothing outlives the job.

``ask_services`` is the quick path: a blocking ``claude -p`` with only the
connectors and web tools (no Bash, no file writes) that answers in one or
three spoken sentences.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from openjarvis.orb import config as orb_config
from openjarvis.orb import hands_guard as guard
from openjarvis.orb.config import JarvisConfig
from openjarvis.orb.toolkit import Notifier, Tool

log = logging.getLogger("openjarvis.orb.hands")

CLAUDE_PATH_ENV = "JARVIS_CLAUDE_PATH"
MAX_TASK_SECONDS = 60 * 60  # hard wall-clock limit per job
KEEP_TASKS = 50  # entries kept in tasks.json

SERVICES_TIMEOUT_S = 90.0  # ask_services is synchronous (brain waits 120 s)
SERVICES_MODEL = "sonnet"  # fast and reliable with MCP tools
SERVICES_BUDGET_USD = 0.5
GUARD_TOKEN = guard.GUARD_TOKEN

# Hosts sandboxed shell commands may reach (WebFetch and MCP servers are not
# concerned). Claude Code refuses "*" or "*.com" (and then silently ignores
# the whole settings JSON), so the list is broad but explicit; "*.x.y" does
# not match "x.y" itself, hence the pairs.
ALLOWED_DOMAINS = [
    "localhost",
    # code hosting, package registries, toolchains
    "github.com",
    "*.github.com",
    "*.githubusercontent.com",
    "ghcr.io",
    "*.ghcr.io",
    "gitlab.com",
    "*.gitlab.com",
    "bitbucket.org",
    "registry.npmjs.org",
    "*.npmjs.org",
    "*.npmjs.com",
    "registry.yarnpkg.com",
    "*.yarnpkg.com",
    "nodejs.org",
    "*.nodejs.org",
    "bun.sh",
    "deno.land",
    "*.deno.land",
    "jsr.io",
    "*.jsr.io",
    "esm.sh",
    "pypi.org",
    "*.pypi.org",
    "files.pythonhosted.org",
    "*.pythonhosted.org",
    "astral.sh",
    "*.astral.sh",
    "anaconda.org",
    "*.anaconda.org",
    "conda.anaconda.org",
    "crates.io",
    "*.crates.io",
    "static.rust-lang.org",
    "sh.rustup.rs",
    "proxy.golang.org",
    "sum.golang.org",
    "go.dev",
    "rubygems.org",
    "*.rubygems.org",
    "repo.maven.apache.org",
    "repo1.maven.org",
    "plugins.gradle.org",
    "services.gradle.org",
    "packagist.org",
    "*.packagist.org",
    "formulae.brew.sh",
    "registry-1.docker.io",
    "auth.docker.io",
    "*.docker.io",
    "*.docker.com",
    "cdn.playwright.dev",
    "playwright.download.prss.microsoft.com",
    # CDNs, fonts, models
    "cdn.jsdelivr.net",
    "*.jsdelivr.net",
    "unpkg.com",
    "cdnjs.cloudflare.com",
    "fonts.googleapis.com",
    "fonts.gstatic.com",
    "huggingface.co",
    "*.huggingface.co",
    "*.hf.co",
    "ollama.com",
    "*.ollama.com",
    # AI APIs
    "api.anthropic.com",
    "api.openai.com",
    "api.mistral.ai",
    "api.groq.com",
    "openrouter.ai",
    "api.deepseek.com",
    "api.elevenlabs.io",
    "api.replicate.com",
    # Google, Microsoft, Apple
    "googleapis.com",
    "*.googleapis.com",
    "accounts.google.com",
    "oauth2.googleapis.com",
    "www.google.com",
    "graph.microsoft.com",
    "login.microsoftonline.com",
    "login.live.com",
    "*.sharepoint.com",
    "outlook.office.com",
    "outlook.office365.com",
    "api.apple-cloudkit.com",
    # productivity, messaging, dev services
    "api.notion.com",
    "notion.so",
    "*.notion.so",
    "slack.com",
    "*.slack.com",
    "discord.com",
    "*.discord.com",
    "api.telegram.org",
    "api.twilio.com",
    "api.sendgrid.com",
    "api.mailgun.net",
    "api.linear.app",
    "*.atlassian.net",
    "api.atlassian.com",
    "api.trello.com",
    "api.airtable.com",
    "api.figma.com",
    "api.todoist.com",
    "api.dropboxapi.com",
    "content.dropboxapi.com",
    "api.box.com",
    "api.zoom.us",
    "api.calendly.com",
    "api.hubapi.com",
    "api.stripe.com",
    "api.paypal.com",
    "api-m.paypal.com",
    "api.spotify.com",
    "accounts.spotify.com",
    "api.vercel.com",
    "api.netlify.com",
    "*.supabase.co",
    "*.firebaseio.com",
    "firestore.googleapis.com",
    "amazonaws.com",
    "*.amazonaws.com",
    "api.cloudflare.com",
    "*.workers.dev",
    "api.digitalocean.com",
    "api.heroku.com",
    "*.herokuapp.com",
    "*.vercel.app",
    "*.netlify.app",
    "*.github.io",
    # open data
    "*.open-meteo.com",
    "open-meteo.com",
    "api.openweathermap.org",
    "wttr.in",
    "*.openstreetmap.org",
    "nominatim.openstreetmap.org",
    "*.wikipedia.org",
    "*.wikimedia.org",
    "www.wikidata.org",
    "api.coingecko.com",
    "api.frankfurter.app",
    "api.exchangerate.host",
    "irail.be",
    "api.irail.be",
    "data.gov",
    "data.europa.eu",
    "ipinfo.io",
    "ifconfig.me",
    "httpbin.org",
    "example.com",
    "example.org",
]
# Sandboxed tools must not read these either (shared with the guard hook).
SECRET_PATHS = guard.SECRET_PATHS
# Built-in tools a job cannot use: they schedule work that outlives the job.
DISALLOWED_TOOLS = [
    "CronCreate",
    "CronDelete",
    "CronList",
    "RemoteTrigger",
    "ScheduleWakeup",
    "PushNotification",
    "EnterWorktree",
    "ExitWorktree",
]
SERVICE_TOOLS = ["WebSearch", "WebFetch", "ToolSearch"]
EDIT_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit"}
_DOMAIN_RE = re.compile(r"^(\*\.)?[a-z0-9-]+(\.[a-z0-9-]+)+$|^localhost$")

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
    tmpdir: str = field(default="", repr=False, compare=False)
    guard_ok: bool = field(default=False, repr=False, compare=False)

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
                name="ask_services",
                description=(
                    "Consulte ou modifie tout de suite les services en ligne de "
                    "l'utilisateur : agenda (rendez-vous, « qu'est-ce que j'ai "
                    "demain ? », ajouter un événement), mails (« ai-je des mails "
                    "importants ? », chercher, résumer, écrire un brouillon), "
                    "Google Drive (trouver un document), Notion (chercher, créer "
                    "une page). À utiliser pour une question ou une action rapide "
                    "sur ces services ; répond en une à trois phrases dans ce tour "
                    "(jusqu'à une minute). Pas pour construire ou automatiser : "
                    "pour ça, delegate_task."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "request": {
                            "type": "string",
                            "description": "La demande complète de l'utilisateur, "
                            "avec dates, noms et précisions (ex. « ajoute un "
                            "rendez-vous chez le dentiste jeudi à 10 h »).",
                        }
                    },
                    "required": ["request"],
                },
                func=self.ask_services,
                progress="Je m'en occupe…",
            ),
            Tool(
                name="delegate_task",
                description=(
                    "Confie un vrai travail sur l'ordinateur à un agent autonome "
                    "(Claude Code) qui tourne en arrière-plan : construire, coder, "
                    "créer un site, une appli ou un script, installer un outil, "
                    "automatiser quelque chose, brancher un nouveau service ou une "
                    "API, naviguer sur un site pour faire une démarche, tout "
                    "travail long ou en plusieurs étapes (il peut aussi utiliser "
                    "l'agenda, les mails, le Drive et Notion). Pour une simple "
                    "question ou action rapide sur ces services, utilise plutôt "
                    "ask_services. Rend la main tout de suite ; je préviens "
                    "l'utilisateur à voix haute quand c'est fini."
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
                func=self.cancel_task,  # the user asks for it: no confirmation
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
                f"en cours ({names}). Attendez qu'une se termine ou annulez-en une."
            )
        folder = self._new_folder(title)
        task = Task(id=uuid.uuid4().hex[:8], title=title, goal=goal, folder=str(folder))
        task.note = "il démarre"
        try:
            task.tmpdir = tempfile.mkdtemp(prefix=f"jarvis-{task.id}-")
            self._spawn(task, claude)
        except OSError as exc:
            log.exception("Could not start Claude Code")
            if task.tmpdir:
                shutil.rmtree(task.tmpdir, ignore_errors=True)
            return f"Je n'ai pas pu lancer la tâche : {exc}"
        with self._lock:
            self._tasks.append(task)
        self._save()
        log.info("Task %s started in %s", task.id, folder)
        return (
            "C'est lancé. Je vous préviens quand c'est prêt. "
            f"(Tâche « {title} », dans {self._spoken_folder(task)}.)"
        )

    def ask_services(self, request: str) -> str:
        """Blocking ``claude -p`` limited to the connectors + web tools."""
        request = (request or "").strip()
        if not request:
            return "Il me faut la demande à transmettre."
        claude = find_claude(self._claude_arg)
        if claude is None:
            return "Je ne trouve pas Claude Code, je ne peux pas joindre vos services."
        cwd = self.data_dir / "hands" / "services"
        cwd.mkdir(parents=True, exist_ok=True)
        cmd = self._services_command(claude)
        t0 = time.monotonic()
        try:
            proc = subprocess.Popen(  # noqa: S603 - the user's claude binary
                cmd,
                cwd=cwd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self._base_env(),
                start_new_session=True,
            )
        except OSError as exc:
            log.exception("Could not start Claude Code for services")
            return f"Je n'ai pas pu joindre vos services : {exc}"
        try:
            out, err = proc.communicate(
                request.encode("utf-8"), timeout=SERVICES_TIMEOUT_S
            )
        except subprocess.TimeoutExpired:
            self._kill_group(proc)
            log.warning("ask_services timed out after %.0fs", SERVICES_TIMEOUT_S)
            return "Vos services mettent trop de temps à répondre, j'ai abandonné."
        elapsed = time.monotonic() - t0
        result: dict[str, Any] = {}
        try:
            parsed = json.loads(out.decode("utf-8", "replace").strip() or "{}")
            result = parsed if isinstance(parsed, dict) else {}
        except ValueError:
            pass
        text = _clean_spoken(str(result.get("result") or ""))
        log.info(
            "ask_services %.1fs cost %s turns %s: %s",
            elapsed,
            result.get("total_cost_usd"),
            result.get("num_turns"),
            text[:160],
        )
        if result.get("is_error") or result.get("subtype") != "success" or not text:
            subtype = str(result.get("subtype") or "")
            if "budget" in subtype:
                return "La demande coûtait trop cher, je l'ai arrêtée."
            low = err.decode("utf-8", "replace").lower()
            if "log in" in low or "login" in low:
                return "Claude Code n'est pas connecté à votre compte."
            return text or "Je n'ai pas réussi à joindre vos services."
        return text

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
            text += " Demandez-moi le résultat pour le détail."
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

    def _domains(self) -> list[str]:
        """Default allow-list + ``hands.extra_domains`` (invalid ones dropped:
        a single bad entry would make Claude Code ignore all the settings)."""
        domains = list(ALLOWED_DOMAINS)
        for raw in getattr(self.hcfg, "extra_domains", None) or []:
            name = str(raw).strip().lower()
            too_broad = name.startswith("*.") and name.count(".") < 2  # "*.com"
            if not _DOMAIN_RE.match(name) or too_broad:
                log.warning("Ignoring invalid hands.extra_domains entry %r", raw)
                continue
            if name not in domains:
                domains.append(name)
        return domains

    def _guard_command(self, *args: str) -> str:
        script = Path(guard.__file__).resolve()
        return " ".join(
            shlex.quote(a) for a in (sys.executable, "-I", str(script), *args)
        )

    def _settings(self, task: Task) -> dict[str, Any]:
        folder = str(Path(task.folder).resolve())
        roots = [folder] + ([task.tmpdir] if task.tmpdir else [])
        guard_args = [a for root in roots for a in ("--root", root)]
        return {
            "sandbox": {
                "enabled": True,
                "failIfUnavailable": True,
                "autoAllowBashIfSandboxed": True,
                "allowUnsandboxedCommands": False,
                # The network intercepts TLS with a CA that Python >= 3.13
                # rejects from a PEM bundle (strict X.509) and Go ignores:
                # pip/uv (truststore), gh, Go tools need the macOS trust
                # service. Only grants the trustd mach service.
                "enableWeakerNetworkIsolation": True,
                "network": {
                    "allowedDomains": self._domains(),
                    "allowLocalBinding": True,  # dev servers on localhost
                },
                "filesystem": {
                    "allowWrite": roots[1:],
                    "denyWrite": [f"{folder}/.claude"],
                    "denyRead": SECRET_PATHS,
                },
            },
            "permissions": {
                "deny": [f"Read({p}/**)" for p in SECRET_PATHS]
                + [f"Read({p})" for p in SECRET_PATHS],
            },
            "hooks": {
                "SessionStart": [
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": self._guard_command("--session-start"),
                                "timeout": 20,
                            }
                        ]
                    }
                ],
                "PreToolUse": [
                    {
                        "matcher": "Write|Edit|MultiEdit|NotebookEdit|Read|Glob"
                        "|Grep|Bash",
                        "hooks": [
                            {
                                "type": "command",
                                "command": self._guard_command(*guard_args),
                                "timeout": 20,
                            }
                        ],
                    }
                ],
            },
        }

    def _mcp_config(self) -> dict[str, Any]:
        """MCP servers given explicitly (user settings are not loaded): the
        Playwright browser when npx exists. Connectors come from the account."""
        servers: dict[str, Any] = {}
        npx = shutil.which("npx")
        if npx and getattr(self.hcfg, "browser", True):
            path = os.pathsep.join(
                [str(Path(npx).parent), os.environ.get("PATH", "/usr/bin:/bin")]
            )
            servers["playwright"] = {
                "command": npx,
                "args": ["-y", "@playwright/mcp@latest"],
                "env": {"PATH": path},
            }
        return {"mcpServers": servers}

    def _disabled_services(self) -> list[str]:
        """``hands.disabled_services`` ("Microsoft 365", "MaisonRadar"…) as
        tool rules matching every tool of those connectors."""
        rules = []
        for raw in getattr(self.hcfg, "disabled_services", None) or []:
            name = str(raw).strip()
            if not name:
                continue
            if not name.startswith("mcp__"):
                name = "mcp__claude_ai_" + re.sub(
                    r"[^A-Za-z0-9_-]", "_", re.sub(r"^claude\.ai\s+", "", name)
                )
            rules.append(name)
        return rules

    def _system_prompt(self, task: Task) -> str:
        spoken = self._spoken_folder(task)
        return (
            "Tu travailles pour Jarvis, un assistant vocal : l'utilisateur ne voit "
            "pas ton écran et ne peut pas te répondre pendant que tu travailles.\n"
            "- Termine la tâche en autonomie, sans jamais poser de question : fais "
            "des choix raisonnables et note tes hypothèses dans le README.\n"
            f"- Travaille dans le dossier courant ({task.folder}) ; ne modifie "
            "rien ailleurs (fichiers temporaires : $TMPDIR). Installe les "
            "dépendances localement (venv, node_modules dans ce dossier), jamais "
            "globalement ni avec sudo.\n"
            "- Tu peux lancer librement des commandes shell (npm, python, git, "
            "curl…). Elles tournent dans un bac à sable : écriture limitée à ce "
            "dossier, secrets illisibles, réseau limité à une large liste "
            "(registres de paquets, GitHub, grandes API). Si un domaine est "
            "bloqué, lis-le avec WebFetch ou trouve une autre voie au lieu "
            "d'abandonner. Ne tente pas de sortir du bac à sable.\n"
            "- Le réseau intercepte le TLS avec une autorité que Python 3.13+ "
            "refuse depuis un fichier PEM : pip fonctionne, mais dans un script "
            "Python qui fait du HTTPS, installe truststore et appelle "
            "truststore.inject_into_ssl() au début (ne désactive jamais la "
            "vérification des certificats).\n"
            "- Services de l'utilisateur : tes outils MCP donnent accès à ses "
            "connecteurs (agenda, mails, Google Drive, Notion…) et à un "
            "navigateur (Playwright) ; certains outils se chargent via "
            "ToolSearch. Lire et chercher est libre. Envoyer un message à "
            "quelqu'un, supprimer des données ou payer : seulement si la demande "
            "le dit explicitement.\n"
            "- Si un service n'est pas connecté, trouve un moyen : son API "
            "(script dans ce dossier, clé dans un fichier .env local), un serveur "
            "MCP ou une bibliothèque installés ici. Si cela exige que "
            "l'utilisateur se connecte lui-même (OAuth, clé d'API, mot de passe), "
            "prépare tout puis dis-le clairement dans ton résumé, par exemple "
            "« il faut que vous connectiez Microsoft 365 une fois dans Claude ».\n"
            "- Vérifie que le résultat fonctionne (lance-le ou teste-le si possible).\n"
            "- Écris un README.md dans le dossier : ce que c'est, comment l'ouvrir "
            "ou le lancer, tes hypothèses.\n"
            "- Ton tout dernier message doit être UNIQUEMENT un résumé en français "
            "de 2 ou 3 phrases, écrit pour être LU À VOIX HAUTE en vouvoyant "
            "l'utilisateur : pas de markdown, "
            "pas de liste, pas de chemin complet ni de code ; pour l'emplacement "
            f"dis « {spoken} ». Dis ce qui a été fait et comment l'utiliser."
        )

    def _services_prompt(self) -> str:
        now = datetime.now().astimezone()
        return (
            "Tu es la partie « services » de Jarvis, un assistant vocal. Tu as "
            "accès aux services connectés de l'utilisateur (agenda, mails, Google "
            "Drive, Notion…) par tes outils MCP (certains se chargent via "
            "ToolSearch), et au web.\n"
            f"- Maintenant : {now:%A %d %B %Y, %H:%M} (fuseau {now:%Z}, UTC{now:%z}).\n"
            "- Réponds vite : le moins d'appels d'outils possible, pas de question "
            "en retour, aucune explication de ta démarche.\n"
            "- Lire, chercher, résumer : libre. Créer ou modifier un élément de "
            "l'utilisateur (rendez-vous, page, brouillon) : fais-le quand la "
            "demande le demande. Actions irréversibles ou vers d'autres personnes "
            "(envoyer un mail ou un message, inviter quelqu'un, supprimer, payer, "
            "publier) : seulement si la demande le dit explicitement ; sinon "
            "prépare un brouillon et dis-le.\n"
            "- Si le service nécessaire n'est pas connecté ou demande une "
            "connexion, dis lequel et qu'il faut le connecter une fois dans "
            "Claude.\n"
            "- N'invente rien : si tu ne trouves pas, dis-le.\n"
            "- Ta réponse finale : une à trois phrases en français, écrites pour "
            "être LUES À VOIX HAUTE, en vouvoyant l'utilisateur : pas de "
            "markdown, pas de liste, pas d'URL, pas d'identifiant ni d'adresse "
            "technique ; heures et dates dites naturellement (« jeudi à dix heures »)."
        )

    def _base_env(self) -> dict[str, str]:
        env = {
            k: v
            for k, v in os.environ.items()
            if not (k.startswith("CLAUDE") and k != "CLAUDE_CONFIG_DIR")
        }
        # Connect the claude.ai connectors before the first turn (in -p mode
        # they load in the background otherwise and may be missing).
        env["MCP_CONNECTION_NONBLOCKING"] = "false"
        bundle = _ca_bundle(self.data_dir)
        if bundle:
            for var in (
                "SSL_CERT_FILE",
                "REQUESTS_CA_BUNDLE",
                "CURL_CA_BUNDLE",
                "PIP_CERT",
                "NODE_EXTRA_CA_CERTS",
                "GIT_SSL_CAINFO",
                "AWS_CA_BUNDLE",
                "CARGO_HTTP_CAINFO",
                "DENO_CERT",
                "HTTPLIB2_CA_CERTS",
            ):
                env.setdefault(var, bundle)
        return env

    def _child_env(self, folder: Path, tmpdir: str = "") -> dict[str, str]:
        env = self._base_env()
        cache = folder / ".cache"
        env.setdefault("npm_config_cache", str(cache / "npm"))
        env.setdefault("PIP_CACHE_DIR", str(cache / "pip"))
        env.setdefault("UV_CACHE_DIR", str(cache / "uv"))
        env.setdefault("YARN_CACHE_FOLDER", str(cache / "yarn"))
        if tmpdir:
            env["TMPDIR"] = tmpdir
            # ~/.npmrc is unreadable in the sandbox (tokens, work registry):
            # give npm a clean public config instead.
            npmrc = Path(tmpdir) / "npmrc"
            lines = ["registry=https://registry.npmjs.org/"]
            if env.get("NODE_EXTRA_CA_CERTS"):
                lines.append(f"cafile={env['NODE_EXTRA_CA_CERTS']}")
            npmrc.write_text("\n".join(lines) + "\n", encoding="utf-8")
            env["NPM_CONFIG_USERCONFIG"] = str(npmrc)
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
            "bypassPermissions",
            "--settings",
            json.dumps(self._settings(task)),
            "--append-system-prompt",
            self._system_prompt(task),
        ]
        optional: list[tuple[str, list[str]]] = [
            ("--permission-prompts", ["none"]),
            ("--setting-sources", ["project"]),
            ("--include-hook-events", []),
        ]
        mcp = self._mcp_config()
        if mcp["mcpServers"]:
            optional.append(("--mcp-config", [json.dumps(mcp)]))
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
        cmd += [
            "--disallowedTools",
            ",".join(DISALLOWED_TOOLS + self._disabled_services()),
        ]
        return cmd

    def _services_command(self, claude: str) -> list[str]:
        flags = self._supported_flags(claude)
        model = str(getattr(self.hcfg, "services_model", "") or SERVICES_MODEL)
        budget = getattr(self.hcfg, "services_budget_usd", SERVICES_BUDGET_USD)
        cmd = [
            claude,
            "-p",
            "--output-format",
            "json",
            # No Bash and no file tools exist in this session (--tools), so
            # bypassing prompts only lets the connectors and web tools run.
            "--permission-mode",
            "bypassPermissions",
            "--append-system-prompt",
            self._services_prompt(),
        ]
        optional: list[tuple[str, list[str]]] = [
            ("--permission-prompts", ["none"]),
            ("--setting-sources", ["project"]),
            ("--no-session-persistence", []),
            ("--model", [model]),
        ]
        if budget and float(budget) > 0:
            optional.append(("--max-budget-usd", [f"{float(budget):g}"]))
        for flag, values in optional:
            if flag in flags:
                cmd += [flag, *values]
        cmd += ["--tools", ",".join(SERVICE_TOOLS)]
        disabled = self._disabled_services()
        if disabled:
            cmd += ["--disallowedTools", ",".join(disabled)]
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
                env=self._child_env(folder, task.tmpdir),
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
        self._kill_group(task.proc)

    @staticmethod
    def _kill_group(proc: Any) -> None:
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
        if kind == "system" and event.get("subtype") == "hook_response":
            if GUARD_TOKEN in str(event.get("stdout") or event.get("output") or ""):
                task.guard_ok = True
            return
        if kind == "system" and event.get("subtype") == "init":
            task.session_id = str(event.get("session_id") or "")
            task.note = "il analyse la demande"
            if not task.guard_ok:  # our --settings were dropped: no sandbox
                log.error("Task %s: safety settings not applied, stopping", task.id)
                task.error = (
                    "le bac à sable de sécurité n'a pas pu être activé, j'ai "
                    "tout arrêté avant qu'elle ne commence."
                )
                self._stop(task, "failed")
                self.notifier.announce(
                    f"La tâche « {task.title} » n'a pas démarré : {task.error}"
                )
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
        if name.startswith("mcp__"):
            server = name.split("__")[1] if name.count("__") >= 2 else name
            server = re.sub(r"^(claude_ai|plugin)_", "", server)
            server = server.split("_")[0] if server.startswith("playwright") else server
            return f"il utilise {server.replace('_', ' ')}"
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
        if task.tmpdir:
            shutil.rmtree(task.tmpdir, ignore_errors=True)
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
            return "Claude Code n'est pas connecté à votre compte."
        if "sandbox" in low:
            return "le bac à sable de sécurité n'a pas pu démarrer."
        return f"Claude Code s'est arrêté avec une erreur (code {code})."

    # -- wording helpers ---------------------------------------------------

    def _spoken_folder(self, task: Task) -> str:
        name = self.workspace.name or "Jarvis"
        return f"votre dossier {name}, sous-dossier {task.slug.replace('-', ' ')}"

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
