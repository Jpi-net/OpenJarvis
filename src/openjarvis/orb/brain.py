"""The orb's brain: persona, self-state, persistent memory, streaming replies.

Talks to Ollama's native ``/api/chat`` with streaming and reasoning disabled.
The prompt is laid out for Ollama's prefix cache:

1. a stable system message (persona + remembered facts + digest of past
   sessions) that only changes when a new fact is learned,
2. the conversation history,
3. a short system message with the live self-state (time, uptime, load),
4. the new user message.

Persistent state lives in ``config.DATA_DIR``:
- ``journal.jsonl``: one event per line (session_start, exchange, session_end)
- ``facts.json``: short things the user asked to remember
- ``self.json``: birth date, session counter, last conversation time

Tool calling: when a ``ToolBox`` is given, its schemas are sent with every
request (they render into the system prefix, so the cache stays valid). The
model either streams text (yielded at once) or ``tool_calls``; calls run in
parallel, their results are appended as ``tool`` messages and the follow-up
reply is streamed, for up to ``MAX_TOOL_ROUNDS`` rounds. Tools marked
``confirm=True`` are not run: the brain asks out loud and runs them only after
a spoken yes on the next turn (rule-based, no model call).
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import platform
import re
import subprocess
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait as wait_futures
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator
from zoneinfo import ZoneInfo

import httpx

from openjarvis.orb import config as orb_config
from openjarvis.orb.config import JarvisConfig
from openjarvis.orb.toolkit import Tool, ToolBox

log = logging.getLogger("openjarvis.orb.brain")

TZ = ZoneInfo("Europe/Brussels")
MAX_FACTS = 30
MAX_FACT_CHARS = 160
NUM_CTX = 8192
NUM_PREDICT = 400
GREETING_BUDGET_S = 1.9
MAX_TOOL_ROUNDS = 4
PROGRESS_AFTER_S = 0.6  # speak the tool's progress line if it runs longer
TOOL_TIMEOUT_S = 120.0
PENDING_TTL_S = 60.0
ACTION_NUDGE = (
    "Tu n'as appelé aucun outil : rien ne s'est passé. Si un outil correspond à "
    "la demande, appelle-le maintenant, sans texte. Sinon, réponds normalement."
)
_ACTION_RE = re.compile(
    r"\b(?:ouvre|lance|d[ée]marre|ferme|quitte|mets?|joue|monte|baisse|coupe"
    r"|r[èe]gle|minuteur|chrono|timer|rappelle[- ]moi|rappel|programme|note|prends"
    r"|[ée]cris|copie|cherche|recherche|regarde|v[ée]rifie|m[ée]t[ée]o|temps qu'il"
    r"|pleut|pleuvoir|temp[ée]rature|musique|chanson|morceau|pause|suivant"
    r"|pr[ée]c[ée]dent|volume|batterie|annule|liste|qui a gagn[ée]|actualit[ée]"
    r"|derni[èe]res? nouvelles|construis|cr[ée]e|code|installe|connecte|automatise)",
    re.I,
)


def _wants_action(text: str) -> bool:
    return bool(_ACTION_RE.search(text))


TOOL_RULE = (
    "Règle absolue pour les outils : si la demande correspond à un outil, ta "
    "réponse est UNIQUEMENT l'appel de l'outil, sans aucun texte avant. Tu ne "
    "parles qu'après avoir reçu le résultat. Dire « je vérifie », « j'ouvre » ou "
    "« c'est fait » sans appeler l'outil est une faute grave : rien ne se passe."
)

ERR_UNREACHABLE = "Je n'arrive pas à joindre mon moteur de réflexion."
ERR_NO_MODEL = "Mon modèle de langage n'est pas installé sur cette machine."
ERR_GENERIC = "Désolé, quelque chose s'est mal passé dans ma réflexion."

PERSONA_PATH = Path(__file__).with_name("persona.md")
_PERSONA_FALLBACK = (
    "Tu es {name}, l'assistant vocal de {user_ref}, en local sur son Mac. "
    "Réponds en {language_name}, en une à trois phrases parlées, sans markdown.\n"
    "{address_rule}"
)


def _load_persona() -> str:
    try:
        return PERSONA_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        log.warning("persona.md missing, using the minimal persona")
        return _PERSONA_FALLBACK


_LANG_NAMES = {
    "fr": "français",
    "en": "anglais",
    "nl": "néerlandais",
    "de": "allemand",
    "es": "espagnol",
    "it": "italien",
}
_DAYS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
_MONTHS = [
    "janvier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
]

# ---------------------------------------------------------------- utilities


def _now() -> datetime:
    return datetime.now(TZ)


def _parse_ts(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value)).astimezone(TZ)
    except (TypeError, ValueError):
        return None


def _say_datetime(dt: datetime) -> str:
    day = f"{_DAYS[dt.weekday()]} {dt.day} {_MONTHS[dt.month - 1]} {dt.year}"
    return f"{day}, {dt.hour} h {dt.minute:02d}"


def _say_duration(delta: timedelta) -> str:
    s = max(0, int(delta.total_seconds()))
    if s < 90:
        return "moins de deux minutes"
    m = s // 60
    if m < 60:
        return f"{m} minutes"
    h = m // 60
    if h < 24:
        rest = m % 60
        unit = "heure" if h == 1 else "heures"
        return f"{h} {unit}" + (f" {rest}" if rest and h < 3 else "")
    d = h // 24
    if d < 60:
        return "1 jour" if d == 1 else f"{d} jours"
    mo = d // 30
    return f"{mo} mois"


def _part_of_day(dt: datetime) -> str:
    h = dt.hour
    if h < 5:
        return "nuit"
    if h < 12:
        return "matin"
    if h < 18:
        return "après-midi"
    if h < 22:
        return "soirée"
    return "nuit"


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write_json(path: Path, data: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        log.warning("Could not write %s", path, exc_info=True)


_EMOJI_RE = re.compile(
    "[\U0001f000-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff\U0000fe0f\U0000200d\U00002b00-\U00002bff]"
)
_MD_RE = re.compile(r"[*#`_~>|]")
_LIST_RE = re.compile(r"(^|\n)\s*(?:[-•]|\d+[.)])\s+")


def _clean_spoken(text: str) -> str:
    """Remove markdown / emoji artefacts from a streamed delta."""
    text = _EMOJI_RE.sub("", text)
    text = _LIST_RE.sub(r"\1", text)
    return _MD_RE.sub("", text)


class _BracketFilter:
    """Drop echoed ``[État …]`` stamps (small models sometimes copy them).

    Text after an unmatched ``[`` is held back until the bracket closes."""

    MAX_HOLD = 300

    def __init__(self) -> None:
        self.buf = ""
        self.removed = False

    def feed(self, text: str) -> str:
        self.buf += text
        out = _STAMP_RE.sub(self._drop, self.buf)
        idx = out.rfind("[")
        if idx >= 0 and "]" not in out[idx:] and len(out) - idx < self.MAX_HOLD:
            self.buf = out[idx:]
            return out[:idx]
        self.buf = ""
        return out

    def _drop(self, _m: re.Match[str]) -> str:
        self.removed = True
        return ""

    def flush(self) -> str:
        rest, self.buf = self.buf, ""
        if rest.lstrip().startswith("[État"):
            self.removed = True
            return ""
        return rest


_STAMP_RE = re.compile(r"\[\s*[ÉE]tat\b[^\]]*\]\s*")


class _ThinkFilter:
    """Streaming filter dropping ``<think>…</think>`` spans split across deltas."""

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self.buf = ""
        self.inside = False

    def feed(self, text: str) -> str:
        self.buf += text
        out: list[str] = []
        while self.buf:
            tag = self.CLOSE if self.inside else self.OPEN
            idx = self.buf.find(tag)
            if idx >= 0:
                if not self.inside:
                    out.append(self.buf[:idx])
                self.buf = self.buf[idx + len(tag) :]
                self.inside = not self.inside
                continue
            # keep a possible partial tag at the end
            keep = 0
            for k in range(min(len(tag) - 1, len(self.buf)), 0, -1):
                if tag.startswith(self.buf[-k:]):
                    keep = k
                    break
            if not self.inside:
                out.append(self.buf[: len(self.buf) - keep])
            self.buf = self.buf[len(self.buf) - keep :] if keep else ""
            break
        return "".join(out)

    def flush(self) -> str:
        rest, self.buf = ("" if self.inside else self.buf), ""
        return rest


# ------------------------------------------------------- yes / no detection


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower().replace("’", "'"))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^\w' -]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_YES = (
    "oui",
    "ouais",
    "ouep",
    "yes",
    "yep",
    "ok",
    "okay",
    "d'accord",
    "dac",
    "vas-y",
    "vas y",
    "allez",
    "allez-y",
    "allez y",
    "go",
    "confirme",
    "je confirme",
    "confirmé",
    "c'est bon",
    "c'est parti",
    "parfait",
    "exactement",
    "bien sur",
    "carrement",
    "absolument",
    "tout a fait",
    "fais-le",
    "fais le",
    "fais-la",
    "lance",
    "lance-le",
    "lance le",
    "envoie",
    "volontiers",
    "avec plaisir",
    "affirmatif",
    "evidemment",
    "certainement",
    "bien entendu",
    "oui oui",
)
_NO = (
    "non",
    "nan",
    "no",
    "annule",
    "annuler",
    "laisse tomber",
    "laisse",
    "stop",
    "pas maintenant",
    "surtout pas",
    "oublie",
    "oublie ca",
    "n'en fais rien",
    "ne fais rien",
    "ne le fais pas",
    "attends",
    "plus tard",
    "negatif",
    "pas question",
    "certainement pas",
    "non merci",
    "pas du tout",
)
_FILLER = {
    "euh",
    "heu",
    "bah",
    "ben",
    "bon",
    "alors",
    "eh",
    "hum",
    "jarvis",
    "merci",
    "stp",
    "svp",
    "s'il",
    "te",
    "vous",
    "plait",
    "plaît",
    "oui",
    "non",
    "vas-y",
    "go",
    "c'est",
    "bon",
    "ok",
    "d'accord",
    "allez",
    "volontiers",
    "parfait",
    "fais-le",
    "pas",
    "ca",
    "ça",
    "maintenant",
    "tout",
    "a",
    "fait",
    "laisse",
    "tomber",
    "annule",
}


def _match_lead(text: str, phrases: tuple[str, ...]) -> str | None:
    """Longest phrase the text starts with (on a word boundary)."""
    best = None
    for p in phrases:
        p = _fold(p)
        if (text == p or text.startswith(p + " ")) and (
            best is None or len(p) > len(best)
        ):
            best = p
    return best


def classify_confirmation(text: str) -> str | None:
    """'yes' / 'no' for a short spoken (French) answer, else None.

    Only short answers count: "oui mais plutôt demain" is a new request."""
    t = _fold(text)
    words = t.split()
    while words and words[0] in ("euh", "heu", "bah", "ben", "alors", "hum", "jarvis"):
        words.pop(0)
    t = " ".join(words)
    if not t:
        return None
    for verdict, phrases in (("no", _NO), ("yes", _YES)):
        lead = _match_lead(t, phrases)
        if lead is None:
            continue
        rest = t[len(lead) :].split()
        allowed = _FILLER | {w for p in phrases for w in _fold(p).split()}
        if all(w in allowed for w in rest):
            return verdict
        return None
    return None


@dataclass
class _Pending:
    calls: list[dict[str, Any]]
    expires: float  # time.monotonic() deadline


def _call_name(call: dict[str, Any]) -> str:
    return str(call.get("function", {}).get("name", ""))


def _call_args(call: dict[str, Any]) -> dict[str, Any]:
    args = call.get("function", {}).get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = {}
    return args if isinstance(args, dict) else {}


def _describe_call(call: dict[str, Any]) -> str:
    """Fallback spoken description of a pending action."""
    name, args = _call_name(call), _call_args(call)
    detail = next((str(v) for v in args.values() if isinstance(v, str) and v), "")
    if name == "quit_app":
        return f"Je ferme {detail or 'cette application'}"
    if detail:
        return f"Je lance « {detail[:120]} »"
    return f"Je lance l'action {name}"


def _run_tool(tool: Tool, args: dict[str, Any]) -> str:
    """Run a tool, dropping arguments its function does not accept."""
    try:
        params = inspect.signature(tool.func).parameters
        if not any(p.kind is p.VAR_KEYWORD for p in params.values()):
            args = {k: v for k, v in args.items() if k in params}
    except (TypeError, ValueError):
        pass
    return tool.run(**args)


# --------------------------------------------------------------- self-state


class _MachineProbe:
    """Cheap, cached view of the machine load."""

    def __init__(self) -> None:
        self._cache: tuple[float, str] = (0.0, "")
        self._lock = threading.Lock()
        self.chip = self._chip()
        try:
            import psutil

            self._psutil: Any = psutil
            psutil.cpu_percent(interval=None)  # prime the counter
        except ImportError:
            self._psutil = None

    @staticmethod
    def _chip() -> str:
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=1,
            ).stdout.split("\n")
            ram = int(out[1]) // (1024**3)
            return f"{out[0].strip()}, {ram} Go de mémoire"
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            return platform.machine()

    def load(self) -> str:
        with self._lock:
            ts, text = self._cache
            if text and time.monotonic() - ts < 5:
                return text
            parts = []
            if self._psutil is not None:
                try:
                    parts.append(
                        f"processeur {self._psutil.cpu_percent(interval=None):.0f} %"
                    )
                    vm = self._psutil.virtual_memory()
                    parts.append(f"mémoire utilisée {vm.percent:.0f} %")
                except Exception:  # noqa: BLE001
                    pass
            if not parts:
                try:
                    parts.append(f"charge moyenne {os.getloadavg()[0]:.1f}")
                except OSError:
                    pass
            text = ", ".join(parts) or "inconnue"
            self._cache = (time.monotonic(), text)
            return text


# ------------------------------------------------------------------ memory

_FACT_PATTERNS = [
    re.compile(
        r"\b(?:souviens[- ]toi|rappelle[- ]toi|retiens|n'oublie pas|note"
        r"|garde en t[eê]te)"
        r"(?: bien)?(?: que| qu'|,|:)\s*(.+)",
        re.I,
    ),
    re.compile(r"\b(?:je m'appelle|mon (?:pr[ée]nom|nom) (?:est|c'est))\s+(.+)", re.I),
    re.compile(
        r"\b((?:j'aime|j'adore|je d[ée]teste|je n'aime pas|je pr[ée]f[èe]re"
        r"|mon .{2,30} pr[ée]f[ée]r[ée]e? "
        r"(?:est|c'est)|j'habite|je vis [àa]|je travaille|je suis allergique)\b.+)",
        re.I,
    ),
]
_NAME_RE = re.compile(
    r"\b(?:je m'appelle|mon (?:pr[ée]nom|nom) (?:est|c'est))\s+([A-ZÀ-Ÿa-zà-ÿ'-]+)",
    re.I,
)
_FORGET_RE = re.compile(r"\boublie (?:que |qu')(.+)", re.I)


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


class _Facts:
    def __init__(self, path: Path) -> None:
        self.path = path
        data = _read_json(path, {})
        self.name: str = data.get("user_name", "") if isinstance(data, dict) else ""
        items = data.get("facts", []) if isinstance(data, dict) else []
        self.items: list[dict[str, str]] = [
            f for f in items if isinstance(f, dict) and f.get("text")
        ]

    def save(self) -> None:
        _write_json(
            self.path, {"user_name": self.name, "facts": self.items[-MAX_FACTS:]}
        )

    def learn(self, user_text: str) -> bool:
        """Rule-based pass over the user's words; returns True if memory changed."""
        text = user_text.strip()
        changed = False
        m = _FORGET_RE.search(text)
        if m:
            key = set(_norm(m.group(1)).split())
            before = len(self.items)
            self.items = [
                f
                for f in self.items
                if len(key & set(_norm(f["text"]).split())) < max(2, len(key) // 2)
            ]
            return len(self.items) != before
        m = _NAME_RE.search(text)
        if m:
            name = m.group(1).strip(" .,!?").capitalize()
            if name and name.lower() not in {"comment", "pas"} and name != self.name:
                self.name, changed = name, True
        for pat in _FACT_PATTERNS:
            m = pat.search(text)
            if not m:
                continue
            fact = m.group(1).strip(" .!?,;:")
            if len(fact) < 4:
                continue
            fact = fact[:MAX_FACT_CHARS]
            n = _norm(fact)
            self.items = [f for f in self.items if _norm(f["text"]) != n]
            self.items.append({"text": fact, "date": _now().date().isoformat()})
            self.items = self.items[-MAX_FACTS:]
            changed = True
            break
        if changed:
            self.save()
        return changed

    def render(self) -> str:
        if not self.items and not self.name:
            return ""
        lines = [
            "Faits retenus sur l'utilisateur (ses propres mots, à la première "
            "personne) :"
        ]
        if self.name:
            lines.append(f"- il s'appelle {self.name}")
        lines += [f"- « {f['text']} »" for f in self.items]
        return "\n".join(lines)


class _Journal:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def append(self, event: dict[str, Any]) -> None:
        event = {"ts": _now().isoformat(timespec="seconds"), **event}
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(event, ensure_ascii=False) + "\n")
            except OSError:
                log.warning("Could not write journal", exc_info=True)

    def tail(self, max_bytes: int = 200_000) -> list[dict[str, Any]]:
        try:
            with self.path.open("rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                fh.seek(max(0, size - max_bytes))
                raw = fh.read().decode("utf-8", errors="ignore")
        except OSError:
            return []
        lines = raw.splitlines()
        if len(raw) >= max_bytes and lines:
            lines = lines[1:]  # first line is probably truncated
        events = []
        for line in lines:
            try:
                ev = json.loads(line)
                if isinstance(ev, dict):
                    events.append(ev)
            except ValueError:
                continue
        return events


# ------------------------------------------------------------------- brain


class Brain:
    """Persona + memory + streaming replies from Ollama.

    Prompt-cache note: qwen3.5 is a hybrid (recurrent + attention) model, so
    Ollama can only reuse its cache when the new prompt *extends* the previous
    one exactly. The session therefore keeps an append-only transcript: a
    frozen system prompt, then turns whose user message carries a one-line
    live state stamp, then the raw assistant output. The system prompt is
    only rebuilt when the history is trimmed (the cache breaks anyway).
    A turn is the list of messages it produced (user, assistant tool calls,
    tool results, final assistant text), stored verbatim.
    """

    def __init__(
        self,
        cfg: JarvisConfig,
        toolbox: ToolBox | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
    ):
        self.cfg = cfg
        self.bcfg = cfg.brain
        self.toolbox = toolbox
        data_dir = Path(orb_config.DATA_DIR)
        data_dir.mkdir(parents=True, exist_ok=True)
        self._client = httpx.Client(
            base_url=self.bcfg.host.rstrip("/"),
            timeout=httpx.Timeout(60.0, connect=3.0),
            transport=transport,
        )
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="orb-tool")
        self._pending: _Pending | None = None
        self._machine = _MachineProbe()
        self._facts = _Facts(data_dir / "facts.json")
        self._journal = _Journal(data_dir / "journal.jsonl")
        self._self_path = data_dir / "self.json"
        # one entry per turn: the messages exactly as sent to / produced by
        # the model (append-only, so Ollama's prefix cache stays valid)
        self._history: list[list[dict[str, Any]]] = []
        self._session_exchanges: list[tuple[str, str]] = []
        self._lock = threading.Lock()
        self._greeting_msg: str | None = None

        self.started = _now()
        state = _read_json(self._self_path, {})
        if not isinstance(state, dict):
            state = {}
        self.first_ever = not state.get("born")
        self.born = _parse_ts(state.get("born")) or self.started
        self.prev_last_seen = _parse_ts(
            state.get("last_exchange") or state.get("last_seen")
        )
        self.session_number = int(state.get("sessions", 0)) + 1
        self._state: dict[str, Any] = {
            "born": self.born.isoformat(timespec="seconds"),
            "sessions": self.session_number,
            "last_seen": self.started.isoformat(timespec="seconds"),
        }
        if state.get("last_exchange"):
            self._state["last_exchange"] = state["last_exchange"]
        _write_json(self._self_path, self._state)

        self._digest = self._build_digest(self._journal.tail())
        self._journal.append(
            {
                "type": "session_start",
                "session": self.session_number,
                "model": self.bcfg.model,
            }
        )
        self._static_prompt = self._build_static_prompt()

    # ---------------------------------------------------------- prompt parts

    def _user_name(self) -> str:
        return self.cfg.assistant.user_name or self._facts.name

    def _build_static_prompt(self) -> str:
        user = self._user_name()
        address = (
            f"- Vouvoie l'utilisateur et appelle-le {user} de temps en temps, "
            "sans en abuser.\n"
            if user
            else "- Vouvoie l'utilisateur.\n"
        )
        lang = self.cfg.assistant.language
        prompt = _load_persona().format(
            name=self.cfg.assistant.name,
            user_ref=user or "ton utilisateur",
            language_name=_LANG_NAMES.get(lang, lang),
            address_rule=address,
        )
        parts = [prompt, self._identity_block()]
        facts = self._facts.render()
        if facts:
            parts.append(facts)
        if self._digest:
            parts.append(self._digest)
        if self.toolbox:
            # Last in the prompt on purpose: small models (qwen3.5 9b, no
            # thinking) otherwise often *say* "j'ouvre…" without calling.
            parts.append(TOOL_RULE)
        return "\n\n".join(parts)

    def _identity_block(self) -> str:
        """Session-stable part of the self-state (lives in the system prompt)."""
        n = self.session_number
        if self.prev_last_seen:
            last = (
                f"{_say_datetime(self.prev_last_seen)}, il y a "
                f"{_say_duration(self.started - self.prev_last_seen)}"
            )
        else:
            last = "aucune, cette session est ta toute première"
        return (
            "Ton état interne (à utiliser quand c'est pertinent, sans le réciter) :\n"
            f"- Tu existes depuis le {_say_datetime(self.born)} ; ceci est ta "
            f"session numéro {n}, démarrée le {_say_datetime(self.started)}.\n"
            f"- Dernière conversation avant cette session : {last}.\n"
            f"- Tu tournes sur le modèle de langage {self.bcfg.model}, servi par "
            f"Ollama, en local sur un Mac ({self._machine.chip}, macOS). "
            "Ta réflexion reste locale ; seuls tes outils web (recherche, météo) "
            "passent par Internet.\n"
            "- Chaque message de l'utilisateur se termine par une ligne [État …] "
            "ajoutée automatiquement (heure, durée de session, charge machine) : "
            "c'est ta perception du moment présent ; l'utilisateur ne l'a pas "
            "dite, et tu ne la recopies jamais."
        )

    def _state_stamp(self) -> str:
        now = _now()
        return (
            f"[État : {_say_datetime(now)}, {_part_of_day(now)} ; session active "
            f"depuis {_say_duration(now - self.started)} ; {self._machine.load()}]"
        )

    def self_state(self) -> str:
        """Full self-state as text (identity block + live stamp)."""
        return self._identity_block() + "\n" + self._state_stamp()

    @staticmethod
    def _build_digest(events: list[dict[str, Any]]) -> str:
        """Short memory of previous sessions: last summary + last few exchanges."""
        summary, summary_ts = "", None
        exchanges: list[dict[str, Any]] = []
        for ev in events:
            if ev.get("type") == "session_end" and ev.get("summary"):
                summary, summary_ts = ev["summary"], _parse_ts(ev.get("ts"))
            elif ev.get("type") == "exchange":
                exchanges.append(ev)
        exchanges = exchanges[-4:]
        if not summary and not exchanges:
            return ""
        lines = ["Souvenirs de tes conversations précédentes :"]
        if summary:
            when = f" (le {_say_datetime(summary_ts)})" if summary_ts else ""
            lines.append(f"Résumé de la dernière session{when} : {summary}")
        for ev in exchanges:
            ts = _parse_ts(ev.get("ts"))
            when = _say_datetime(ts) if ts else "?"
            lines.append(
                f"[{when}] Utilisateur : {str(ev.get('user', ''))[:200]} / "
                f"Toi : {str(ev.get('assistant', ''))[:200]}"
            )
        return "\n".join(lines)

    def _messages(self, user_msg: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
        msgs: list[dict[str, Any]] = [
            {"role": "system", "content": self._static_prompt}
        ]
        for turn in self._history:
            msgs.extend(turn)
        if isinstance(user_msg, str):
            msgs.append({"role": "user", "content": user_msg})
        else:
            msgs.extend(user_msg)
        return msgs

    def _payload(
        self,
        messages: list[dict[str, Any]],
        stream: bool,
        *,
        tools: bool = True,
        **options: Any,
    ) -> dict[str, Any]:
        opts = {
            "temperature": self.bcfg.temperature,
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
        }
        opts.update(options)
        payload: dict[str, Any] = {
            "model": self.bcfg.model,
            "messages": messages,
            "stream": stream,
            "think": False,
            "keep_alive": self.bcfg.keep_alive,
            "options": opts,
        }
        # Same tool list on every request: it renders into the cached prefix.
        if tools and self.toolbox:
            payload["tools"] = self.toolbox.schemas()
        return payload

    def _greeting_prompt(self) -> str:
        if self.first_ever:
            situation = (
                "C'est ton tout premier démarrage : présente-toi brièvement (ton "
                "nom, que tu tournes en local sur ce Mac) et propose ton aide."
            )
        elif self.prev_last_seen:
            gap = _say_duration(_now() - self.prev_last_seen)
            situation = (
                f"Tu redémarres ; ta dernière conversation remonte à {gap}. Salue "
                "en tenant compte du moment de la journée et de ce délai, en "
                "évoquant la dernière conversation seulement si c'est naturel."
            )
        else:
            situation = "Tu redémarres. Salue brièvement."
        return (
            f"[Démarrage, message système et non de l'utilisateur] {situation} "
            f"Une ou deux phrases courtes.\n\n{self._state_stamp()}"
        )

    # ------------------------------------------------------------- public API

    def warm_up(self) -> None:
        """Load the model and pre-compute the greeting prompt in Ollama's cache."""
        try:
            r = self._client.post(
                "/api/chat",
                json={
                    "model": self.bcfg.model,
                    "messages": [],
                    "keep_alive": self.bcfg.keep_alive,
                    "options": {"num_ctx": NUM_CTX},
                },
            )
            r.raise_for_status()
            # Prefill system prompt + greeting request so greeting() only decodes.
            self._greeting_msg = self._greeting_prompt()
            self._client.post(
                "/api/chat",
                json=self._payload(
                    self._messages(self._greeting_msg), False, num_predict=1
                ),
            )
        except httpx.HTTPError as exc:
            log.warning("Brain warm-up failed: %s", exc)

    def stream_reply(self, user_text: str, stop: threading.Event) -> Iterator[str]:
        user_text = (user_text or "").strip()
        if not user_text:
            return
        try:
            # Persisted now; enters the system prompt at the next rebuild. Until
            # then the statement itself is in the conversation.
            self._facts.learn(user_text)
        except Exception:  # noqa: BLE001 — memory must never break a reply
            log.warning("Fact extraction failed", exc_info=True)

        user_msg = f"{user_text}\n\n{self._state_stamp()}"
        pending, self._pending = self._pending, None
        if pending is not None and time.monotonic() > pending.expires:
            log.info("Pending action expired")
            pending = None
        if pending is not None:
            verdict = classify_confirmation(user_text)
            log.info("Pending action, answer classified as %s", verdict)
            if verdict == "yes":
                yield from self._run_turn(
                    user_text, user_msg, stop, forced_calls=pending.calls
                )
                return
            if verdict == "no":
                ack = "D'accord, j'annule."
                turn = [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": ack},
                ]
                self._record(user_text, turn, ack)
                yield ack
                return
        yield from self._run_turn(user_text, user_msg, stop)

    # ------------------------------------------------------------ the loop

    def _stream_round(
        self,
        messages: list[dict[str, Any]],
        stop: threading.Event,
        tools: bool,
        out: dict[str, Any],
    ) -> Iterator[str]:
        """Stream one model response. Yields spoken deltas; fills ``out`` with
        raw content, tool_calls, error, removed (echoed stamp dropped)."""
        payload = self._payload(messages, True, tools=tools)
        raw: list[str] = []
        calls: list[dict[str, Any]] = []
        think = _ThinkFilter()
        brackets = _BracketFilter()
        spoken = False
        t0 = time.monotonic()
        out.update(raw="", tool_calls=calls, error=None, removed=False, first=None)
        try:
            with self._client.stream("POST", "/api/chat", json=payload) as resp:
                if resp.status_code == 404:
                    resp.read()
                    log.error("Ollama model missing: %s", resp.text[:200])
                    out["error"] = ERR_NO_MODEL
                    return
                if resp.status_code >= 400:
                    resp.read()
                    log.error("Ollama error %s: %s", resp.status_code, resp.text[:200])
                    out["error"] = ERR_GENERIC
                    return
                leading = True
                for line in resp.iter_lines():
                    if stop.is_set():
                        break
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except ValueError:
                        continue
                    if chunk.get("error"):
                        log.error("Ollama stream error: %s", chunk["error"])
                        out["error"] = ERR_GENERIC
                        break
                    msg = chunk.get("message") or {}
                    calls.extend(msg.get("tool_calls") or [])
                    # message.thinking (if any) is deliberately ignored
                    content = msg.get("content", "")
                    raw.append(content)
                    delta = _clean_spoken(brackets.feed(think.feed(content)))
                    if leading:
                        delta = delta.lstrip()
                        leading = not delta
                    if delta:
                        if out["first"] is None:
                            out["first"] = time.monotonic() - t0
                        spoken = True
                        yield delta
                    if chunk.get("done"):
                        log.debug(
                            "Ollama: prompt %s tok in %.2fs, gen %s tok in %.2fs",
                            chunk.get("prompt_eval_count"),
                            chunk.get("prompt_eval_duration", 0) / 1e9,
                            chunk.get("eval_count"),
                            chunk.get("eval_duration", 0) / 1e9,
                        )
                        break
                tail = _clean_spoken(brackets.feed(think.flush()) + brackets.flush())
                if tail and not stop.is_set():
                    spoken = True
                    yield tail
        except httpx.ConnectError:
            out["error"] = ERR_UNREACHABLE
        except httpx.HTTPError as exc:
            log.error("Ollama request failed: %s", exc)
            if not spoken:
                out["error"] = ERR_UNREACHABLE
        out["raw"] = "".join(raw)
        out["removed"] = brackets.removed

    def _execute(
        self, calls: list[dict[str, Any]], stop: threading.Event, quiet: bool
    ) -> Iterator[str]:
        """Run tool calls in parallel; yields at most one progress line.
        Returns the ``tool`` messages (via StopIteration value)."""
        jobs = []
        for call in calls:
            name, args = _call_name(call), _call_args(call)
            tool = self.toolbox.get(name) if self.toolbox else None
            t0 = time.monotonic()
            if tool is None:
                fut = None
            else:
                fut = self._pool.submit(_run_tool, tool, args)
            jobs.append((call, name, args, tool, fut, t0))
        futures = [j[4] for j in jobs if j[4] is not None]
        if futures:
            _done, running = wait_futures(futures, timeout=PROGRESS_AFTER_S)
            if running and not quiet:
                progress = next(
                    (j[3].progress for j in jobs if j[4] in running and j[3].progress),
                    "",
                )
                if progress:
                    yield progress
            deadline = time.monotonic() + TOOL_TIMEOUT_S
            while running and not stop.is_set() and time.monotonic() < deadline:
                _done, running = wait_futures(running, timeout=0.2)
        results: list[dict[str, Any]] = []
        for call, name, args, tool, fut, t0 in jobs:
            if tool is None:
                content = f"Outil inconnu : {name}."
            elif fut is not None and fut.done():
                content = fut.result()
            else:
                content = (
                    "Interrompu avant la fin." if stop.is_set() else "Délai dépassé."
                )
            log.info(
                "Tool %s(%s) -> %.2fs: %s",
                name,
                json.dumps(args, ensure_ascii=False)[:200],
                time.monotonic() - t0,
                content[:160].replace("\n", " "),
            )
            msg: dict[str, Any] = {
                "role": "tool",
                "content": content,
                "tool_name": name,
            }
            if call.get("id"):
                msg["tool_call_id"] = call["id"]
            results.append(msg)
        return results

    def _run_turn(
        self,
        user_text: str,
        user_msg: str,
        stop: threading.Event,
        forced_calls: list[dict[str, Any]] | None = None,
    ) -> Iterator[str]:
        turn: list[dict[str, Any]] = [{"role": "user", "content": user_msg}]
        spoken: list[str] = []
        t0 = time.monotonic()
        first: float | None = None
        removed = False
        error: str | None = None
        ask_confirmation: list[dict[str, Any]] = []
        gave_up = False

        def emit(delta: str, new_segment: bool = True) -> str:
            nonlocal first
            if first is None:
                first = time.monotonic() - t0
            # keep sentences apart across rounds ("Je regarde." + "Demain…")
            if (
                new_segment
                and spoken
                and not spoken[-1][-1:].isspace()
                and not delta[:1].isspace()
            ):
                delta = " " + delta
            spoken.append(delta)
            return delta

        if forced_calls:  # the user just said yes to a pending action
            turn.append(
                {"role": "assistant", "content": "", "tool_calls": forced_calls}
            )
            results = yield from self._emit_all(
                self._execute(forced_calls, stop, quiet=False), emit
            )
            turn.extend(results)

        for rnd in range(MAX_TOOL_ROUNDS):
            if stop.is_set():
                break
            # Last round (or right after deferring an action): text only. The
            # tools stay in the payload anyway: dropping them would change the
            # cached prefix and force a full re-evaluation of the prompt.
            can_act = rnd < MAX_TOOL_ROUNDS - 1 and not ask_confirmation
            out: dict[str, Any] = {}
            round_text: list[str] = []
            # Clear action request ("ouvre…", "mets un minuteur…"): hold the
            # first round's text; if the model only *talks* about acting
            # instead of calling a tool, nudge it once (see ACTION_NUDGE).
            hold = rnd == 0 and not forced_calls and _wants_action(user_text)
            held: list[str] = []
            for delta in self._stream_round(self._messages(turn), stop, True, out):
                if hold:
                    held.append(delta)
                    continue
                yield emit(delta, new_segment=not round_text)
                round_text.append(delta)
            if hold and not out.get("error") and not out["tool_calls"]:
                log.info("Action request answered without a tool: nudging once")
                retry: dict[str, Any] = {}
                nudge = {"role": "system", "content": ACTION_NUDGE}
                list(
                    self._stream_round(
                        self._messages([*turn, nudge]), stop, True, retry
                    )
                )
                if retry.get("tool_calls") and not retry.get("error"):
                    out = retry
                    held = []
            for delta in held:
                yield emit(delta, new_segment=not round_text)
                round_text.append(delta)
            removed = removed or out.get("removed", False)
            if out.get("error"):
                error = out["error"]
                break
            calls = out["tool_calls"]
            if calls and not can_act:
                log.info("Ignoring %d tool call(s) in a text-only round", len(calls))
                calls = []
                gave_up = not round_text and not ask_confirmation
            msg: dict[str, Any] = {"role": "assistant", "content": out["raw"]}
            if calls:
                msg["tool_calls"] = calls
            turn.append(msg)
            if not calls or stop.is_set():
                break
            run_now, deferred = [], []
            for call in calls:
                tool = self.toolbox.get(_call_name(call)) if self.toolbox else None
                (deferred if tool is not None and tool.confirm else run_now).append(
                    call
                )
            results: list[dict[str, Any]] = []
            if run_now:
                results = yield from self._emit_all(
                    self._execute(run_now, stop, quiet=bool(round_text)), emit
                )
            for call in deferred:
                log.info(
                    "Tool %s(%s) needs confirmation",
                    _call_name(call),
                    json.dumps(_call_args(call), ensure_ascii=False)[:200],
                )
                waiting: dict[str, Any] = {
                    "role": "tool",
                    "tool_name": _call_name(call),
                    "content": "NON EXÉCUTÉ : en attente d'un oui de l'utilisateur. "
                    "Demande-lui en une phrase courte s'il confirme, en nommant "
                    "précisément l'action.",
                }
                if call.get("id"):
                    waiting["tool_call_id"] = call["id"]
                results.append(waiting)
            turn.extend(results)
            if deferred:
                ask_confirmation = deferred

        if ask_confirmation and not stop.is_set():
            self._pending = _Pending(
                calls=ask_confirmation, expires=time.monotonic() + PENDING_TTL_S
            )
            text = "".join(spoken)
            if "?" not in text:  # the model forgot to ask: use a template
                question = f"{_describe_call(ask_confirmation[0])}, je confirme ?"
                turn.append({"role": "assistant", "content": question})
                yield emit(question)

        if gave_up and not stop.is_set():
            sorry = "Désolé, je n'ai pas réussi à aller au bout."
            turn[-1] = {"role": "assistant", "content": sorry}
            yield emit(sorry)
        if error and not spoken:
            yield error
            return
        reply = "".join(spoken).strip()
        log.info(
            "Reply: first token %.2fs, total %.2fs, %d chars, %d tool msgs",
            first if first is not None else -1.0,
            time.monotonic() - t0,
            len(reply),
            sum(1 for m in turn if m["role"] == "tool"),
        )
        if not reply:
            return
        if stop.is_set() or removed:
            # Don't let an interruption / echoed stamp reinforce itself: the
            # final assistant message becomes what was actually said.
            final = reply + (" (interrompu)" if stop.is_set() else "")
            if turn[-1]["role"] == "assistant" and not turn[-1].get("tool_calls"):
                turn[-1] = {"role": "assistant", "content": final}
            else:
                turn.append({"role": "assistant", "content": final})
            reply = final
        elif turn[-1]["role"] != "assistant":
            turn.append({"role": "assistant", "content": reply})
        self._record(user_text, turn, reply)

    @staticmethod
    def _emit_all(gen: Any, emit: Any) -> Iterator[str]:
        """Re-yield a sub-generator's text through ``emit``; return its value."""
        try:
            while True:
                yield emit(next(gen))
        except StopIteration as done:
            return done.value

    def record_spoken(self, text: str) -> None:
        """Remember something said unprompted (reminder fired, task done)."""
        text = (text or "").strip()
        if not text:
            return
        self._append_turn([{"role": "assistant", "content": text}])
        self._journal.append({"type": "spoken", "assistant": text})

    def _append_turn(self, turn: list[dict[str, Any]]) -> None:
        with self._lock:
            self._history.append(turn)
            limit = max(1, self.bcfg.max_history_turns)
            if len(self._history) > limit:
                # Trim in one go to ~2/3 so the prompt cache breaks rarely, and
                # refresh the system prompt (new facts) since it breaks anyway.
                del self._history[: len(self._history) - max(1, limit * 2 // 3)]
                self._static_prompt = self._build_static_prompt()

    def _record(self, user_text: str, turn: list[dict[str, Any]], reply: str) -> None:
        self._append_turn(turn)
        with self._lock:
            self._session_exchanges.append((user_text, reply))
            event: dict[str, Any] = {
                "type": "exchange",
                "user": user_text,
                "assistant": reply,
            }
            tools = [
                {"name": _call_name(c), "args": _call_args(c)}
                for m in turn
                for c in m.get("tool_calls") or []
            ]
            if tools:
                event["tools"] = tools
            self._journal.append(event)
            self._state["last_exchange"] = _now().isoformat(timespec="seconds")
            self._state["last_seen"] = self._state["last_exchange"]
            _write_json(self._self_path, self._state)

    def _complete(
        self, messages: list[dict[str, Any]], timeout: float, **options: Any
    ) -> str:
        r = self._client.post(
            "/api/chat",
            json=self._payload(messages, False, tools=False, **options),
            timeout=httpx.Timeout(timeout, connect=min(timeout, 2.0)),
        )
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "")

    @staticmethod
    def _to_spoken(text: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        return _clean_spoken(text).strip()

    def _greeting_template(self) -> str:
        now = _now()
        hello = "Bonsoir" if now.hour >= 18 or now.hour < 5 else "Bonjour"
        user = self._user_name()
        who = f" {user}" if user else ""
        if self.first_ever:
            return (
                f"{hello}{who}. Je suis {self.cfg.assistant.name}, votre assistant. "
                "Je fonctionne entièrement sur votre Mac. Que puis-je faire pour vous ?"
            )
        if self.prev_last_seen:
            gap = now - self.prev_last_seen
            if gap > timedelta(hours=6):
                since = _say_duration(gap)
                return f"{hello}{who}. Ravi de vous retrouver après {since}."
        return f"Rebonjour{who}. Je suis à votre écoute."

    def greeting(self) -> str:
        """Short spoken greeting; model-generated if it answers within ~2 s."""
        fallback = self._greeting_template()
        msg = self._greeting_msg or self._greeting_prompt()
        msgs = self._messages(msg)
        deadline = time.monotonic() + GREETING_BUDGET_S
        parts: list[str] = []
        done = threading.Event()
        abort = threading.Event()

        def run() -> None:
            payload = self._payload(msgs, True, num_predict=70)
            try:
                with self._client.stream(
                    "POST",
                    "/api/chat",
                    json=payload,
                    timeout=httpx.Timeout(GREETING_BUDGET_S + 1, connect=1.0),
                ) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if abort.is_set():
                            break
                        if line:
                            chunk = json.loads(line)
                            parts.append(chunk.get("message", {}).get("content", ""))
                            if chunk.get("done"):
                                break
            except Exception as exc:  # noqa: BLE001
                log.info("Greeting generation failed: %s", exc)
            finally:
                done.set()

        threading.Thread(target=run, daemon=True, name="brain-greeting").start()
        done.wait(max(0.0, deadline - time.monotonic()))
        abort.set()
        raw = "".join(parts)
        text = _STAMP_RE.sub("", self._to_spoken(raw)).strip()
        if text and text[-1] not in ".!?…":
            # Timed out or hit the token limit: keep complete sentences only.
            cut = max(text.rfind(c) for c in ".!?…")
            text = text[: cut + 1] if cut > 0 else ""
        if text != raw.strip():
            raw = text
        if not text or len(text) > 300:
            log.info("Greeting: using template")
            text, raw = fallback, fallback
        with self._lock:
            # Keep it in the transcript: continuity, and the cache stays valid.
            if not self._history:
                self._history.append(
                    [
                        {"role": "user", "content": msg},
                        {"role": "assistant", "content": raw},
                    ]
                )
        return text

    def shutdown(self) -> None:
        """Write the end-of-session note (with a model-made summary when possible)."""
        summary = ""
        if self._session_exchanges:
            convo = "\n".join(
                f"Utilisateur : {u[:300]}\nAssistant : {a[:300]}"
                for u, a in self._session_exchanges[-12:]
            )
            msgs = [
                {
                    "role": "system",
                    "content": "Tu résumes des conversations pour un journal. "
                    "Réponds en français, une ou deux phrases, sans préambule.",
                },
                {
                    "role": "user",
                    "content": "Résume cette session, en notant ce que l'utilisateur a "
                    f"demandé et ce qui reste en suspens :\n\n{convo}",
                },
            ]
            try:
                summary = self._to_spoken(
                    self._complete(msgs, timeout=10.0, num_predict=120, temperature=0.3)
                )
            except Exception as exc:  # noqa: BLE001
                log.info("Session summary skipped: %s", exc)
        end = _now()
        self._journal.append(
            {
                "type": "session_end",
                "session": self.session_number,
                "exchanges": len(self._session_exchanges),
                "duration_s": int((end - self.started).total_seconds()),
                "summary": summary,
            }
        )
        self._state["last_seen"] = end.isoformat(timespec="seconds")
        _write_json(self._self_path, self._state)
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._client.close()
