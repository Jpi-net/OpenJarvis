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
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import subprocess
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator
from zoneinfo import ZoneInfo

import httpx

from openjarvis.orb import config as orb_config
from openjarvis.orb.config import JarvisConfig

log = logging.getLogger("openjarvis.orb.brain")

TZ = ZoneInfo("Europe/Brussels")
MAX_FACTS = 30
MAX_FACT_CHARS = 160
NUM_CTX = 8192
NUM_PREDICT = 400
GREETING_BUDGET_S = 1.9

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
    """

    def __init__(self, cfg: JarvisConfig):
        self.cfg = cfg
        self.bcfg = cfg.brain
        data_dir = Path(orb_config.DATA_DIR)
        data_dir.mkdir(parents=True, exist_ok=True)
        self._client = httpx.Client(
            base_url=self.bcfg.host.rstrip("/"),
            timeout=httpx.Timeout(60.0, connect=3.0),
        )
        self._machine = _MachineProbe()
        self._facts = _Facts(data_dir / "facts.json")
        self._journal = _Journal(data_dir / "journal.jsonl")
        self._self_path = data_dir / "self.json"
        # (user message as sent to the model, raw assistant output)
        self._history: list[tuple[str, str]] = []
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
            "Aucune donnée ne quitte la machine.\n"
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

    def _messages(self, user_msg: str) -> list[dict[str, str]]:
        msgs = [{"role": "system", "content": self._static_prompt}]
        for u, a in self._history:
            msgs.append({"role": "user", "content": u})
            msgs.append({"role": "assistant", "content": a})
        msgs.append({"role": "user", "content": user_msg})
        return msgs

    def _payload(
        self, messages: list[dict[str, str]], stream: bool, **options: Any
    ) -> dict[str, Any]:
        opts = {
            "temperature": self.bcfg.temperature,
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
        }
        opts.update(options)
        return {
            "model": self.bcfg.model,
            "messages": messages,
            "stream": stream,
            "think": False,
            "keep_alive": self.bcfg.keep_alive,
            "options": opts,
        }

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
        payload = self._payload(self._messages(user_msg), True)
        raw: list[str] = []
        spoken: list[str] = []
        think = _ThinkFilter()
        brackets = _BracketFilter()
        error: str | None = None
        t0 = time.monotonic()
        first = None
        try:
            with self._client.stream("POST", "/api/chat", json=payload) as resp:
                if resp.status_code == 404:
                    resp.read()
                    log.error("Ollama model missing: %s", resp.text[:200])
                    error = ERR_NO_MODEL
                elif resp.status_code >= 400:
                    resp.read()
                    log.error("Ollama error %s: %s", resp.status_code, resp.text[:200])
                    error = ERR_GENERIC
                else:
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
                            error = ERR_GENERIC
                            break
                        # message.thinking (if any) is deliberately ignored
                        content = chunk.get("message", {}).get("content", "")
                        raw.append(content)
                        delta = _clean_spoken(brackets.feed(think.feed(content)))
                        if leading:
                            delta = delta.lstrip()
                            leading = not delta
                        if delta:
                            if first is None:
                                first = time.monotonic() - t0
                            spoken.append(delta)
                            yield delta
                        if chunk.get("done"):
                            break
                    tail = _clean_spoken(
                        brackets.feed(think.flush()) + brackets.flush()
                    )
                    if tail and not stop.is_set():
                        spoken.append(tail)
                        yield tail
        except httpx.ConnectError:
            error = ERR_UNREACHABLE
        except httpx.HTTPError as exc:
            log.error("Ollama request failed: %s", exc)
            if not spoken:
                error = ERR_UNREACHABLE
        if error and not spoken:
            yield error
            return
        reply = "".join(spoken).strip()
        log.info(
            "Reply: first token %.2fs, total %.2fs, %d chars",
            first if first is not None else -1.0,
            time.monotonic() - t0,
            len(reply),
        )
        if reply:
            if stop.is_set():
                reply += " (interrompu)"
                raw = [reply]
            elif brackets.removed:
                raw = [reply]  # don't let an echoed stamp reinforce itself
            self._record(user_text, user_msg, "".join(raw), reply)

    def _record(self, user_text: str, user_msg: str, raw: str, reply: str) -> None:
        with self._lock:
            self._history.append((user_msg, raw))
            limit = max(1, self.bcfg.max_history_turns)
            if len(self._history) > limit:
                # Trim in one go to ~2/3 so the prompt cache breaks rarely, and
                # refresh the system prompt (new facts) since it breaks anyway.
                del self._history[: len(self._history) - max(1, limit * 2 // 3)]
                self._static_prompt = self._build_static_prompt()
            self._session_exchanges.append((user_text, reply))
            self._journal.append(
                {"type": "exchange", "user": user_text, "assistant": reply}
            )
            self._state["last_exchange"] = _now().isoformat(timespec="seconds")
            self._state["last_seen"] = self._state["last_exchange"]
            _write_json(self._self_path, self._state)

    def _complete(
        self, messages: list[dict[str, str]], timeout: float, **options: Any
    ) -> str:
        r = self._client.post(
            "/api/chat",
            json=self._payload(messages, False, **options),
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
                self._history.append((msg, raw))
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
        self._client.close()
