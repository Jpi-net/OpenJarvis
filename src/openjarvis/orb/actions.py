"""Reflexes: instant local actions the brain can call (no API key, < 1 s).

Apps, URLs, web search (DuckDuckGo HTML), weather (Open-Meteo), timers and
reminders (persistent, one scheduler thread), music, volume, system status,
notes and clipboard. Every tool returns a short French string for the model.

Usage::

    reflexes = Reflexes(cfg, notifier)   # starts the reminder scheduler
    toolbox.add(*reflexes.tools())
    ...
    reflexes.close()

``build_actions(cfg, notifier)`` is the one-call shortcut (the scheduler
thread is a daemon; call ``close_all()`` on shutdown to stop it cleanly).
"""

from __future__ import annotations

import difflib
import html
import json
import logging
import re
import shutil
import subprocess
import threading
import unicodedata
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
from zoneinfo import ZoneInfo

from openjarvis.orb import config as orb_config
from openjarvis.orb.config import JarvisConfig
from openjarvis.orb.toolkit import Notifier, Tool

log = logging.getLogger("openjarvis.orb.actions")

TZ = ZoneInfo("Europe/Brussels")
NOTES_PATH = Path.home() / "Jarvis" / "notes.md"
APP_DIRS = [
    Path("/Applications"),
    Path("/Applications/Utilities"),
    Path("/System/Applications"),
    Path("/System/Applications/Utilities"),
    Path.home() / "Applications",
]
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15",
    "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
    "Accept-Language": "fr-BE,fr;q=0.9,en;q=0.8",
}
PAGE_TEXT_CHARS = 1000  # each extra 1000 chars costs ~1 s of prompt eval

_DAYS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
_MONTHS = [
    "janvier",
    "fevrier",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "aout",
    "septembre",
    "octobre",
    "novembre",
    "decembre",
]


# ------------------------------------------------------------------ helpers


def _fold(text: str) -> str:
    """Lowercase, strip accents, normalise apostrophes and spaces."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", text).strip()


def _say_time(dt: datetime) -> str:
    return f"{dt.hour} h {dt.minute:02d}"


def _say_when(dt: datetime, now: datetime) -> str:
    """'à 14 h 30', 'demain à 9 h 00', 'jeudi 20 novembre à 10 h 00'."""
    day = (dt.date() - now.date()).days
    if day == 0:
        prefix = ""
    elif day == 1:
        prefix = "demain "
    elif day == 2:
        prefix = "après-demain "
    else:
        month = [
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
        ][dt.month - 1]
        prefix = f"{_DAYS[dt.weekday()]} {dt.day} {month} "
    return f"{prefix}à {_say_time(dt)}"


def say_duration(seconds: float) -> str:
    s = int(round(seconds))
    h, rest = divmod(s, 3600)
    m, sec = divmod(rest, 60)
    parts = []
    if h:
        parts.append(f"{h} heure" + ("s" if h > 1 else ""))
    if m:
        parts.append(f"{m} minute" + ("s" if m > 1 else ""))
    if sec and not h:
        parts.append(f"{sec} seconde" + ("s" if sec > 1 else ""))
    return " ".join(parts) or "0 seconde"


def _osa_quote(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


# ------------------------------------------------- French numbers & time

_UNITS = {
    "zero": 0,
    "un": 1,
    "une": 1,
    "deux": 2,
    "trois": 3,
    "quatre": 4,
    "cinq": 5,
    "six": 6,
    "sept": 7,
    "huit": 8,
    "neuf": 9,
    "dix": 10,
    "onze": 11,
    "douze": 12,
    "treize": 13,
    "quatorze": 14,
    "quinze": 15,
    "seize": 16,
}
_TENS = {
    "vingt": 20,
    "trente": 30,
    "quarante": 40,
    "cinquante": 50,
    "soixante": 60,
}
_NUM_WORD = (
    r"(?:zero|une?|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze"
    r"|treize|quatorze|quinze|seize|vingts?|trente|quarante|cinquante|soixante"
    r"|quatrevingts?|quatre-vingts?|et)"
)
_NUM_SEQ_RE = re.compile(rf"\b{_NUM_WORD}(?:[- ]{_NUM_WORD})*\b")


def _parse_number(tokens: list[str], i: int) -> tuple[int, int] | None:
    """Greedy French number (0..99) starting at tokens[i] -> (value, next i)."""
    n = len(tokens)

    def unit(j: int, top: int) -> tuple[int, int] | None:
        if j >= n:
            return None
        t = tokens[j]
        if t == "dix" and j + 1 < n and tokens[j + 1] in ("sept", "huit", "neuf"):
            if 10 + _UNITS[tokens[j + 1]] <= top:
                return 10 + _UNITS[tokens[j + 1]], j + 2
        if t in _UNITS and 0 < _UNITS[t] <= top:
            return _UNITS[t], j + 1
        return None

    t = tokens[i]
    ten = 80 if t == "quatrevingt" else _TENS.get("vingt" if t == "vingts" else t)
    if ten is not None:
        j = i + 1
        top = 19 if ten in (60, 80) else 9
        if j < n and tokens[j] == "et":
            u = unit(j + 1, top)
            if u and u[0] in (1, 11):
                return ten + u[0], u[1]
            return ten, j
        u = unit(j, top)
        if u:
            return ten + u[0], u[1]
        return ten, j
    u = unit(i, 19)
    if u:
        return u
    if t == "zero":
        return 0, i + 1
    return None


def words_to_digits(text: str) -> str:
    """'dix-huit heures trente' -> '18 heures 30' (0..99, folded text)."""

    def repl(m: re.Match[str]) -> str:
        raw = re.sub(r"quatre[- ]vingts?", "quatrevingt", m.group(0))
        tokens = [t for t in re.split(r"[- ]", raw) if t]
        out: list[str] = []
        i = 0
        while i < len(tokens):
            parsed = None if tokens[i] == "et" else _parse_number(tokens, i)
            if parsed is None:
                out.append(tokens[i])
                i += 1
            else:
                out.append(str(parsed[0]))
                i = parsed[1]
        return " ".join(out)

    return _NUM_SEQ_RE.sub(repl, text)


_DUR_UNITS = {
    "s": 1,
    "sec": 1,
    "secs": 1,
    "seconde": 1,
    "secondes": 1,
    "second": 1,
    "seconds": 1,
    "m": 60,
    "mn": 60,
    "min": 60,
    "mins": 60,
    "minute": 60,
    "minutes": 60,
    "h": 3600,
    "heure": 3600,
    "heures": 3600,
    "hour": 3600,
    "hours": 3600,
    "j": 86400,
    "jour": 86400,
    "jours": 86400,
}
_DUR_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*([a-z]+)?")


def parse_duration(text: str) -> float | None:
    """French/English duration -> seconds. '10 minutes', '1h30', '90 secondes',
    'une heure et demie', 'un quart d'heure', 'une demi-heure', '00:05:00'."""
    t = _fold(str(text))
    if not t:
        return None
    t = t.replace("demi-heure", "30 minutes").replace("demi heure", "30 minutes")
    t = re.sub(r"(?:un |1 )?quarts? d'heure", "15 minutes", t)
    t = re.sub(r"trois quarts d'heure", "45 minutes", t)
    t = words_to_digits(t)
    m = re.fullmatch(r"(\d+):(\d{2})(?::(\d{2}))?", t)
    if m:
        a, b, c = int(m.group(1)), int(m.group(2)), m.group(3)
        return a * 3600 + b * 60 + int(c) if c is not None else a * 60 + b
    if re.fullmatch(r"\d+(?:[.,]\d+)?", t):  # bare number = minutes
        return float(t.replace(",", ".")) * 60
    total = 0.0
    found = False
    last_unit = 0
    for m in _DUR_RE.finditer(t):
        value = float(m.group(1).replace(",", "."))
        unit = m.group(2) or ""
        if unit in _DUR_UNITS:
            last_unit = _DUR_UNITS[unit]
            total += value * last_unit
            found = True
        elif not unit and found and last_unit > 1:
            # '1h30' / '2 minutes 30': trailing number = next smaller unit
            total += value * (60 if last_unit == 3600 else 1)
        elif unit in ("et",) and found:
            continue
    if not found:
        return None
    if "et demie" in t or "et demi" in t:
        total += last_unit / 2 if last_unit >= 60 else 0
    elif "et quart" in t and last_unit == 3600:
        total += 900
    return total if total > 0 else None


_TIME_RE = re.compile(
    r"\b(\d{1,2})\s*(?:h|heures?|:)\s*(\d{1,2})?\b(?!\s*(?:minutes?|min|secondes?))"
)


def _parse_clock(t: str) -> tuple[int, int] | None:
    """Find a time of day in folded, digit-converted text."""
    if re.search(r"\bmidi\b", t):
        h, m = 12, 0
        mm = re.search(r"midi\s*(?:et\s*)?(\d{1,2})\b", t)
        if mm:
            m = int(mm.group(1))
        base: tuple[int, int] | None = (h, m)
    elif re.search(r"\bminuit\b", t):
        base = (0, 0)
    else:
        base = None
        for mt in _TIME_RE.finditer(t):
            h = int(mt.group(1))
            m = int(mt.group(2)) if mt.group(2) else 0
            if h < 24 and m < 60:
                base = (h, m)
                break
        if base is None:
            mt = re.search(
                r"\ba (\d{1,2})\b(?!\s*(?:minutes?|min|jours?|secondes?))", t
            )
            if mt and int(mt.group(1)) < 24:
                base = (int(mt.group(1)), 0)
    if base is None:
        return None
    h, m = base
    if re.search(r"et demie?\b", t):
        m = 30
    elif re.search(r"et quart\b", t):
        m = 15
    elif re.search(r"moins (?:le )?quart\b", t):
        h, m = (h - 1) % 24, 45
    if (re.search(r"\b(?:du soir|de l'apres-midi|de l'aprem|pm)\b", t) and h < 12) or (
        re.search(r"\bce soir\b", t) and 1 <= h < 12
    ):
        h += 12
    return h, m


def parse_when(text: str, now: datetime | None = None) -> datetime | None:
    """Natural French date/time -> aware datetime (Europe/Brussels).

    'dans 20 minutes', 'demain à 9h', 'à 18h30', 'lundi à 10h', 'ce soir',
    'après-demain', 'le 25 décembre à 8h', ISO strings."""
    now = (now or datetime.now(TZ)).astimezone(TZ)
    raw = str(text or "").strip()
    if not raw:
        return None
    try:  # the model sometimes sends ISO timestamps
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt.astimezone(TZ) if dt.tzinfo else dt.replace(tzinfo=TZ)
    except ValueError:
        pass
    t = words_to_digits(_fold(raw))
    t = t.replace("apres midi", "apres-midi")

    m = re.search(r"\bdans (.+)", t)
    if m:
        secs = parse_duration(m.group(1))
        if secs:
            return now + timedelta(seconds=secs)

    day: datetime | None = None
    explicit_day = False
    if re.search(r"\bapres-demain\b", t):
        day, explicit_day = now + timedelta(days=2), True
    elif re.search(r"\bdemain\b", t):
        day, explicit_day = now + timedelta(days=1), True
    elif re.search(r"\baujourd'hui\b|\bce (?:soir|matin)\b|\bcet apres-midi\b", t):
        day, explicit_day = now, True
    else:
        for i, name in enumerate(_DAYS):
            if re.search(rf"\b{name}\b", t):
                ahead = (i - now.weekday()) % 7
                if "prochain" in t and ahead == 0:
                    ahead = 7
                day, explicit_day = now + timedelta(days=ahead), True
                break
        if day is None:
            md = re.search(
                r"\b(?:le )?(\d{1,2}|1er)\s+(" + "|".join(_MONTHS) + r")\b", t
            )
            if md:
                d = 1 if md.group(1) == "1er" else int(md.group(1))
                month = _MONTHS.index(md.group(2)) + 1
                try:
                    cand = now.replace(month=month, day=d)
                    if cand.date() < now.date():
                        cand = cand.replace(year=now.year + 1)
                    day, explicit_day = cand, True
                except ValueError:
                    return None

    clock = _parse_clock(t)
    if clock is None:
        if re.search(r"\bce soir\b|\bsoir\b", t):
            clock = (20, 0)
        elif re.search(r"\bmatin\b", t):
            clock = (9, 0)
        elif re.search(r"\bapres-midi\b", t):
            clock = (15, 0)
        elif explicit_day:
            clock = (9, 0)
        else:
            return None
    base = day or now
    dt = base.replace(hour=clock[0], minute=clock[1], second=0, microsecond=0)
    if dt <= now:
        if not explicit_day:
            dt += timedelta(days=1)
        elif (
            day is not None
            and day.date() == now.date()
            and re.search("|".join(rf"\b{d}\b" for d in _DAYS), t)
        ):
            dt += timedelta(days=7)
    return dt


# ------------------------------------------------------------- web search

_RESULT_RE = re.compile(
    r'<div class="result(?P<cls>(?: [^"]*)?)">.*?'
    r'class="result__a"[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>'
    r'(?P<body>.*?)(?=<div class="result[ "]|\Z)',
    re.S,
)
_SNIPPET_RE = re.compile(r'class="result__snippet"[^>]*>(?P<snip>.*?)</a>', re.S)
_TAG_RE = re.compile(r"<[^>]+>")


def _strip_tags(fragment: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(_TAG_RE.sub("", fragment))).strip()


def _ddg_url(href: str) -> str:
    href = html.unescape(href)
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg")
        if target:
            return unquote(target[0])
    return href


def parse_ddg_html(page: str, limit: int = 5) -> list[dict[str, str]]:
    """Extract organic results (title, url, snippet) from html.duckduckgo.com."""
    results = []
    for m in _RESULT_RE.finditer(page):
        if "result--ad" in m.group("cls"):
            continue
        url = _ddg_url(m.group("href"))
        if "duckduckgo.com/y.js" in url:  # ad redirect
            continue
        snip = _SNIPPET_RE.search(m.group("body"))
        results.append(
            {
                "title": _strip_tags(m.group("title")),
                "url": url,
                "snippet": _strip_tags(snip.group("snip")) if snip else "",
            }
        )
        if len(results) >= limit:
            break
    return results


def extract_main_text(page: str, limit: int = PAGE_TEXT_CHARS) -> str:
    """Rough readable text of an HTML page: paragraphs first."""
    page = re.sub(
        r"<(script|style|noscript|nav|header|footer|aside|form|svg)\b.*?</\1>",
        " ",
        page,
        flags=re.S | re.I,
    )
    paras = [
        _strip_tags(p)
        for p in re.findall(
            r"<(?:p|li|h[1-3])\b[^>]*>(.*?)</(?:p|li|h[1-3])>", page, re.S | re.I
        )
    ]
    paras = [p for p in paras if len(p) > 40]
    text = " ".join(paras) if paras else _strip_tags(page)
    return text[:limit].rsplit(" ", 1)[0] if len(text) > limit else text


# ---------------------------------------------------------------- weather

WMO_FR = {
    0: "ciel dégagé",
    1: "plutôt dégagé",
    2: "partiellement nuageux",
    3: "couvert",
    45: "brouillard",
    48: "brouillard givrant",
    51: "bruine légère",
    53: "bruine",
    55: "bruine dense",
    56: "bruine verglaçante",
    57: "bruine verglaçante dense",
    61: "pluie faible",
    63: "pluie modérée",
    65: "forte pluie",
    66: "pluie verglaçante",
    67: "forte pluie verglaçante",
    71: "neige faible",
    73: "neige modérée",
    75: "forte neige",
    77: "grains de neige",
    80: "averses faibles",
    81: "averses",
    82: "violentes averses",
    85: "averses de neige",
    86: "fortes averses de neige",
    95: "orage",
    96: "orage avec grêle",
    99: "orage avec forte grêle",
}


def wmo_to_french(code: Any) -> str:
    try:
        return WMO_FR.get(int(code), "temps variable")
    except (TypeError, ValueError):
        return "temps inconnu"


def _deg(v: Any) -> str:
    return f"{round(float(v))} °C"


def format_weather(place: str, data: dict[str, Any], when: str) -> str:
    daily = data.get("daily", {})

    def day_line(i: int, label: str) -> str:
        code = wmo_to_french(daily["weather_code"][i])
        tmin, tmax = daily["temperature_2m_min"][i], daily["temperature_2m_max"][i]
        rain = daily.get("precipitation_probability_max", [None] * 8)[i]
        wind = daily.get("wind_speed_10m_max", [None] * 8)[i]
        s = f"{label} : {code}, de {_deg(tmin)} à {_deg(tmax)}"
        if rain is not None:
            s += f", risque de pluie {rain} %"
        if wind is not None:
            s += f", vent jusqu'à {round(wind)} km/h"
        return s

    dates = daily.get("time", [])

    def label(i: int) -> str:
        d = datetime.fromisoformat(dates[i])
        return f"{_DAYS[d.weekday()]} {d.day}"

    if when == "tomorrow":
        return f"{place}, demain ({label(1)}) — " + day_line(1, "prévision")
    if when == "week":
        return f"{place}, 7 jours. " + " ; ".join(
            day_line(i, label(i)) for i in range(min(7, len(dates)))
        )
    cur = data.get("current", {})
    now = (
        f"{place}, maintenant : {wmo_to_french(cur.get('weather_code'))}, "
        f"{_deg(cur.get('temperature_2m', 0))} (ressenti "
        f"{_deg(cur.get('apparent_temperature', 0))}), vent "
        f"{round(cur.get('wind_speed_10m', 0))} km/h"
    )
    return now + ". " + day_line(0, "Aujourd'hui")


def _norm_when(when: str) -> str:
    w = _fold(when or "")
    if "demain" in w or "tomorrow" in w:
        return "tomorrow"
    if "semaine" in w or "week" in w or "jours" in w:
        return "week"
    return "today"


# ------------------------------------------------------------ app lookup

APP_ALIASES = {
    "vs code": "Visual Studio Code",
    "vscode": "Visual Studio Code",
    "code": "Visual Studio Code",
    "calculatrice": "Calculator",
    "calculette": "Calculator",
    "calendrier": "Calendar",
    "agenda": "Calendar",
    "musique": "Music",
    "plans": "Maps",
    "cartes": "Maps",
    "rappels": "Reminders",
    "horloge": "Clock",
    "apercu": "Preview",
    "reglages": "System Settings",
    "reglages systeme": "System Settings",
    "preferences systeme": "System Settings",
    "parametres": "System Settings",
    "moniteur d'activite": "Activity Monitor",
    "utilitaire de disque": "Disk Utility",
    "chrome": "Google Chrome",
    "edge": "Microsoft Edge",
    "word": "Microsoft Word",
    "excel": "Microsoft Excel",
    "powerpoint": "Microsoft PowerPoint",
    "outlook": "Microsoft Outlook",
    "teams": "Microsoft Teams",
    "livres": "Books",
    "dictionnaire": "Dictionary",
    "echecs": "Chess",
    "localiser": "FindMy",
    "contacts": "Contacts",
    "meteo": "Weather",
    "photos": "Photos",
    "capture d'ecran": "Screenshot",
    "terminal": "Terminal",
    "finder": "Finder",
}


def _app_index() -> dict[str, str]:
    """folded display name -> real app name, for every installed .app."""
    index: dict[str, str] = {}
    for d in APP_DIRS:
        try:
            entries = list(d.iterdir())
        except OSError:
            continue
        for p in entries:
            if p.suffix == ".app":
                index.setdefault(_fold(p.stem), p.stem)
    return index


def resolve_app(name: str, index: dict[str, str]) -> str | None:
    q = _fold(name).removesuffix(".app").strip(" .!?")
    q = re.sub(r"^(?:l'|la |le |les |l |application |app )+", "", q).strip()
    if not q:
        return None
    if q in APP_ALIASES:
        return APP_ALIASES[q]
    if q in index:
        return index[q]
    names = list(index)
    for key in names:  # prefix, then word prefix, then substring
        if key.startswith(q):
            return index[key]
    for key in names:
        if any(w.startswith(q) for w in key.split()):
            return index[key]
    for key in names:
        if q in key:
            return index[key]
    compact = q.replace(" ", "")
    for key in names:  # initials: "vsc" / "vs code"
        initials = "".join(w[0] for w in key.split() if w)
        if len(compact) >= 2 and initials.startswith(compact[: len(initials)]):
            if compact.startswith(initials[:2]) and len(initials) >= 2:
                return index[key]
    close = difflib.get_close_matches(q, names, n=1, cutoff=0.72)
    return index[close[0]] if close else None


SITES = {
    "youtube": "https://www.youtube.com",
    "google": "https://www.google.com",
    "gmail": "https://mail.google.com",
    "google maps": "https://maps.google.com",
    "maps": "https://maps.google.com",
    "wikipedia": "https://fr.wikipedia.org",
    "github": "https://github.com",
    "netflix": "https://www.netflix.com",
    "twitter": "https://x.com",
    "x": "https://x.com",
    "facebook": "https://www.facebook.com",
    "instagram": "https://www.instagram.com",
    "linkedin": "https://www.linkedin.com",
    "amazon": "https://www.amazon.fr",
    "reddit": "https://www.reddit.com",
    "chatgpt": "https://chatgpt.com",
    "claude": "https://claude.ai",
    "rtbf": "https://www.rtbf.be",
    "le soir": "https://www.lesoir.be",
    "spotify": "https://open.spotify.com",
}


def resolve_url(target: str) -> str:
    t = target.strip().strip("\"'")
    if re.match(r"^[a-z][a-z0-9+.-]*://", t, re.I):
        return t
    key = _fold(t).removeprefix("le site ").removeprefix("site ").strip()
    if key in SITES:
        return SITES[key]
    if re.fullmatch(r"[\w.-]+\.[a-z]{2,}(?:/\S*)?", key):
        return "https://" + key
    # unknown name: DuckDuckGo "I'm feeling ducky" jumps to the first result
    return "https://duckduckgo.com/?q=" + quote_plus("! " + t)


# --------------------------------------------------------------- reminders


class ReminderStore:
    """Timers + reminders persisted as JSON, fired by one scheduler thread."""

    def __init__(
        self,
        path: Path,
        on_fire: Callable[[dict[str, Any], bool], None],
        now: Callable[[], datetime] = lambda: datetime.now(TZ),
    ) -> None:
        self.path = path
        self.on_fire = on_fire
        self.now = now
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.items: list[dict[str, Any]] = self._load()

    def _load(self) -> list[dict[str, Any]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return [i for i in data if isinstance(i, dict) and i.get("due") and i.get("id")]

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(self.items, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            tmp.replace(self.path)
        except OSError:
            log.warning("Could not save reminders", exc_info=True)

    def add(self, text: str, due: datetime, kind: str = "reminder") -> dict[str, Any]:
        item = {
            "id": uuid.uuid4().hex[:8],
            "kind": kind,
            "text": text,
            "due": due.astimezone(TZ).isoformat(timespec="seconds"),
            "created": self.now().isoformat(timespec="seconds"),
        }
        with self._cond:
            self.items.append(item)
            self.items.sort(key=lambda i: i["due"])
            self._save()
            self._cond.notify_all()
        return item

    def remove(self, ids: set[str]) -> list[dict[str, Any]]:
        with self._cond:
            gone = [i for i in self.items if i["id"] in ids]
            self.items = [i for i in self.items if i["id"] not in ids]
            self._save()
            self._cond.notify_all()
        return gone

    def pending(self) -> list[dict[str, Any]]:
        with self._cond:
            return sorted(self.items, key=lambda i: i["due"])

    def check_due(self, *, startup: bool = False) -> list[dict[str, Any]]:
        """Fire everything due now; returns the fired items."""
        now = self.now()
        with self._cond:
            due = [i for i in self.items if datetime.fromisoformat(i["due"]) <= now]
            if due:
                ids = {i["id"] for i in due}
                self.items = [i for i in self.items if i["id"] not in ids]
                self._save()
        for item in due:
            late = startup or (
                now - datetime.fromisoformat(item["due"]) > timedelta(seconds=90)
            )
            try:
                self.on_fire(item, late)
            except Exception:  # noqa: BLE001
                log.exception("Reminder callback failed")
        return due

    def _next_wait(self) -> float:
        with self._cond:
            if not self.items:
                return 3600.0
            nxt = min(datetime.fromisoformat(i["due"]) for i in self.items)
        return max(0.0, min(3600.0, (nxt - self.now()).total_seconds()))

    def start(self) -> None:
        self.check_due(startup=True)
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="orb-reminders"
        )
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            wait = self._next_wait()
            with self._cond:
                if wait > 0:
                    self._cond.wait(min(wait, 30.0))  # re-sync clock periodically
            if self._stop.is_set():
                break
            self.check_due()

    def close(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2)


# ---------------------------------------------------------------- reflexes


def _default_runner(
    args: list[str], timeout: float, input_text: str | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed local binaries
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
        input=input_text,
        env={"LANG": "en_US.UTF-8", "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
    )


def _obj(props: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required or []}


def _s(desc: str, **extra: Any) -> dict[str, Any]:
    return {"type": "string", "description": desc, **extra}


class Reflexes:
    def __init__(
        self,
        cfg: JarvisConfig,
        notifier: Notifier,
        *,
        data_dir: Path | None = None,
        notes_path: Path | None = None,
        now: Callable[[], datetime] | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
        http: Any = None,
        start: bool = True,
    ) -> None:
        self.cfg = cfg
        self.acfg = cfg.actions
        self.notifier = notifier
        self.data_dir = Path(data_dir or orb_config.DATA_DIR)
        self.notes_path = Path(notes_path or NOTES_PATH)
        self.now = now or (lambda: datetime.now(TZ))
        self._run = runner or _default_runner
        self._http = http
        self._http_lock = threading.Lock()
        self._geo_cache: dict[str, tuple[float, float, str]] = {}
        self._apps: dict[str, str] | None = None
        self.reminders = ReminderStore(
            self.data_dir / "reminders.json", self._fire, now=self.now
        )
        if start and self.acfg.enabled:
            self.reminders.start()

    # ------------------------------------------------------------ plumbing

    def _client(self) -> Any:
        with self._http_lock:
            if self._http is None:
                import httpx
                import truststore

                truststore.inject_into_ssl()
                self._http = httpx.Client(
                    headers=HTTP_HEADERS,
                    timeout=httpx.Timeout(6.0, connect=3.0),
                    follow_redirects=True,
                )
            return self._http

    def _osascript(self, script: str, timeout: float = 5.0) -> tuple[bool, str]:
        try:
            r = self._run(["osascript", "-e", script], timeout)
        except (OSError, subprocess.SubprocessError) as exc:
            return False, str(exc)
        return r.returncode == 0, (r.stdout or r.stderr or "").strip()

    def _fire(self, item: dict[str, Any], late: bool) -> None:
        due = datetime.fromisoformat(item["due"]).astimezone(TZ)
        if item.get("kind") == "timer":
            text = f"Le minuteur de {item['text']} est terminé."
            if late:
                text = f"Minuteur en retard, il a fini à {_say_time(due)}."
        else:
            text = f"Rappel : {item['text']}."
            if late:
                text = f"Rappel en retard, prévu à {_say_time(due)} : {item['text']}."
        self.notifier.announce(text, urgent=True)
        title = "Minuteur" if item.get("kind") == "timer" else "Rappel"
        self._osascript(
            f"display notification {_osa_quote(text)} with title "
            f'{_osa_quote(self.cfg.assistant.name + " — " + title)} sound name "Glass"',
            timeout=3.0,
        )

    def close(self) -> None:
        self.reminders.close()
        if self._http is not None:
            try:
                self._http.close()
            except Exception:  # noqa: BLE001
                pass

    # --------------------------------------------------------------- apps

    def open_app(self, name: str) -> str:
        if self._apps is None:
            self._apps = _app_index()
        app = resolve_app(name, self._apps)
        if app is None:
            return f"Aucune application ne correspond à « {name} »."
        r = self._run(["open", "-a", app], 5.0)
        if r.returncode != 0:
            return f"Impossible d'ouvrir {app}."
        return f"{app} est ouvert."

    def quit_app(self, name: str) -> str:
        if self._apps is None:
            self._apps = _app_index()
        app = resolve_app(name, self._apps) or name.strip()
        ok, out = self._osascript(f"tell application {_osa_quote(app)} to quit")
        return f"{app} est fermé." if ok else f"Impossible de fermer {app} : {out[:80]}"

    def open_url(self, url_or_site: str) -> str:
        url = resolve_url(url_or_site)
        r = self._run(["open", url], 5.0)
        if r.returncode != 0:
            return f"Impossible d'ouvrir {url}."
        return f"Ouvert dans le navigateur : {url}"

    # ------------------------------------------------------------ the web

    def web_search(self, query: str, read_top: bool = True) -> str:
        client = self._client()
        try:
            r = client.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query, "kl": "be-fr"},
                headers={"Referer": "https://html.duckduckgo.com/"},
            )
            r.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            return f"Recherche impossible ({type(exc).__name__})."
        if "anomaly-modal" in r.text:
            return "Recherche bloquée par DuckDuckGo pour le moment."
        results = parse_ddg_html(r.text, limit=4)
        if not results:
            return f"Aucun résultat pour « {query} »."
        lines = [
            f"{i}. {res['title']} ({urlparse(res['url']).netloc}) : {res['snippet']}"
            for i, res in enumerate(results, 1)
        ]
        out = "Résultats web :\n" + "\n".join(lines)
        if read_top:
            text = ""
            for res in results[:2]:
                try:
                    page = client.get(res["url"], timeout=2.5)
                    if page.status_code < 400 and "html" in page.headers.get(
                        "content-type", "html"
                    ):
                        text = extract_main_text(page.text)
                except Exception:  # noqa: BLE001
                    continue
                if len(text) > 200:
                    out += f"\n\nExtrait de {urlparse(res['url']).netloc} : {text}"
                    break
        return out

    def _geocode(self, place: str) -> tuple[float, float, str] | None:
        key = _fold(place)
        if key in self._geo_cache:
            return self._geo_cache[key]
        r = self._client().get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": place, "count": 1, "language": "fr", "format": "json"},
        )
        r.raise_for_status()
        res = (r.json().get("results") or [None])[0]
        if not res:
            return None
        hit = (float(res["latitude"]), float(res["longitude"]), res.get("name", place))
        self._geo_cache[key] = hit
        return hit

    def weather(self, place: str = "", when: str = "today") -> str:
        place = (place or "").strip() or self.acfg.city
        try:
            geo = self._geocode(place)
            if geo is None:
                return f"Lieu introuvable : {place}."
            lat, lon, name = geo
            r = self._client().get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,apparent_temperature,weather_code,"
                    "wind_speed_10m",
                    "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
                    "precipitation_probability_max,wind_speed_10m_max",
                    "timezone": "Europe/Brussels",
                    "forecast_days": 7,
                },
            )
            r.raise_for_status()
            return format_weather(name, r.json(), _norm_when(when))
        except Exception as exc:  # noqa: BLE001
            log.warning("Weather failed: %s", exc)
            return f"Météo indisponible ({type(exc).__name__})."

    # ----------------------------------------------------- timers/reminders

    def set_timer(self, duration: str, label: str = "") -> str:
        secs = parse_duration(duration)
        if not secs:
            return f"Durée incomprise : « {duration} »."
        spoken = say_duration(secs)
        self.reminders.add(
            label or spoken, self.now() + timedelta(seconds=secs), kind="timer"
        )
        return f"Minuteur de {spoken} lancé."

    def set_reminder(self, text: str, when: str) -> str:
        now = self.now()
        due = parse_when(when, now)
        if due is None:
            return f"Moment incompris : « {when} ». Demande une heure précise."
        if due <= now:
            return "Ce moment est déjà passé."
        self.reminders.add(text.strip() or "rappel", due)
        return f"Rappel enregistré {_say_when(due, now)} : {text}."

    def list_reminders(self) -> str:
        items = self.reminders.pending()
        if not items:
            return "Aucun rappel ni minuteur en cours."
        now = self.now()
        parts = []
        for i, it in enumerate(items, 1):
            due = datetime.fromisoformat(it["due"]).astimezone(TZ)
            if it.get("kind") == "timer":
                left = say_duration(max(0, (due - now).total_seconds()))
                parts.append(f"{i}. minuteur {it['text']}, reste {left}")
            else:
                parts.append(f"{i}. {it['text']}, {_say_when(due, now)}")
        return f"{len(items)} en cours : " + " ; ".join(parts)

    def cancel_reminder(self, which: str) -> str:
        items = self.reminders.pending()
        if not items:
            return "Il n'y a rien à annuler."
        q = _fold(which or "")
        if re.search(r"\b(?:tous|tout|toutes)\b", q):
            chosen = items
        elif re.fullmatch(r"\d+", q) and 1 <= int(q) <= len(items):
            chosen = [items[int(q) - 1]]
        elif "dernier" in q:
            chosen = [max(items, key=lambda i: i["created"])]
        elif "minuteur" in q or "timer" in q:
            timers = [i for i in items if i.get("kind") == "timer"]
            chosen = timers[:1] if len(timers) == 1 else timers
        else:
            words = set(re.findall(r"\w{3,}", q)) - {"rappel", "annule", "pour"}
            scored = [
                (len(words & set(re.findall(r"\w{3,}", _fold(i["text"])))), i)
                for i in items
            ]
            best = max(s for s, _ in scored)
            chosen = [i for s, i in scored if s == best and s > 0][:1]
            if not chosen and len(items) == 1:
                chosen = items
        if not chosen:
            return "Je ne trouve pas ce rappel. " + self.list_reminders()
        gone = self.reminders.remove({i["id"] for i in chosen})
        names = ", ".join(
            ("minuteur " if i.get("kind") == "timer" else "") + i["text"] for i in gone
        )
        return f"Annulé : {names}."

    # ------------------------------------------------------- music/volume

    def music(self, action: str, query: str = "") -> str:
        app = self.acfg.music_app or "Music"
        a = _fold(action)
        verbs = {
            "play": "play",
            "joue": "play",
            "lecture": "play",
            "reprendre": "play",
            "resume": "play",
            "pause": "pause",
            "stop": "pause",
            "toggle": "playpause",
            "next": "next track",
            "suivant": "next track",
            "suivante": "next track",
            "previous": "previous track",
            "precedent": "previous track",
            "precedente": "previous track",
        }
        if a in ("volume", "son"):
            try:
                level = max(0, min(100, int(re.findall(r"\d+", query)[0])))
            except IndexError:
                return "Indique un volume entre 0 et 100."
            ok, out = self._osascript(
                f"tell application {_osa_quote(app)} to set sound volume to {level}"
            )
            return f"Volume de {app} à {level} %." if ok else f"Échec : {out[:80]}"
        if a in ("now", "current", "quoi", "titre", "what"):
            ok, out = self._osascript(
                f"tell application {_osa_quote(app)} to if player state is playing "
                'then (name of current track) & " — " & (artist of current track)'
            )
            return f"En cours : {out}" if ok and out else "Rien ne joue."
        verb = verbs.get(a)
        if verb is None:
            return f"Action musicale inconnue : {action}."
        if verb == "play" and query.strip():
            return self._play_query(app, query.strip())
        ok, out = self._osascript(f"tell application {_osa_quote(app)} to {verb}")
        if not ok:
            return f"{app} ne répond pas : {out[:80]}"
        done = {
            "play": "Lecture lancée.",
            "pause": "Musique en pause.",
            "playpause": "Lecture basculée.",
            "next track": "Morceau suivant.",
            "previous track": "Morceau précédent.",
        }
        return done[verb]

    def _play_query(self, app: str, query: str) -> str:
        if app.lower() == "spotify":
            self._run(["open", "spotify:search:" + quote_plus(query)], 5.0)
            return f"Recherche « {query} » ouverte dans Spotify."
        q = _osa_quote(query)
        script = (
            'tell application "Music"\n'
            f"  set hits to (every user playlist whose name contains {q})\n"
            "  if hits is not {} then\n"
            "    play item 1 of hits\n"
            '    return "playlist " & name of item 1 of hits\n'
            "  end if\n"
            "  set hits to (every track of library playlist 1 "
            f"whose name contains {q})\n"
            f"  if hits is {{}} then set hits to (every track of library playlist 1 "
            f"whose artist contains {q})\n"
            f"  if hits is {{}} then set hits to (every track of library playlist 1 "
            f"whose album contains {q})\n"
            "  if hits is not {} then\n"
            "    play item 1 of hits\n"
            '    return "morceau " & name of item 1 of hits & " — " & artist of '
            "item 1 of hits\n"
            "  end if\n"
            '  return ""\n'
            "end tell"
        )
        ok, out = self._osascript(script, timeout=8.0)
        if ok and out:
            return f"Lecture : {out}."
        if not ok:
            return f"Musique ne répond pas : {out[:80]}"
        return f"Rien trouvé pour « {query} » dans la bibliothèque."

    def set_volume(self, level: int) -> str:
        try:
            level = max(0, min(100, int(float(level))))
        except (TypeError, ValueError):
            return "Volume invalide (0 à 100)."
        ok, out = self._osascript(f"set volume output volume {level}")
        return f"Volume réglé à {level} %." if ok else f"Échec : {out[:80]}"

    def get_status(self) -> str:
        parts = [f"Il est {_say_time(self.now())}"]
        try:
            batt = self._run(["pmset", "-g", "batt"], 2.0).stdout
            m = re.search(
                r"(\d+)%;\s*([^;]+);\s*(\S+ remaining|\(no estimate\))?", batt
            )
            if m:
                state = {
                    "charging": "en charge",
                    "charged": "chargée",
                    "discharging": "sur batterie",
                    "finishing charge": "fin de charge",
                    "AC attached": "sur secteur",
                }.get(m.group(2).strip(), m.group(2).strip())
                s = f"batterie {m.group(1)} % ({state}"
                rem = re.search(r"(\d+):(\d{2}) remaining", batt)
                if rem and state == "sur batterie" and rem.group(0) != "0:00 remaining":
                    s += f", {int(rem.group(1))} h {rem.group(2)} restantes"
                parts.append(s + ")")
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            out = self._run(["ipconfig", "getsummary", "en0"], 2.0).stdout
            m = re.search(r"^\s*SSID : (.+)$", out, re.M)
            if m and "redacted" not in m.group(1):
                parts.append(f"Wi-Fi {m.group(1).strip()}")
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            du = shutil.disk_usage("/")
            parts.append(
                f"disque {du.free / 1e9:.0f} Go libres sur {du.total / 1e9:.0f}"
            )
        except OSError:
            pass
        ok, vol = self._osascript("output volume of (get volume settings)", timeout=2.0)
        if ok and vol.isdigit():
            parts.append(f"volume {vol} %")
        return ", ".join(parts) + "."

    # ------------------------------------------------------- notes/clip

    def take_note(self, text: str) -> str:
        text = " ".join(str(text).split())
        if not text:
            return "Note vide."
        try:
            self.notes_path.parent.mkdir(parents=True, exist_ok=True)
            new = not self.notes_path.exists()
            with self.notes_path.open("a", encoding="utf-8") as fh:
                if new:
                    fh.write("# Notes de Jarvis\n\n")
                fh.write(f"- {self.now():%Y-%m-%d %H:%M} — {text}\n")
        except OSError as exc:
            return f"Impossible d'écrire la note : {exc}"
        return "Noté."

    def read_notes(self, last_n: int = 5) -> str:
        try:
            lines = [
                ln[2:]
                for ln in self.notes_path.read_text(encoding="utf-8").splitlines()
                if ln.startswith("- ")
            ]
        except OSError:
            return "Aucune note pour l'instant."
        if not lines:
            return "Aucune note pour l'instant."
        n = max(1, min(20, int(last_n or 5)))
        return f"{len(lines)} notes au total, les dernières : " + " | ".join(lines[-n:])

    def copy_to_clipboard(self, text: str) -> str:
        try:
            r = self._run(["pbcopy"], 2.0, str(text))
        except (OSError, subprocess.SubprocessError) as exc:
            return f"Échec de la copie : {exc}"
        return "Copié dans le presse-papiers." if r.returncode == 0 else "Échec."

    # ---------------------------------------------------------- schemas

    def tools(self) -> list[Tool]:
        if not self.acfg.enabled:
            return []
        tools = [
            Tool(
                "open_app",
                "Ouvre une application du Mac.",
                _obj(
                    {"name": _s("nom de l'application, ex. Safari, VS Code")}, ["name"]
                ),
                self.open_app,
            ),
            Tool(
                "quit_app",
                "Quitte une application du Mac.",
                _obj({"name": _s("nom de l'application")}, ["name"]),
                self.quit_app,
                confirm=True,
            ),
            Tool(
                "open_url",
                "Ouvre un site ou une adresse web dans le navigateur.",
                _obj(
                    {"url_or_site": _s("URL ou nom de site, ex. youtube")},
                    ["url_or_site"],
                ),
                self.open_url,
            ),
            Tool(
                "weather",
                "Météo réelle (actuelle ou prévisions) d'un lieu.",
                _obj(
                    {
                        "place": _s("ville ; vide = ville par défaut"),
                        "when": _s("période", enum=["today", "tomorrow", "week"]),
                    }
                ),
                self.weather,
                progress="Je regarde la météo.",
            ),
            Tool(
                "set_timer",
                "Lance un minuteur (compte à rebours).",
                _obj({"duration": _s("durée, ex. 10 minutes, 1h30")}, ["duration"]),
                self.set_timer,
            ),
            Tool(
                "set_reminder",
                "Programme un rappel à un moment donné.",
                _obj(
                    {
                        "text": _s("ce qu'il faut rappeler"),
                        "when": _s("moment en mots, ex. dans 20 minutes, demain à 9h"),
                    },
                    ["text", "when"],
                ),
                self.set_reminder,
            ),
            Tool(
                "list_reminders",
                "Liste les rappels et minuteurs en cours.",
                _obj({}),
                self.list_reminders,
            ),
            Tool(
                "cancel_reminder",
                "Annule un rappel ou minuteur.",
                _obj(
                    {"which": _s("texte, numéro, 'minuteur', 'dernier' ou 'tous'")},
                    ["which"],
                ),
                self.cancel_reminder,
            ),
            Tool(
                "music",
                f"Musique ({self.acfg.music_app}) : jouer, pause, suivant, précédent.",
                _obj(
                    {
                        "action": _s(
                            "action",
                            enum=["play", "pause", "next", "previous", "volume", "now"],
                        ),
                        "query": _s(
                            "morceau/artiste/playlist à jouer, ou volume 0-100"
                        ),
                    },
                    ["action"],
                ),
                self.music,
            ),
            Tool(
                "set_volume",
                "Règle le volume du Mac (monter, baisser, couper le son).",
                _obj(
                    {"level": {"type": "integer", "description": "0 à 100"}}, ["level"]
                ),
                self.set_volume,
            ),
            Tool(
                "get_status",
                "État du Mac : batterie, disque, volume.",
                _obj({}),
                self.get_status,
            ),
            Tool(
                "take_note",
                "Écrit une note dans le carnet (« note que… », « prends note »).",
                _obj({"text": _s("contenu de la note")}, ["text"]),
                self.take_note,
            ),
            Tool(
                "read_notes",
                "Relit les dernières notes.",
                _obj({"last_n": {"type": "integer", "description": "nombre de notes"}}),
                self.read_notes,
            ),
            Tool(
                "copy_to_clipboard",
                "Copie un texte dans le presse-papiers.",
                _obj({"text": _s("texte à copier")}, ["text"]),
                self.copy_to_clipboard,
            ),
        ]
        if self.acfg.web_search:
            tools.insert(
                3,
                Tool(
                    "web_search",
                    "Recherche sur Internet : actualité, faits récents, tout ce que "
                    "tu ne sais pas avec certitude. Pour l'actualité, mets le mois "
                    "et l'année dans la requête.",
                    _obj({"query": _s("requête de recherche")}, ["query"]),
                    self.web_search,
                    progress="Je cherche.",
                ),
            )
        return tools


_ACTIVE: list[Reflexes] = []


def build_actions(cfg: JarvisConfig, notifier: Notifier) -> list[Tool]:
    """Create the reflexes (starts the reminder scheduler) and return the tools."""
    reflexes = Reflexes(cfg, notifier)
    _ACTIVE.append(reflexes)
    return reflexes.tools()


def close_all() -> None:
    """Stop the scheduler(s) started by ``build_actions``."""
    while _ACTIVE:
        _ACTIVE.pop().close()


__all__ = [
    "Reflexes",
    "ReminderStore",
    "build_actions",
    "close_all",
    "parse_duration",
    "parse_when",
    "parse_ddg_html",
    "extract_main_text",
    "wmo_to_french",
]
