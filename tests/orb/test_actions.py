"""Tests for the orb's reflexes (pure logic only: no network, no side effects)."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from openjarvis.orb import actions
from openjarvis.orb.actions import (
    TZ,
    Reflexes,
    ReminderStore,
    extract_main_text,
    format_weather,
    parse_ddg_html,
    parse_duration,
    parse_when,
    resolve_app,
    resolve_url,
    wmo_to_french,
    words_to_digits,
)
from openjarvis.orb.config import JarvisConfig
from openjarvis.orb.toolkit import Notifier

# Wednesday 18 November 2026, noon, Brussels.
NOW = datetime(2026, 11, 18, 12, 0, tzinfo=TZ)


class FakeClock:
    def __init__(self, start: datetime = NOW) -> None:
        self.t = start

    def __call__(self) -> datetime:
        return self.t

    def advance(self, **kw: float) -> None:
        self.t += timedelta(**kw)


class FakeRunner:
    """Records commands instead of running them."""

    def __init__(self, stdout: str = "", returncode: int = 0) -> None:
        self.calls: list[tuple[list[str], Any]] = []
        self.stdout = stdout
        self.returncode = returncode

    def __call__(
        self, args: list[str], timeout: float, input_text: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((args, input_text))
        return subprocess.CompletedProcess(args, self.returncode, self.stdout, "")


@pytest.fixture
def reflexes(tmp_path: Path) -> Reflexes:
    r = Reflexes(
        JarvisConfig(),
        Notifier(),
        data_dir=tmp_path / "orb",
        notes_path=tmp_path / "Jarvis" / "notes.md",
        now=FakeClock(),
        runner=FakeRunner(),
        http=object(),  # any network use would fail loudly
        start=False,
    )
    yield r
    r.close()


# ------------------------------------------------------------ numbers/time


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("dix-huit heures trente", "18 heures 30"),
        ("vingt-cinq minutes", "25 minutes"),
        ("quatre-vingt-dix secondes", "90 secondes"),
        ("soixante et onze", "71"),
        ("vingt et une heures", "21 heures"),
    ],
)
def test_words_to_digits(text: str, expected: str) -> None:
    assert words_to_digits(text) == expected


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("10 minutes", 600),
        ("1h30", 5400),
        ("1 h 30", 5400),
        ("90 secondes", 90),
        ("5 secondes", 5),
        ("une heure et demie", 5400),
        ("un quart d'heure", 900),
        ("une demi-heure", 1800),
        ("deux minutes trente", 150),
        ("vingt-cinq minutes", 1500),
        ("10 min 30 s", 630),
        ("00:05:00", 300),
        ("3", 180),
    ],
)
def test_parse_duration(text: str, seconds: float) -> None:
    assert parse_duration(text) == seconds


def test_parse_duration_rejects_garbage() -> None:
    assert parse_duration("bientôt") is None
    assert parse_duration("") is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("dans 20 minutes", NOW + timedelta(minutes=20)),
        ("dans 10 secondes", NOW + timedelta(seconds=10)),
        ("dans une heure et demie", NOW + timedelta(minutes=90)),
        ("demain à 9h", datetime(2026, 11, 19, 9, 0, tzinfo=TZ)),
        ("à 18h30", datetime(2026, 11, 18, 18, 30, tzinfo=TZ)),
        ("à dix-huit heures trente", datetime(2026, 11, 18, 18, 30, tzinfo=TZ)),
        ("à 9h", datetime(2026, 11, 19, 9, 0, tzinfo=TZ)),  # passed -> tomorrow
        ("lundi à 10h", datetime(2026, 11, 23, 10, 0, tzinfo=TZ)),
        ("mercredi à 14h", datetime(2026, 11, 18, 14, 0, tzinfo=TZ)),
        ("mercredi à 10h", datetime(2026, 11, 25, 10, 0, tzinfo=TZ)),
        ("ce soir", datetime(2026, 11, 18, 20, 0, tzinfo=TZ)),
        ("ce soir à 8h", datetime(2026, 11, 18, 20, 0, tzinfo=TZ)),
        ("demain matin", datetime(2026, 11, 19, 9, 0, tzinfo=TZ)),
        ("après-demain à 7h15", datetime(2026, 11, 20, 7, 15, tzinfo=TZ)),
        ("le 25 décembre à 8h", datetime(2026, 12, 25, 8, 0, tzinfo=TZ)),
        ("à midi et demi", datetime(2026, 11, 18, 12, 30, tzinfo=TZ)),
        ("à 17 heures moins le quart", datetime(2026, 11, 18, 16, 45, tzinfo=TZ)),
        ("2026-11-19T09:00:00", datetime(2026, 11, 19, 9, 0, tzinfo=TZ)),
    ],
)
def test_parse_when(text: str, expected: datetime) -> None:
    assert parse_when(text, NOW) == expected


def test_parse_when_unknown() -> None:
    assert parse_when("un de ces jours", NOW) is None


# --------------------------------------------------------------- reminders


def test_reminder_persists_and_fires_with_fake_clock(tmp_path: Path) -> None:
    clock = FakeClock()
    fired: list[tuple[str, bool]] = []
    path = tmp_path / "reminders.json"
    store = ReminderStore(
        path, lambda it, late: fired.append((it["text"], late)), clock
    )
    store.add("boire de l'eau", clock() + timedelta(seconds=10))
    assert json.loads(path.read_text())[0]["text"] == "boire de l'eau"

    # a new process sees it
    store2 = ReminderStore(
        path, lambda it, late: fired.append((it["text"], late)), clock
    )
    assert [i["text"] for i in store2.pending()] == ["boire de l'eau"]
    assert store2.check_due() == []
    clock.advance(seconds=11)
    assert [i["text"] for i in store2.check_due()] == ["boire de l'eau"]
    assert fired == [("boire de l'eau", False)]
    assert json.loads(path.read_text()) == []


def test_missed_reminder_fires_late_at_startup(tmp_path: Path) -> None:
    clock = FakeClock()
    path = tmp_path / "reminders.json"
    ReminderStore(path, lambda *_: None, clock).add("appeler maman", clock())
    clock.advance(hours=2)
    fired: list[tuple[str, bool]] = []
    store = ReminderStore(
        path, lambda it, late: fired.append((it["text"], late)), clock
    )
    store.check_due(startup=True)
    assert fired == [("appeler maman", True)]


def test_reflex_timer_announces_and_notifies(reflexes: Reflexes) -> None:
    assert reflexes.set_timer("5 secondes") == "Minuteur de 5 secondes lancé."
    reflexes.now.advance(seconds=6)  # type: ignore[attr-defined]
    reflexes.reminders.check_due()
    said = reflexes.notifier.drain()
    assert said[0].urgent and "minuteur de 5 secondes" in said[0].text.lower()
    runner: FakeRunner = reflexes._run  # type: ignore[assignment]
    assert runner.calls[-1][0][0] == "osascript"
    assert "display notification" in runner.calls[-1][0][2]


def test_late_reminder_text(reflexes: Reflexes) -> None:
    reflexes.set_reminder("sortir le chien", "dans 5 minutes")
    reflexes.now.advance(hours=1)  # type: ignore[attr-defined]
    reflexes.reminders.check_due(startup=True)
    text = reflexes.notifier.drain()[0].text
    assert "en retard" in text and "sortir le chien" in text


def test_list_and_cancel_reminders(reflexes: Reflexes) -> None:
    assert "Aucun" in reflexes.list_reminders()
    reflexes.set_reminder("acheter du pain", "demain à 9h")
    reflexes.set_reminder("appeler Paul", "à 18h30")
    reflexes.set_timer("10 minutes")
    listing = reflexes.list_reminders()
    assert "3 en cours" in listing and "acheter du pain" in listing
    assert "pain" in reflexes.cancel_reminder("le rappel pour le pain")
    assert "minuteur" in reflexes.cancel_reminder("le minuteur")
    assert "Paul" in reflexes.cancel_reminder("tous")
    assert reflexes.reminders.pending() == []


def test_set_reminder_rejects_unknown_time(reflexes: Reflexes) -> None:
    assert "incompris" in reflexes.set_reminder("x", "un jour")


# ------------------------------------------------------------ web/weather

DDG_FIXTURE = """
<div class="result results_links results_links_deep result--ad">
  <a rel="nofollow" class="result__a" href="https://duckduckgo.com/y.js?ad=1">Pub</a>
  <a class="result__snippet" href="#">Achetez maintenant</a>
</div>
<div class="result results_links results_links_deep web-result ">
  <h2 class="result__title">
    <a rel="nofollow" class="result__a" href="https://fr.wikipedia.org/wiki/F1">Formule 1 &mdash; <b>Wikipédia</b></a>
  </h2>
  <div class="result__extras"><div class="result__extras__url">
    <a class="result__url" href="https://fr.wikipedia.org/wiki/F1">fr.wikipedia.org</a>
  </div></div>
  <a class="result__snippet" href="https://fr.wikipedia.org/wiki/F1">Le <b>champion</b> 2025 est Lando Norris.</a>
</div>
<div class="result results_links results_links_deep web-result ">
  <h2 class="result__title">
    <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fgp&amp;rut=abc">Résultats F1 2026</a>
  </h2>
  <a class="result__snippet" href="#">Vainqueurs et podiums par Grand Prix.</a>
</div>
"""  # noqa: E501


def test_parse_ddg_html() -> None:
    res = parse_ddg_html(DDG_FIXTURE)
    assert [r["url"] for r in res] == [
        "https://fr.wikipedia.org/wiki/F1",
        "https://example.com/gp",
    ]
    assert res[0]["title"] == "Formule 1 — Wikipédia"
    assert res[0]["snippet"] == "Le champion 2025 est Lando Norris."


def test_extract_main_text() -> None:
    page = (
        "<html><script>var x=1;</script><nav><p>Menu menu menu menu menu menu "
        "menu menu menu</p></nav><p>Lando Norris a remporté le Grand Prix de "
        "Las Vegas devant Max Verstappen.</p><p>court</p></html>"
    )
    text = extract_main_text(page)
    assert text.startswith("Lando Norris a remporté")
    assert "Menu" not in text and "var x" not in text


def test_wmo_mapping() -> None:
    assert wmo_to_french(0) == "ciel dégagé"
    assert wmo_to_french(63) == "pluie modérée"
    assert wmo_to_french("95") == "orage"
    assert wmo_to_french(42) == "temps variable"
    assert wmo_to_french(None) == "temps inconnu"


def test_format_weather_tomorrow_and_today() -> None:
    data = {
        "current": {
            "temperature_2m": 7.4,
            "apparent_temperature": 4.6,
            "weather_code": 3,
            "wind_speed_10m": 14.2,
        },
        "daily": {
            "time": ["2026-11-18", "2026-11-19"],
            "weather_code": [3, 61],
            "temperature_2m_min": [3.1, 4.2],
            "temperature_2m_max": [8.8, 9.6],
            "precipitation_probability_max": [10, 80],
            "wind_speed_10m_max": [20.0, 31.5],
        },
    }
    tomorrow = format_weather("Bruxelles", data, "tomorrow")
    assert "jeudi 19" in tomorrow and "pluie faible" in tomorrow
    assert "4 °C à 10 °C" in tomorrow and "80 %" in tomorrow
    today = format_weather("Bruxelles", data, "today")
    assert "couvert" in today and "7 °C" in today


# ------------------------------------------------------------- apps/urls


def test_resolve_app() -> None:
    index = {
        "visual studio code": "Visual Studio Code",
        "calculator": "Calculator",
        "google chrome": "Google Chrome",
        "safari": "Safari",
        "system settings": "System Settings",
    }
    assert resolve_app("VS Code", index) == "Visual Studio Code"
    assert resolve_app("la calculatrice", index) == "Calculator"
    assert resolve_app("safari", index) == "Safari"
    assert resolve_app("Safar", index) == "Safari"
    assert resolve_app("visual studio", index) == "Visual Studio Code"
    assert resolve_app("zzzzzz", index) is None


def test_resolve_url() -> None:
    assert resolve_url("youtube") == "https://www.youtube.com"
    assert resolve_url("lesoir.be") == "https://lesoir.be"
    assert resolve_url("https://x.org/a") == "https://x.org/a"
    assert resolve_url("le site de la SNCB").startswith("https://duckduckgo.com/?q=")


def test_open_app_uses_open_dash_a(reflexes: Reflexes) -> None:
    reflexes._apps = {"calculator": "Calculator"}
    assert reflexes.open_app("calculatrice") == "Calculator est ouvert."
    assert reflexes._run.calls[-1][0] == ["open", "-a", "Calculator"]  # type: ignore[attr-defined]


# ---------------------------------------------------------- notes/misc


def test_notes_roundtrip(reflexes: Reflexes) -> None:
    assert "Aucune" in reflexes.read_notes()
    assert reflexes.take_note("acheter des piles") == "Noté."
    reflexes.take_note("rappeler le garage")
    out = reflexes.read_notes(last_n=1)
    assert "2 notes" in out and "rappeler le garage" in out
    assert "acheter des piles" not in out


def test_clipboard_passes_text_on_stdin(reflexes: Reflexes) -> None:
    assert "Copié" in reflexes.copy_to_clipboard("héllo")
    assert reflexes._run.calls[-1] == (["pbcopy"], "héllo")  # type: ignore[attr-defined]


def test_tools_respect_flags(tmp_path: Path) -> None:
    cfg = JarvisConfig()
    cfg.actions.web_search = False
    r = Reflexes(cfg, Notifier(), data_dir=tmp_path, start=False)
    names = {t.name for t in r.tools()}
    assert "web_search" not in names and "weather" in names
    assert {t.name for t in r.tools() if t.confirm} == {"quit_app"}
    cfg.actions.enabled = False
    assert r.tools() == []


def test_build_actions_and_close_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openjarvis.orb import config as orb_config

    monkeypatch.setattr(orb_config, "DATA_DIR", tmp_path)
    tools = actions.build_actions(JarvisConfig(), Notifier())
    assert len(tools) >= 14
    actions.close_all()
