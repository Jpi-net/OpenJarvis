"""Waking the orb by saying its name (local Whisper transcript)."""

from __future__ import annotations

import pytest

from openjarvis.orb.ears import _after_name


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Jarvis, quelle heure est-il ?", "quelle heure est-il"),
        ("Hey Jarvis", ""),
        ("Jarvie, rappelle-moi d'appeler Camille.", "rappelle-moi d'appeler Camille"),
        ("J'arvis ?", ""),
        ("Dis Jarvis, mets de la musique", "mets de la musique"),
        ("Bonjour Jarvisse.", ""),
        ("Je parlais de Jarvis hier", None),  # name not addressed
        ("Quelle heure est-il ?", None),
    ],
)
def test_after_name(text: str, expected: str | None) -> None:
    assert _after_name(text, ["jarvis"]) == expected
