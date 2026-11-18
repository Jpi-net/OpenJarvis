"""Single-file configuration for the voice-only orb (``~/.openjarvis/jarvis.toml``).

Every user-facing setting of the orb lives here. The file is created with
commented defaults on first run; missing keys fall back to the defaults below.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import tomllib

CONFIG_PATH = Path.home() / ".openjarvis" / "jarvis.toml"
DATA_DIR = Path.home() / ".openjarvis" / "orb"  # models, journal, cache


@dataclass
class AssistantConfig:
    name: str = "Jarvis"
    language: str = "fr"
    user_name: str = ""  # how the assistant addresses you; empty = no name


@dataclass
class BrainConfig:
    host: str = "http://127.0.0.1:11434"  # Ollama
    model: str = "qwen3.5:9b"
    temperature: float = 0.6
    max_history_turns: int = 12
    keep_alive: str = "30m"  # keep the model loaded in RAM between requests


@dataclass
class EarsConfig:
    wake_word: str = "hey_jarvis"  # openWakeWord model name
    wake_threshold: float = 0.5
    # "auto" = mlx-whisper on the Apple GPU when available, else faster-whisper
    stt_engine: str = "auto"
    stt_model: str = "large-v3-turbo"  # tiny, base, small, medium, large-v3-turbo
    silence_seconds: float = 0.8  # end of utterance after this much silence
    max_utterance_seconds: float = 30.0
    follow_up_seconds: float = 6.0  # answer again without the wake word
    input_device: str = ""  # empty = system default microphone


@dataclass
class VoiceConfig:
    engine: str = "piper"  # "piper" (neural, local) or "say" (macOS built-in)
    piper_voice: str = "fr_FR-tom-medium"
    say_voice: str = "Thomas"
    speed: float = 1.0


@dataclass
class OrbConfig:
    size: int = 220  # window size in pixels
    # top-left, top-right, bottom-left, bottom-right, center
    position: str = "bottom-right"
    color: str = "#4fc3ff"


@dataclass
class JarvisConfig:
    assistant: AssistantConfig = field(default_factory=AssistantConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    ears: EarsConfig = field(default_factory=EarsConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    orb: OrbConfig = field(default_factory=OrbConfig)


DEFAULT_TOML = """\
# Configuration de l'orbe Jarvis — le seul endroit où se règle l'assistant.
# Lancer : jarvis orb

[assistant]
name = "Jarvis"
language = "fr"
user_name = ""            # comment il t'appelle (vide = pas de nom)

[brain]
host = "http://127.0.0.1:11434"   # serveur Ollama
model = "qwen3.5:9b"
temperature = 0.6
max_history_turns = 12
keep_alive = "30m"        # garde le modèle chargé en mémoire

[ears]
wake_word = "hey_jarvis"  # mot d'activation (openWakeWord)
wake_threshold = 0.5      # plus haut = moins de faux déclenchements
stt_engine = "auto"       # auto (GPU Apple si dispo), mlx, faster-whisper
stt_model = "large-v3-turbo" # tiny, base, small, medium, large-v3-turbo
silence_seconds = 0.8     # fin de phrase après ce silence
max_utterance_seconds = 30
follow_up_seconds = 6     # délai pour répondre sans redire le mot d'activation
input_device = ""         # vide = micro par défaut

[voice]
engine = "piper"          # "piper" (voix neuronale locale) ou "say" (voix macOS)
piper_voice = "fr_FR-tom-medium"
say_voice = "Thomas"
speed = 1.0

[orb]
size = 220
position = "bottom-right" # top-left, top-right, bottom-left, bottom-right, center
color = "#4fc3ff"
"""


def _merge(section_cls: type, raw: dict[str, Any]) -> Any:
    known = {f.name for f in fields(section_cls)}
    return section_cls(**{k: v for k, v in raw.items() if k in known})


def load_config(path: Path = CONFIG_PATH) -> JarvisConfig:
    """Load the orb config, creating the file with defaults if missing."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(DEFAULT_TOML, encoding="utf-8")
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    cfg = JarvisConfig()
    for f in fields(JarvisConfig):
        section = raw.get(f.name)
        if isinstance(section, dict):
            setattr(cfg, f.name, _merge(type(getattr(cfg, f.name)), section))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return cfg
