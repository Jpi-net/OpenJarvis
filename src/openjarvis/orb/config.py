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
    user_name: str = "Monsieur"  # how the assistant addresses you


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
    barge_in: bool = True  # say the wake word while it talks to interrupt it
    barge_in_threshold: float = 0.75  # stricter: the mic also hears its voice


@dataclass
class VoiceConfig:
    # Local engines: "piper", "say" (macOS built-in).
    # Online engines (text leaves the Mac): "edge" (Microsoft), "elevenlabs".
    engine: str = "piper"
    local_only: bool = True  # refuse online engines even if selected
    edge_voice: str = "fr-FR-RemyMultilingualNeural"
    pitch: str = "+0Hz"  # edge only, e.g. "-2Hz"
    elevenlabs_voice: str = ""  # empty = default premade voice
    elevenlabs_model: str = "eleven_flash_v2_5"
    online_effect: str = "none"  # voice_fx preset for edge / elevenlabs
    piper_voice: str = "fr_FR-tom-medium"
    say_voice: str = "Thomas"
    speed: float = 1.15
    effect: str = "jarvis"  # piper/say only: none, jarvis, jarvis_deep, …


@dataclass
class OrbConfig:
    size: int = 220  # window size in pixels
    # top-left, top-right, bottom-left, bottom-right, center
    position: str = "bottom-right"
    color: str = "#4fc3ff"


@dataclass
class ActionsConfig:
    """Instant local actions (the "reflexes")."""

    enabled: bool = True
    city: str = "Bruxelles"  # default place for weather
    music_app: str = "Music"  # "Music" or "Spotify"
    web_search: bool = True


@dataclass
class HandsConfig:
    """Long tasks (build, connect…) delegated to Claude Code in the background."""

    enabled: bool = True
    workspace: str = "~/Jarvis"  # where delegated work happens
    model: str = ""  # Claude Code model alias; empty = Claude Code default
    confirm: bool = False  # True = ask out loud before starting a task
    # Per-task cap in API-equivalent USD. With a Claude subscription nothing
    # is billed: it only limits how much of the usage quota one task can use.
    max_budget_usd: float = 2.0
    max_parallel: int = 2
    services_model: str = ""  # model for ask_services; empty = sonnet
    services_budget_usd: float = 0.5
    extra_domains: list[str] = field(default_factory=list)  # e.g. "api.x.com"
    disabled_services: list[str] = field(default_factory=list)  # "Microsoft 365"
    browser: bool = True  # Playwright browser automation in jobs


@dataclass
class JarvisConfig:
    assistant: AssistantConfig = field(default_factory=AssistantConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    ears: EarsConfig = field(default_factory=EarsConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    orb: OrbConfig = field(default_factory=OrbConfig)
    actions: ActionsConfig = field(default_factory=ActionsConfig)
    hands: HandsConfig = field(default_factory=HandsConfig)


DEFAULT_TOML = """\
# Configuration de l'orbe Jarvis — le seul endroit où se règle l'assistant.
# Lancer : jarvis orb

[assistant]
name = "Jarvis"
language = "fr"
user_name = "Monsieur"    # comment il t'appelle

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
barge_in = true           # dire le mot d'activation pendant qu'il parle l'interrompt
barge_in_threshold = 0.75 # plus strict car le micro entend aussi sa voix

[voice]
# voix locales : "piper", "say" (macOS)
# voix en ligne (le texte part sur un serveur) : "edge" (Microsoft), "elevenlabs"
engine = "piper"
local_only = true         # true = jamais de voix en ligne, même si choisie
# voix à écouter : ~/Jarvis/voix-reelles/index.txt
edge_voice = "fr-FR-RemyMultilingualNeural"
pitch = "+0Hz"            # edge : hauteur, ex. "-2Hz"
# clé ElevenLabs : ~/.openjarvis/elevenlabs.key ou $ELEVENLABS_API_KEY
elevenlabs_voice = ""
elevenlabs_model = "eleven_flash_v2_5"
online_effect = "none"    # effet pour edge/elevenlabs (none = voix naturelle)
piper_voice = "fr_FR-tom-medium"
say_voice = "Thomas"
speed = 1.15              # débit de parole (1.0 = normal)
# effets piper/say : none, jarvis, jarvis_deep, jarvis_air, helmet
# (échantillons à écouter : ~/Jarvis/echantillons-voix)
effect = "jarvis"

[orb]
size = 220
position = "bottom-right" # top-left, top-right, bottom-left, bottom-right, center
color = "#4fc3ff"

[actions]                 # réflexes : actions locales instantanées
enabled = true
city = "Bruxelles"        # ville par défaut pour la météo
music_app = "Music"       # "Music" ou "Spotify"
web_search = true

[hands]                   # tâches longues déléguées à Claude Code
enabled = true
workspace = "~/Jarvis"    # dossier où il construit
model = ""                # vide = modèle par défaut de Claude Code
confirm = false           # true = demande ton accord avant de lancer une tâche
# plafond par tâche (en équivalent API) ; avec un abonnement Claude, rien
# n'est facturé : ça évite juste qu'une tâche épuise ton quota d'utilisation
max_budget_usd = 2.0
max_parallel = 2
# modèle pour agenda/mails/Drive/Notion : vide = sonnet, "haiku" = plus léger
# pour ton quota d'abonnement
services_model = ""
services_budget_usd = 0.5
extra_domains = []        # domaines réseau en plus, ex. ["api.monservice.com"]
disabled_services = []    # services à ne pas utiliser, ex. ["Microsoft 365"]
browser = true            # navigateur automatisé dans les tâches
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
