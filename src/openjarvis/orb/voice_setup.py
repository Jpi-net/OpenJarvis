"""One-command setup of the local Chatterbox voice.

    python -m openjarvis.orb.voice_setup            # install everything
    python -m openjarvis.orb.voice_setup --check    # say what is missing

Steps (each one is skipped when already done):

1. a private venv ``~/.openjarvis/orb/chatterbox-venv`` (Python 3.11) with
   pinned ``mlx-audio`` (Chatterbox ported to MLX, the Apple GPU). Its torch-
   free deps never touch the project venv;
2. the model weights, downloaded once from Hugging Face into the usual cache
   (``mlx-community/chatterbox-multilingual-v3``, MIT, ~2.7 GB, and its
   speech tokenizer). Synthesis then runs offline;
3. reference voices in ``~/.openjarvis/orb/voices/`` (Chatterbox clones the
   timbre of a ~10 s clip). Only material we may use:

   - ``grave``, ``pose``, ``clair``: three male French readers of LibriVox
     audiobooks (public-domain recordings), taken from the Multilingual
     LibriSpeech corpus (CC BY 4.0, Pratap et al. 2020);
   - ``tom``, ``pierre``: clips made by the local piper voices (training data
     AGPL-3.0 for tom, CC BY-SA 4.0 for upmc/pierre).

   Your own voice (or a consenting person's): put a WAV of 8-12 s of calm,
   clean speech at ``~/.openjarvis/orb/voices/reference.wav``; it is used
   whenever ``chatterbox_reference`` is empty.

4. a check: the worker starts offline and says a sentence.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any

from openjarvis.orb.config import DATA_DIR

logger = logging.getLogger("openjarvis.orb.voice_setup")

CHATTERBOX_VENV = DATA_DIR / "chatterbox-venv"
VOICES_DIR = DATA_DIR / "voices"
USER_REFERENCE = VOICES_DIR / "reference.wav"
DEFAULT_REFERENCE = "grave"

PACKAGES = ("mlx-audio==0.6.6", "truststore")
# mlx-audio loads the tokenizer by name (branch "main"), so both repos are
# fetched at "main" (tested: model 03565773, tokenizer e0c9886f).
MODEL_REPO = "mlx-community/chatterbox-multilingual-v3"
TOKENIZER_REPO = "mlx-community/S3TokenizerV2"

_MLS_URL = (
    "https://huggingface.co/datasets/facebook/multilingual_librispeech/resolve/"
    "2e83e61823b4c47dcbcb1980bb88601274127609/french/1_hours-00000-of-00001.parquet"
)
# name -> (MLS utterance id, description)
MLS_REFERENCES: dict[str, tuple[str, str]] = {
    "grave": ("9242_8734_000355", "voix d'homme grave et posée (MLS 9242)"),
    "pose": ("5526_3078_000006", "voix d'homme medium, très nette (MLS 5526)"),
    "clair": ("3344_2567_000000", "voix d'homme claire (MLS 3344)"),
}
# name -> (piper voice spec, description)
PIPER_REFERENCES: dict[str, tuple[str, str]] = {
    "tom": ("fr_FR-tom-medium", "timbre de la voix piper tom"),
    "pierre": ("fr_FR-upmc-medium#pierre", "timbre de la voix piper pierre"),
}
_PIPER_TEXT = (
    "Bonsoir. Je me permets de vous rappeler que le dîner est servi à vingt "
    "heures. J'ai pris la liberté de préparer vos documents pour demain matin, "
    "et la voiture vous attendra devant l'entrée."
)
_REFERENCE_S = 9.0  # Chatterbox uses up to 10 s (6 s for the token prompt)
ATTRIBUTION = """\
Voix de référence pour Chatterbox (clonage de timbre, 100 % local).

grave.wav, pose.wav, clair.wav : extraits de lecteurs bénévoles LibriVox
(enregistrements dans le domaine public), via le corpus Multilingual
LibriSpeech (MLS), licence CC BY 4.0 — V. Pratap et al., "MLS: A Large-Scale
Multilingual Dataset for Speech Research", Interspeech 2020.
Locuteurs MLS : grave = 9242, pose = 5526, clair = 3344.

tom.wav, pierre.wav : phrases synthétisées par les voix piper locales
(fr_FR-tom-medium : données AGPL-3.0 ; fr_FR-upmc-medium/pierre : données
CC BY-SA 4.0, https://github.com/marytts/upmc-pierre-data).

reference.wav (si présent) : votre propre fichier ; il est utilisé quand
chatterbox_reference est vide. N'y mettez que votre voix ou celle d'une
personne consentante.
"""


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------


def worker_python(venv: Path = CHATTERBOX_VENV) -> Path:
    return venv / "bin" / "python"


def reference_path(name: str) -> Path:
    return VOICES_DIR / f"{name}.wav"


def venv_ready(venv: Path = CHATTERBOX_VENV) -> bool:
    py = worker_python(venv)
    if not py.exists():
        return False
    probe = "import mlx_audio.tts.models.chatterbox, huggingface_hub"
    result = subprocess.run([str(py), "-I", "-c", probe], capture_output=True)
    return result.returncode == 0


def models_ready(venv: Path = CHATTERBOX_VENV) -> bool:
    if not worker_python(venv).exists():
        return False
    code = (
        "from huggingface_hub import snapshot_download as s\n"
        f"s({MODEL_REPO!r}, local_files_only=True)\n"
        f"s({TOKENIZER_REPO!r}, local_files_only=True)\n"
    )
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    result = subprocess.run(
        [str(worker_python(venv)), "-I", "-c", code], capture_output=True, env=env
    )
    return result.returncode == 0


# --------------------------------------------------------------------------
# 1. venv
# --------------------------------------------------------------------------


def install_venv(venv: Path = CHATTERBOX_VENV) -> None:
    uv = shutil.which("uv")
    py = worker_python(venv)
    if not py.exists():
        logger.info("Creating %s", venv)
        if uv:
            _run([uv, "venv", str(venv), "--python", "3.11"])
        else:
            base = shutil.which("python3.11") or sys.executable
            _run([base, "-m", "venv", str(venv)])
    logger.info("Installing %s", ", ".join(PACKAGES))
    if uv:
        _run([uv, "pip", "install", "--python", str(py), *PACKAGES])
    else:
        _run([str(py), "-m", "pip", "install", "--upgrade", "pip"])
        _run([str(py), "-m", "pip", "install", *PACKAGES])


# --------------------------------------------------------------------------
# 2. weights
# --------------------------------------------------------------------------

_DOWNLOAD = """\
import sys, truststore
truststore.inject_into_ssl()
from huggingface_hub import snapshot_download
for repo in sys.argv[1:]:
    print(snapshot_download(repo), flush=True)
"""


def download_models(venv: Path = CHATTERBOX_VENV) -> None:
    cmd = [
        str(worker_python(venv)),
        "-I",
        "-c",
        _DOWNLOAD,
        MODEL_REPO,
        TOKENIZER_REPO,
    ]
    env = {**os.environ}
    env.pop("HF_HUB_OFFLINE", None)
    logger.info("Downloading %s (~2.7 GB, once)", MODEL_REPO)
    if subprocess.run(cmd, env=env).returncode == 0:
        return
    logger.warning("Download failed; retrying without the xet transfer backend")
    env["HF_HUB_DISABLE_XET"] = "1"
    _run(cmd, env=env)


# --------------------------------------------------------------------------
# 3. reference voices
# --------------------------------------------------------------------------


def _write_wav(path: Path, audio: Any, sr: int) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    tmp = path.with_suffix(".part")
    with wave.open(str(tmp), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    os.replace(tmp, path)


def cut_reference(audio: Any, sr: int, seconds: float = _REFERENCE_S) -> Any:
    """Keep ~``seconds`` of speech, cut at the quietest 20 ms near the end."""
    import numpy as np

    if len(audio) <= int(seconds * sr):
        return audio
    lo, hi = int((seconds - 1.5) * sr), min(len(audio), int((seconds + 0.8) * sr))
    block = sr // 50
    zone = audio[lo:hi]
    n = len(zone) // block
    rms = np.sqrt((zone[: n * block].reshape(n, block) ** 2).mean(axis=1))
    cut = lo + int(np.argmin(rms)) * block + block // 2
    out = audio[:cut].astype(np.float32).copy()
    fade = int(0.02 * sr)
    out[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    peak = float(np.abs(out).max()) or 1.0
    return out * min(4.0, 0.9 / peak)  # consistent level, no clipping


def build_mls_references(force: bool = False) -> list[Path]:
    todo = {
        k: v
        for k, v in MLS_REFERENCES.items()
        if force or not reference_path(k).exists()
    }
    if not todo:
        return []
    import httpx
    import pyarrow.parquet as pq
    import soundfile as sf
    import truststore

    truststore.inject_into_ssl()
    logger.info("Downloading the MLS French sample (15 MB) for the reference voices")
    data = httpx.get(_MLS_URL, follow_redirects=True, timeout=120.0)
    data.raise_for_status()
    table = pq.read_table(io.BytesIO(data.content), columns=["id", "audio"])
    rows = {r["id"]: r for r in table.to_pylist()}
    made = []
    for name, (utt, _desc) in todo.items():
        row = rows.get(utt)
        if row is None:
            logger.warning("MLS utterance %s not found; skipping %s", utt, name)
            continue
        audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        _write_wav(reference_path(name), cut_reference(audio, sr), sr)
        made.append(reference_path(name))
    return made


def build_piper_references(force: bool = False) -> list[Path]:
    import numpy as np

    from openjarvis.orb import voice as V

    made = []
    for name, (spec, _desc) in PIPER_REFERENCES.items():
        if reference_path(name).exists() and not force:
            continue
        try:
            from piper import PiperVoice, SynthesisConfig

            voice_name, speaker = V.split_voice_name(spec)
            model = V.PIPER_DIR / f"{voice_name}.onnx"
            config = V.PIPER_DIR / f"{voice_name}.onnx.json"
            base = f"{V._HF_BASE}/{V._voice_url_path(voice_name)}"
            if not config.exists():
                V._download(f"{base}.onnx.json?download=true", config)
            if not model.exists():
                V._download(f"{base}.onnx?download=true", model)
            piper = PiperVoice.load(model, config_path=config)
            pcfg = piper.config
            sid = V.resolve_speaker(
                speaker, dict(pcfg.speaker_id_map or {}), int(pcfg.num_speakers or 1)
            )
            syn = SynthesisConfig(speaker_id=sid, length_scale=1.05)
            audio = np.concatenate(
                [c.audio_float_array for c in piper.synthesize(_PIPER_TEXT, syn)]
            )
            sr = int(pcfg.sample_rate)
            _write_wav(reference_path(name), cut_reference(audio, sr, 10.0), sr)
            made.append(reference_path(name))
        except Exception as exc:
            logger.warning("Piper reference %s not built (%s)", name, exc)
    return made


def build_references(force: bool = False) -> list[Path]:
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []
    try:
        made += build_mls_references(force)
    except Exception as exc:
        logger.warning("LibriVox/MLS references not built (%s)", exc)
    made += build_piper_references(force)
    (VOICES_DIR / "ATTRIBUTION.txt").write_text(ATTRIBUTION, encoding="utf-8")
    return made


# --------------------------------------------------------------------------
# 4. check
# --------------------------------------------------------------------------


def check_synthesis() -> float:
    """Start the worker (offline) and time one sentence; returns its RTF."""
    from openjarvis.orb import voice as V

    worker = V.ChatterboxWorker(V.chatterbox_command())
    try:
        info = worker.start(
            {"model": MODEL_REPO},
            {"reference": str(reference_path(DEFAULT_REFERENCE)), "exaggeration": 0.4},
        )
        logger.info("Model loaded in %.1f s", float(info.get("load_s", 0.0)))
        t0 = time.perf_counter()
        audio = worker.say(
            "Bonjour Monsieur, tous les systèmes sont opérationnels.", {}
        )
        took = time.perf_counter() - t0
    finally:
        worker.close()
    seconds = len(audio) / V.CHATTERBOX_SAMPLE_RATE
    rtf = took / seconds if seconds else float("inf")
    logger.info(
        "Test sentence: %.2f s of audio in %.2f s (RTF %.2f)", seconds, took, rtf
    )
    return rtf


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    logger.debug("$ %s", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def status() -> dict[str, bool]:
    refs = all(reference_path(n).exists() for n in (*MLS_REFERENCES, *PIPER_REFERENCES))
    return {
        "venv": venv_ready(),
        "models": models_ready(),
        "references": refs,
        "user_reference": USER_REFERENCE.exists(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m openjarvis.orb.voice_setup",
        description="Install the local Chatterbox voice (offline synthesis).",
    )
    parser.add_argument(
        "--check", action="store_true", help="only report what is missing"
    )
    parser.add_argument(
        "--rebuild-references", action="store_true", help="rebuild the reference clips"
    )
    parser.add_argument(
        "--no-test", action="store_true", help="skip the synthesis test"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if sys.platform != "darwin":
        logger.error("The Chatterbox voice uses MLX: Apple Silicon Macs only.")
        return 1
    if args.check:
        for key, ok in status().items():
            logger.info("%-15s %s", key, "ok" if ok else "missing")
        return 0
    if not venv_ready():
        install_venv()
    if not models_ready():
        download_models()
    made = build_references(force=args.rebuild_references)
    for path in made:
        logger.info("Reference voice: %s", path)
    if not args.no_test:
        check_synthesis()
    logger.info(
        'Done. In ~/.openjarvis/jarvis.toml, [voice]: engine = "chatterbox" '
        "(voices: %s; your own: %s)",
        ", ".join((*MLS_REFERENCES, *PIPER_REFERENCES)),
        USER_REFERENCE,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
