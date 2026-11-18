"""``jarvis orb`` — voice-only assistant shown as a floating orb."""

from __future__ import annotations

import logging

import click


@click.command()
@click.option("--headless", is_flag=True, help="No orb window; voice only (debugging).")
@click.option(
    "--text",
    "text_input",
    is_flag=True,
    help="Type in the terminal instead of speaking (debugging).",
)
@click.option("-v", "--verbose", is_flag=True, help="Log details to the terminal.")
def orb(headless: bool, text_input: bool, verbose: bool) -> None:
    """Start the voice-only orb. Say "Jarvis" to wake it.

    All settings live in ~/.openjarvis/jarvis.toml (created on first run).
    """
    from openjarvis.orb.config import CONFIG_PATH, DATA_DIR, load_config

    try:
        cfg = load_config()
    except Exception as exc:  # invalid TOML, wrong types…
        raise click.ClickException(f"Invalid config {CONFIG_PATH}: {exc}") from exc

    # The root CLI sets the "openjarvis" logger to WARNING; the orb logs on its
    # own logger so its timings land in ~/.openjarvis/orb/orb.log.
    orb_log = logging.getLogger("openjarvis.orb")
    orb_log.setLevel(logging.INFO)
    orb_log.propagate = False
    orb_log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    handlers: list[logging.Handler] = [
        logging.FileHandler(DATA_DIR / "orb.log", encoding="utf-8")
    ]
    if verbose:
        handlers.append(logging.StreamHandler())
    for handler in handlers:
        handler.setFormatter(fmt)
        orb_log.addHandler(handler)

    try:
        from openjarvis.orb.app import OrbApp
    except ImportError as exc:
        raise click.ClickException(
            f"Missing orb dependencies ({exc.name}). Run: uv sync --extra orb"
        ) from exc

    click.echo(f"{cfg.assistant.name} démarre… (réglages : {CONFIG_PATH})")
    OrbApp(cfg, headless=headless, text_input=text_input).run()
