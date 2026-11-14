"""JSON-driven built-in profiles; custom files use standard dictConfig semantics."""
from __future__ import annotations

import json
import logging
import logging.config
import sys
from pathlib import Path
from typing import Any, TextIO

from elisa.base.error import raise_elisa_settings_error
from elisa.conf.enums import LoggingProfile

_console_handler: logging.StreamHandler | None = None


def configure_logging(selector: str) -> None:
    """Apply a built-in policy without disturbing host handlers, or custom JSON.

    Built-ins reuse one stdout handler and touch only explicitly declared loggers.
    Custom files use dictConfig and can replace/close handlers globally.

    :param selector: ``default``, ``fit``, or a trusted JSON configuration path.
    :raises ElisaSettingsError: If the selector is unknown or the file does not exist.
    """
    try:
        _configure_logging(selector)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise_elisa_settings_error(f"Invalid logging configuration {selector!r}: {exc}", exc)


def _configure_logging(selector: str) -> None:
    """Apply the selected policy after entering the settings error boundary."""
    global _console_handler  # noqa: PLW0603
    builtin = selector in tuple(profile.value for profile in LoggingProfile)
    path = Path(__file__).parent / "logging_schemas" / f"{selector}.json" if builtin else Path(selector)
    if not path.is_file():
        raise_elisa_settings_error(f"Unknown logging profile or missing JSON configuration file: {selector}")
    config = json.loads(path.read_text())
    if not builtin:
        # Standard dictConfig may close our console too, even on failure. Do not
        # reuse it afterward. No custom resources or logger state are retained.
        if _console_handler is not None:
            # dictConfig can leave closed handlers on omitted named loggers.
            # Detach only our console from the finite built-in logger list.
            default_path = Path(__file__).parent / "logging_schemas" / f"{LoggingProfile.DEFAULT.value}.json"
            default = json.loads(default_path.read_text())
            for name in ("", *default["loggers"]):
                logging.getLogger(name).removeHandler(_console_handler)
        _console_handler = None
        logging.config.dictConfig(config)
        return

    console = config["handlers"]["console"]
    if _console_handler is None:
        _console_handler = logging.StreamHandler(sys.stdout)
    elif getattr(_console_handler.stream, "closed", False):
        # A host may replace/close stdout. Retarget the same standard handler.
        _console_handler.stream = sys.stdout
    _console_handler.setLevel(console["level"])
    formatter = config["formatters"][console["formatter"]]
    _console_handler.setFormatter(logging.Formatter(formatter["format"], datefmt=formatter.get("datefmt")))
    for name, policy in [("", config["root"]), *config["loggers"].items()]:
        _apply_policy(logging.getLogger(name), policy, _console_handler)


def _apply_policy(logger: logging.Logger, policy: dict[str, Any], console: logging.StreamHandler) -> None:
    """Apply a complete built-in logger rule, preserving all external handlers.

    :param logger: Explicitly declared logger (including root).
    :param policy: Its rule from the selected built-in JSON.
    :param console: ELISA's single reusable built-in handler.
    """
    logger.setLevel(policy["level"])
    logger.propagate = policy.get("propagate", True)
    logger.disabled = False
    if "console" in policy.get("handlers", []):
        logger.addHandler(console)
    else:
        logger.removeHandler(console)


def restore_console_stream(captured: TextIO, restored: TextIO) -> None:
    """Release a temporary capture stream without changing the logging policy.

    Capture owners call this after restoring stdout. Only ELISA's console is
    retargeted, and only if it still points at that capture's stream.

    :param captured: Stream whose capture lifetime has ended.
    :param restored: Live output stream restored by the capture owner.
    """
    if _console_handler is not None and _console_handler.stream is captured:
        _console_handler.stream = restored
