"""Shared utility helpers for the ELISa Gradio UI.

Pure-Python helpers used across multiple tab-level compute modules so they
do not have to be duplicated.  No Gradio dependency - safe to import from
any layer.
"""

from __future__ import annotations

import math
import tempfile
from datetime import datetime
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from elisa.utc import UTC

if TYPE_CHECKING:
    from typing import SupportsFloat, SupportsIndex


def opt_float(value: object) -> float | None:
    """Convert *value* to a float or return ``None`` for absent/empty inputs.

    Accepts numeric types, string representations produced by ``gr.Textbox``
    or ``gr.Number``, and ``None``.  Empty strings and ``None`` both signal
    "not supplied" and return ``None``.  Non-parseable strings also return
    ``None`` so callers never need to guard against conversion errors.

    :param value: Numeric value, string representation, or ``None``.
    :type value: object
    :returns: Parsed float or ``None``.
    :rtype: Float | None
    """
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    try:
        return float(cast("str | SupportsFloat | SupportsIndex", value))
    except (TypeError, ValueError):
        return None


def opt_int(value: object) -> int | None:
    """Convert *value* to an int or return ``None`` for absent/empty inputs.

    Accepts numeric types, string representations produced by ``gr.Textbox``
    or ``gr.Number``, and ``None``. Empty strings and ``None`` both signal
    "not supplied" and return ``None``.

    :param value: Numeric value, string representation, or ``None``.
    :type value: object
    :returns: Parsed int or ``None``.
    :rtype: Int | None
    :raises ValueError: If *value* is a non-empty string that cannot be
        parsed as a finite integer without truncation.
    """
    if value is None:
        return None
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        try:
            parsed = float(stripped)
        except ValueError as exc:
            msg = f"Cannot parse '{stripped}' as a number."
            raise ValueError(msg) from exc
        value = parsed
    # Values arrive from untyped form dictionaries; float performs validation.
    # noinspection PyUnnecessaryCast
    parsed = float(cast("str | SupportsFloat | SupportsIndex", value))
    if not math.isfinite(parsed) or not parsed.is_integer():
        msg_0 = "Expected a finite integer."
        raise ValueError(msg_0)
    return int(parsed)


def result_temp_path(category: str, prefix: str) -> Path:
    """Return a timestamped temp-file path for a fit result JSON.

    The filename follows the pattern
    ``elisa_{category}_{prefix}_YYYY-MM-DD_HH-MM-SS_<uuid>.json``.

    :param category: Short type label embedded in the filename
        (e.g. ``"lc"`` or ``"rv"``).
    :type category: str
    :param prefix: Short method label embedded in the filename
        (e.g. ``"lsqrt"`` or ``"mcmc"``).
    :type prefix: str
    :returns: Path inside the system temp directory.
    :rtype: pathlib.Path
    """
    return export_temp_path(f"{category}_{prefix}", ".json")


def collect_param_values(
    param_keys: tuple[str, ...],
    values: tuple[object, ...],
    offset: int,
) -> dict[str, object]:
    """Slice *values* starting at *offset* into a named dict.

    :param param_keys: Ordered key names.
    :type param_keys: tuple[str, ...]
    :param values: Full flat value tuple from Gradio callbacks.
    :type values: tuple[object, ...]
    :param offset: Start index within *values*.
    :type offset: int
    :returns: Dict mapping each key to its corresponding value.
    :rtype: dict[str, object]
    """
    return dict(zip(param_keys, values[offset : offset + len(param_keys)], strict=True))


def export_temp_path(prefix: str, suffix: str) -> Path:
    """Return a unique export path, retaining a readable timestamp and prefix."""
    ts = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
    return Path(tempfile.gettempdir()) / f"elisa_{prefix}_{ts}_{uuid4().hex}{suffix}"
