"""Shared utility helpers for the ELISa Gradio UI.

Pure-Python helpers used across multiple tab-level compute modules so they
do not have to be duplicated.  No Gradio dependency - safe to import from
any layer.
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from elisa.utc import UTC

if TYPE_CHECKING:
    from elisa.types import Float


def opt_float(value: object) -> Float | None:
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
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def result_temp_path(category: str, prefix: str) -> Path:
    """Return a timestamped temp-file path for a fit result JSON.

    The filename follows the pattern
    ``elisa_{category}_{prefix}_YYYY-MM-DD_HH-MM-SS.json``.

    :param category: Short type label embedded in the filename
        (e.g. ``"lc"`` or ``"rv"``).
    :type category: str
    :param prefix: Short method label embedded in the filename
        (e.g. ``"lsqrt"`` or ``"mcmc"``).
    :type prefix: str
    :returns: Path inside the system temp directory.
    :rtype: pathlib.Path
    """
    ts = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
    return Path(tempfile.gettempdir()) / f"elisa_{category}_{prefix}_{ts}.json"


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
