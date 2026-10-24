"""Shared value validators for ELISa UI parsing logic."""

from __future__ import annotations

import math
import sys
from typing import TYPE_CHECKING, cast

from elisa.ui.shared.utils import opt_float, opt_int

if TYPE_CHECKING:
    from typing import SupportsFloat, SupportsIndex

    from elisa.types import Float


def validated_float(
    value: object,
    *,
    name: str,
    lo: Float | None = None,
    hi: Float | None = None,
) -> float | None:
    """Parse *value* as an optional float and validate it against a range.

    :param value: Numeric value, string, or ``None``.
    :type value: object
    :param name: Parameter name used in error messages.
    :type name: str
    :param lo: Optional inclusive lower bound.
    :type lo: Float | None
    :param hi: Optional inclusive upper bound.
    :type hi: Float | None
    :returns: Parsed and validated float, or ``None`` if not supplied.
    :rtype: Float | None
    :raises ValueError: If the parsed value lies outside ``[lo, hi]``.
    """
    parsed = opt_float(value)
    if parsed is None:
        if value is not None and str(value).strip():
            raise ValueError(f"'{name}' must be a number.")
        return None
    if not math.isfinite(parsed):
        raise ValueError(f"'{name}' must be finite.")
    if lo is not None and parsed < lo:
        msg = f"'{name}' must be >= {lo}, got {parsed}."
        raise ValueError(msg)
    if hi is not None and parsed > hi:
        msg = f"'{name}' must be <= {hi}, got {parsed}."
        raise ValueError(msg)
    return parsed


def validated_positive_int(value: object, *, name: str) -> int | None:
    """Parse *value* as an optional positive integer.

    :param value: Numeric value, string, or ``None``.
    :type value: object
    :param name: Parameter name used in error messages.
    :type name: str
    :returns: Parsed positive integer, or ``None`` if not supplied.
    :rtype: Int | None
    :raises ValueError: If the parsed value is not a positive integer.
    """
    parsed = opt_int(value)
    if parsed is None:
        return None
    if parsed <= 0:
        msg = f"'{name}' must be a positive integer, got {parsed}."
        raise ValueError(msg)
    return parsed


def mcmc_integer(value: object, *, name: str, default: int, minimum: int = 1) -> int:
    """Parse an MCMC count without replacing zero or truncating fractions."""
    parsed = opt_int(value)
    parsed = default if parsed is None else parsed
    if parsed < minimum:
        raise ValueError(f"'{name}' must be >= {minimum}.")
    return parsed


def phase_sampling(params: dict[str, object]) -> tuple[float, float, float]:
    """Validate phase sampling before constructing a scientific model.

    A workload limit needs an explicit UI policy; this only rejects invalid
    or unrepresentable ranges, without allocating the sampling array.
    """
    values: list[float] = []
    for key, default in (("from_phase", -0.6), ("to_phase", 0.6), ("phase_step", 0.01)):
        raw = params.get(key)
        value = default if raw is None or raw == "" else float(cast("str | SupportsFloat | SupportsIndex", raw))
        if not math.isfinite(value):
            raise ValueError(f"'{key}' must be finite.")
        values.append(value)
    start, stop, step = values
    if stop <= start:
        raise ValueError("to_phase must be greater than from_phase.")
    if step <= 0 or start + step == start:
        raise ValueError("phase_step must be positive and advance the phase.")
    count = (stop - start) / step
    if not math.isfinite(count) or count > sys.maxsize:
        raise ValueError("Phase sampling range is not representable.")
    return start, stop, step
