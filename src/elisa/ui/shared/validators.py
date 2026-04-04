"""Shared value validators for ELISa UI parsing logic."""

from __future__ import annotations

from typing import TYPE_CHECKING

from elisa.ui.shared.utils import opt_float, opt_int

if TYPE_CHECKING:
    from elisa.types import Float, Int


def validated_float(
    value: float | str | None,
    *,
    name: str,
    lo: Float | None = None,
    hi: Float | None = None,
) -> Float | None:
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
        return None
    if lo is not None and parsed < lo:
        msg = f"'{name}' must be >= {lo}, got {parsed}."
        raise ValueError(msg)
    if hi is not None and parsed > hi:
        msg = f"'{name}' must be <= {hi}, got {parsed}."
        raise ValueError(msg)
    return parsed


def validated_positive_int(value: float | str | None, *, name: str) -> Int | None:
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

