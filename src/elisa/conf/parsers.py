from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from elisa.types import Number

_NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_TUPLE_RE = re.compile(rf"^\s*\(\s*({_NUM})\s*,\s*({_NUM})\s*\)\s*$")


def _parse_number(token: str) -> Number:
    """Parse a string token as an integer or float.

    Preserves integer type if possible, otherwise returns float.

    :param token: String token representing a number.
    :type token: str
    :returns: Parsed number as int or float.
    :rtype: int | float
    """
    if "." in token or "e" in token.lower():
        return float(token)
    return int(token)


def parse_tuple_interval(
        raw: str,
        *,
        name: str,
        require_ordered: bool = True,
) -> tuple[Number, Number]:
    """Parse a string representing a tuple of two numbers.

    Accepts only the format '(a, b)' where a and b are numbers. Preserves int vs float.

    Examples::

        (5000, 6000)   -> (int, int)
        (5000.0, 6000) -> (float, int)
        (5e3, 6e3)     -> (float, float)

    :param raw: String to parse, expected in the form '(a, b)'.
    :type raw: str
    :param name: Name of the parameter (for error messages).
    :type name: str
    :param require_ordered: If True, require that the first value is not greater than the second.
    :type require_ordered: bool
    :returns: Tuple of two numbers (int or float).
    :rtype: tuple[int | float, int | float]
    :raises ValueError: If the input is missing, not in the correct format, or not ordered when required.
    """
    if raw is None:
        msg = f"{name}: missing value"
        raise ValueError(msg)

    m = _TUPLE_RE.match(raw)
    if not m:
        msg = f"{name}: invalid format {raw!r}. Expected '(a, b)' with numbers."
        raise ValueError(msg)

    a_token, b_token = m.group(1), m.group(2)
    low = _parse_number(a_token)
    high = _parse_number(b_token)

    if require_ordered and low > high:
        msg = f"{name}: low > high in {raw!r} ({low} > {high})"
        raise ValueError(msg)

    return low, high
