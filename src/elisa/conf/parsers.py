# numeric token with optional decimal and exponent
import re
from typing import Union, Tuple

_NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_TUPLE_RE = re.compile(rf"^\s*\(\s*({_NUM})\s*,\s*({_NUM})\s*\)\s*$")


Number = Union[int, float]


def _parse_number(token: str) -> Number:
    """
    Preserve int if possible, otherwise float.
    """
    # If it contains '.' or exponent, it must be float
    if "." in token or "e" in token.lower():
        return float(token)
    return int(token)


def parse_tuple_interval(
        raw: str,
        *,
        name: str,
        require_ordered: bool = True,
) -> Tuple[Number, Number]:
    """
    Accept ONLY '(a, b)' where a and b are numbers.

    Preserves int vs float:
      (5000, 6000)   -> (int, int)
      (5000.0, 6000) -> (float, int)
      (5e3, 6e3)     -> (float, float)
    """
    if raw is None:
        raise ValueError(f"{name}: missing value")

    m = _TUPLE_RE.match(raw)
    if not m:
        raise ValueError(
            f"{name}: invalid format {raw!r}. Expected '(a, b)' with numbers."
        )

    a_token, b_token = m.group(1), m.group(2)

    low = _parse_number(a_token)
    high = _parse_number(b_token)

    if require_ordered and low > high:
        raise ValueError(
            f"{name}: low > high in {raw!r} ({low} > {high})"
        )

    return low, high
