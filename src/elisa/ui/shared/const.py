from __future__ import annotations

import os


def _read_positive_int_from_env(var_name: str, default: int) -> int:
    """Read a positive integer from environment with fallback to default.

    :param var_name: Environment variable name.
    :type var_name: str
    :param default: Fallback default value.
    :type default: int
    :returns: Positive integer read from environment or *default*.
    :rtype: int
    """
    raw_value = os.getenv(var_name)
    if raw_value is None:
        return default

    text = raw_value.strip()
    if text == "":
        return default

    try:
        parsed = int(text)
    except ValueError:
        return default

    if parsed <= 0:
        return default

    return parsed


# Supported atmosphere model identifiers recognized by ELISa.
ATMOSPHERE_CHOICES: tuple[str, ...] = ("bb", "ck04", "k93")

# RV computation methods supported by ELISa Observer.rv()
RV_METHODS: tuple[str, ...] = ("kinematic", "radiometric")

# Public constants
MAX_SPOTS: int = _read_positive_int_from_env("ELISA_UI_MAX_SPOTS", 3)
MAX_PULSE_MODES: int = _read_positive_int_from_env("ELISA_UI_MAX_PULSE_MODES", 3)
