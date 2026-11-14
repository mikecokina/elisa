"""Builtin error helpers local to the standalone PYPEX geometry code."""
from typing import NoReturn


def raise_value_error(message: str) -> NoReturn:
    """Raise a builtin ValueError without depending on ELISA error categories."""
    raise ValueError(message)
