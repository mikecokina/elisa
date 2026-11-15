"""Shared contracts and candidate geometry checks for binary surface builders."""

from __future__ import annotations

from math import isfinite
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np

from elisa.base.error import raise_elisa_numeric_error, raise_elisa_unexpected_error

if TYPE_CHECKING:
    from elisa.types import Float


def validate_component_distance(components_distance: Float) -> None:
    """Require a supplied real scalar representing a finite positive separation.

    The container resolves optional inputs before calling surface builders.
    Missing/non-scalar data therefore violate the internal calling contract.
    Unusable numeric values are defensively treated as candidate geometry
    failures, even if an earlier model/orbit validation should have caught them.
    """
    if components_distance is None:
        raise_elisa_unexpected_error("Surface construction requires a resolved component distance; received None.")
    if isinstance(components_distance, (bool, np.bool_)) or not isinstance(components_distance, Real):
        msg = (
            "Surface construction requires a real scalar component distance; "
            f"received {type(components_distance).__name__}."
        )
        raise_elisa_unexpected_error(msg)
    try:
        finite = isfinite(components_distance)
    except OverflowError as exc:
        return raise_elisa_numeric_error("Component distance exceeds the representable numeric range.", exc)
    if not finite or components_distance <= 0:
        msg = f"Component distance must be finite and strictly positive; received {components_distance!r}."
        raise_elisa_numeric_error(msg)
    return None
