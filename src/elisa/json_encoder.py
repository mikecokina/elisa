"""Shared JSON encoding for ELISA data."""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

import numpy as np
from astropy.units import Quantity, UnitBase


class ElisaJSONEncoder(json.JSONEncoder):
    """Encode units, NumPy values and dates without changing physical values.

    Units use the same string representation as fitting result metadata.
    Quantities and other unsupported objects retain the standard TypeError.
    """

    def default(self, obj: Any) -> Any:  # noqa: PLR0911
        """Convert supported objects to JSON-compatible values.

        :param obj: Object that the standard JSON encoder cannot encode.
        :returns: Unit string, native scalar, array contents or ISO 8601 string.
        :raises TypeError: If the object has no supported JSON representation.
        """
        if isinstance(obj, UnitBase):
            return str(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        # Quantity inherits from ndarray, but needs an explicit application contract.
        if isinstance(obj, np.ndarray) and not isinstance(obj, Quantity):
            return obj.tolist()
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)
