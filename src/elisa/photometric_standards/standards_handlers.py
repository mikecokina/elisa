from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from elisa import settings
from elisa.base.error import raise_elisa_unexpected_error

if TYPE_CHECKING:
    from typing import Literal

    from elisa.types import ZeroPointType


def load_standard(
        system: Literal["vega", "ab", "st"],
        *,
        source_path: str | Path | None = None,
) -> ZeroPointType:
    """Load zero points for magnitude calculations.

    Load photometric zero-point fluxes for the requested ``system`` from the
    package data directory. The data file is expected at
    ``<settings.DATA_PATH>/zero_points/{system}.json`` and must contain a
    JSON object mapping filter names (strings) to numeric zero-point
    fluxes.

    :param system: Photometric zero-point system; allowed values are
        ``vega``, ``ab``, or ``st``.
    :type system: Literal['vega','ab','st']
    :param source_path: Optional resolved resource selected by a cache owner.
    :returns: Mapping from filter name to zero-point flux value (flux
        units depend on the zero-points file contents).
    :rtype: dict[str, elisa.types.Float]
    :raises ElisaUnexpectedError: If the packaged calibration file is missing or invalid JSON.
    """
    file_path = (
        Path(settings.DATA_PATH) / "zero_points" / f"{system.lower()}.json"
        if source_path is None else Path(source_path)
    )

    # These calibration resources ship with ELISA. Their existence and JSON syntax
    # are package contracts, independent of candidate parameters during fitting.
    try:
        with file_path.open("r", encoding="utf-8") as fl:
            return json.load(fl)
    except FileNotFoundError as exc:
        msg = f"Zero points file for system {system!r} not found at {file_path}"
        return raise_elisa_unexpected_error(msg, exc)
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON in zero points file for system {system!r} at {file_path}: {exc}"
        return raise_elisa_unexpected_error(msg, exc)
