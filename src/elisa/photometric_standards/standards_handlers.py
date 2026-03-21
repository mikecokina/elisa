from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from elisa import settings

if TYPE_CHECKING:
    from typing import Literal

    from elisa.types import ZeroPointType


def load_standard(system: Literal["vega", "ab", "st"]) -> ZeroPointType:
    """Load zero points for magnitude calculations.

    Load photometric zero-point fluxes for the requested ``system`` from the
    package data directory. The data file is expected at
    ``<settings.DATA_PATH>/zero_points/{system}.json`` and must contain a
    JSON object mapping filter names (strings) to numeric zero-point
    fluxes.

    :param system: Photometric zero-point system; allowed values are
        ``vega``, ``ab``, or ``st``.
    :type system: Literal['vega','ab','st']
    :returns: Mapping from filter name to zero-point flux value (flux
        units depend on the zero-points file contents).
    :rtype: dict[str, elisa.types.Float]
    :raises FileNotFoundError: If the expected zero-points JSON file is
        missing.
    :raises json.JSONDecodeError: If the zero-points file is not valid JSON.
    """
    data_dir = Path(settings.DATA_PATH) / "zero_points"
    file_path = data_dir / f"{system.lower()}.json"

    if not file_path.is_file():
        msg = f"Zero points file for system {system!r} not found at {file_path}"
        raise FileNotFoundError(msg)

    with file_path.open("r", encoding="utf-8") as fl:
        return json.load(fl)
