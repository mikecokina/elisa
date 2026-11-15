from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import contextlib


def set_astropy_units() -> None:

    if "astropy.units" not in sys.modules:
        from astropy import physical_constants
        from astropy.utils.state import ScienceState

        const_value = "astropyconst20"

        if hasattr(physical_constants, "set"):

            class PhysicalConstants(physical_constants):
                _value = const_value

                _versions: ClassVar[dict[str, str]] = {
                    "codata2018": "codata2018",
                    "codata2014": "codata2014",
                    "codata2010": "codata2010",
                    "astropyconst40": "codata2018",
                    "astropyconst20": "codata2014",
                    "astropyconst13": "codata2010",
                }

                ScienceState._value = _value
                ScienceState._versions = _versions

                @classmethod
                def set(cls, value: str) -> contextlib.AbstractContextManager[None]:
                    return ScienceState.set(value)

            sys.modules["astropy"].physical_constants = PhysicalConstants
            physical_constants = PhysicalConstants

        physical_constants.set(const_value)


# Private worker/test INI. Do not replace the user-selected ELISA_CONFIG on import.
_test_config_directory = tempfile.TemporaryDirectory(prefix="elisa-tests-")
CONFIG_FILE = str(Path(_test_config_directory.name).joinpath("elisa.ini"))

DEFAULT_TEST_CONFIG = "[computational]\nmesh_generator = trapezoidal\nmax_discretization_factor = 12\n"

with Path(CONFIG_FILE).open("w") as f:
    f.write(DEFAULT_TEST_CONFIG)

set_astropy_units()


__all__ = ("set_astropy_units",)
