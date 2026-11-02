import os
import sys
import tempfile


def set_astropy_units():

    if "astropy.units" not in sys.modules:
        from astropy import physical_constants
        from astropy.utils.state import ScienceState

        const_value = "astropyconst20"

        if hasattr(physical_constants, "set"):
            class PhysicalConstants(physical_constants):
                _value = const_value

                _versions = dict(codata2018="codata2018", codata2014="codata2014",
                                 codata2010="codata2010", astropyconst40="codata2018",
                                 astropyconst20="codata2014", astropyconst13="codata2010")

                ScienceState._value = _value
                ScienceState._versions = _versions

                @classmethod
                def set(cls, value):
                    return ScienceState.set(value)

            sys.modules["astropy"].physical_constants = PhysicalConstants
            physical_constants = PhysicalConstants

        physical_constants.set(const_value)


# Private worker/test INI. Do not replace the user-selected ELISA_CONFIG on import.
_test_config_directory = tempfile.TemporaryDirectory(prefix="elisa-tests-")
CONFIG_FILE = os.path.join(_test_config_directory.name, "elisa.ini")

DEFAULT_TEST_CONFIG = "[computational]\nmesh_generator = trapezoidal\nmax_discretization_factor = 12\n"

with open(CONFIG_FILE, "w") as f:
    f.write(DEFAULT_TEST_CONFIG)

set_astropy_units()


__all__ = ("set_astropy_units",)
