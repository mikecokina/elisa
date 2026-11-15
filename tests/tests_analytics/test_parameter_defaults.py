# The tests package selects Astropy constants before test-module imports.
from numpy.testing import assert_array_equal

from elisa import units as u
from elisa.analytics.params import conf
from tests import set_astropy_units
from tests.utils import ElisaTestCase

set_astropy_units()

TOL = 1e-5


class ConfTestCase(ElisaTestCase):
    @staticmethod
    def test_DEFAULT_FLOAT_UNITS() -> None:
        # this is important, do not mess up default units in analytics
        expected = {
            "inclination": u.deg,
            "eccentricity": u.dimensionless_unscaled,
            "argument_of_periastron": u.deg,
            "gamma": u.VELOCITY_UNIT,
            "mass": u.solMass,
            "t_eff": u.TEMPERATURE_UNIT,
            "metallicity": u.dimensionless_unscaled,
            "surface_potential": u.dimensionless_unscaled,
            "albedo": u.dimensionless_unscaled,
            "gravity_darkening": u.dimensionless_unscaled,
            "synchronicity": u.dimensionless_unscaled,
            "mass_ratio": u.dimensionless_unscaled,
            "semi_major_axis": u.solRad,
            "asini": u.solRad,
            "period": u.PERIOD_UNIT,
            "primary_minimum_time": u.PERIOD_UNIT,
            "additional_light": u.dimensionless_unscaled,
            "phase_shift": u.dimensionless_unscaled,
            # SPOTS
            "latitude": u.deg,
            "longitude": u.deg,
            "angular_radius": u.deg,
            "temperature_factor": u.dimensionless_unscaled,
            # PULSATIONS
            "l": u.dimensionless_unscaled,
            "m": u.dimensionless_unscaled,
            "amplitude": u.VELOCITY_UNIT,
            "frequency": u.dimensionless_unscaled / u.d,
            "start_phase": u.rad,
            "mode_axis_theta": u.deg,
            "mode_axis_phi": u.deg,
            # NUISANCE
            "ln_f": None,
        }
        # in python3.6 >= order is maintain
        assert_array_equal(list(conf.DEFAULT_FLOAT_UNITS.values()), list(expected.values()))
        assert_array_equal(list(conf.DEFAULT_FLOAT_UNITS.keys()), list(expected.keys()))

    @staticmethod
    def test_COMPOSITE_FLAT_PARAMS() -> None:
        assert_array_equal(conf.COMPOSITE_FLAT_PARAMS, ["spot", "pulsation"])

    @staticmethod
    def test_DEFAULT_FLOAT_ANGULAR_UNIT() -> None:
        assert conf.DEFAULT_FLOAT_ANGULAR_UNIT, u.deg

    @staticmethod
    def test_DEFAULT_FLOAT_MASS_UNIT() -> None:
        assert conf.DEFAULT_FLOAT_MASS_UNIT, u.solMass
