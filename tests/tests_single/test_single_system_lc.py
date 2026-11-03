# The tests package selects Astropy constants before test-module imports.
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from elisa import settings
from elisa import umpy as up
from elisa import units as u
from elisa.observer.observer import Observer
from tests import set_astropy_units
from tests.utils import SOLAR_MODEL, ElisaTestCase, load_light_curve, normalize_lc_for_unittests, prepare_single_system

set_astropy_units()

TOL = 5e-3


@pytest.mark.requires_ld(passbands=("bolometric", "Generic.Bessell.V"))
@pytest.mark.requires_atmosphere(atlas="ck04")
class ComputeLightCurvesTestCase(ElisaTestCase):
    PARAMS: ClassVar[dict[str, dict[str, float | object]]] = {
        "solar": {
            "mass": 1.0,
            "t_eff": 5772 * u.K,
            "gravity_darkening": 0.32,
            "polar_log_g": 4.43775 * u.dex(u.cm / u.s**2),
            "gamma": 0.0,
            # "inclination": 82.5 * units.deg,
            "inclination": 90.0 * u.deg,
            "rotation_period": 25.38 * u.d,
        },
    }

    SPOTS_META: ClassVar[dict[str, list[dict[str, float | int]]]] = {
        "standard": [
            {"longitude": 80, "latitude": 58, "angular_radius": 5, "temperature_factor": 0.7},
            {"longitude": 60, "latitude": 53, "angular_radius": 6, "temperature_factor": 0.68},
        ],
    }

    def setUp(self) -> None:
        super().setUp()
        self.base_path = str(Path(__file__).resolve().parent.joinpath("../data", "light_curves"))

        settings.configure(ATM_ATLAS="ck04", LIMB_DARKENING_LAW="linear")

    def test_clear_single_system(self) -> None:
        s = prepare_single_system(self.PARAMS["solar"])
        o = Observer(passband=["Generic.Bessell.V"], system=s)

        start_phs, stop_phs, step = -0.2, 1.2, 0.1

        expected = load_light_curve("single.clear.v.json")
        expected_phases = expected[0]
        expected_flux = normalize_lc_for_unittests(expected[1]["Generic.Bessell.V"])

        obtained = o.lc(from_phase=start_phs, to_phase=stop_phs, phase_step=step)
        obtained_phases = obtained[0]
        obtained_flux = normalize_lc_for_unittests(obtained[1]["Generic.Bessell.V"])

        assert np.all(up.abs(np.round(obtained_phases, 3) - np.round(expected_phases, 3)) < TOL)
        assert np.all(up.abs(np.round(obtained_flux, 3) - np.round(expected_flux, 3)) < TOL)

    def test_spotty_single_system(self) -> None:
        s = prepare_single_system(self.PARAMS["solar"], spots=self.SPOTS_META["standard"])
        o = Observer(passband=["Generic.Bessell.V"], system=s)

        start_phs, stop_phs, step = -0.2, 1.2, 0.1

        expected = load_light_curve("single.spotted.v.json")
        expected_phases = expected[0]
        expected_flux = normalize_lc_for_unittests(expected[1]["Generic.Bessell.V"])

        obtained = o.lc(from_phase=start_phs, to_phase=stop_phs, phase_step=step)
        obtained_phases = obtained[0]
        obtained_flux = normalize_lc_for_unittests(obtained[1]["Generic.Bessell.V"])

        assert np.all(up.abs(np.round(obtained_phases, 3) - np.round(expected_phases, 3)) < TOL)
        assert np.all(up.abs(np.round(obtained_flux, 3) - np.round(expected_flux, 3)) < TOL)

    def test_solar_constant(self) -> None:
        s = prepare_single_system(SOLAR_MODEL)
        o = Observer(passband=["bolometric"], system=s)

        start_phs, stop_phs, step = 0.0, 0.1, 0.1

        expected = 1362.0

        obtained = o.lc(from_phase=start_phs, to_phase=stop_phs, phase_step=step)
        obtained_flux = obtained[1]["bolometric"][0]

        np.testing.assert_almost_equal(obtained_flux, expected, decimal=0)
