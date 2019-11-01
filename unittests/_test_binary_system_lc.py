import os.path as op

import numpy as np
from astropy import units as u
from numpy.testing import assert_array_equal
from unittest import skip

from elisa import const as c, units
from elisa.binary_system.curves import lc
from elisa.conf import config
from elisa.observer.observer import Observer
from elisa.orbit.container import OrbitalSupplements
from unittests.utils import prepare_binary_system, load_light_curve, normalize_lc_for_unittests, ElisaTestCase



class SupportMethodsTestCase(ElisaTestCase):

    def test__compute_rel_d_radii(self):
        mock_supplements = OrbitalSupplements([[1., 10.]], [[1., 10.]])
        expected = np.array([[0.1111, 0.0625, 0.4118, 0.4333], [0.0909, 0.1579, 0.525, 0.2121]])
        obtained = np.round(lc._compute_rel_d_radii(MockSelf, mock_supplements), 4)
        self.assertTrue(np.all(expected == obtained))


class ComputeLightCurvesTestCase(ElisaTestCase):
    params = {
        'detached': {
            "primary_mass": 2.0, "secondary_mass": 1.0,
            "primary_surface_potential": 5.0, "secondary_surface_potential": 5.0,
            "primary_synchronicity": 1.0, "secondary_synchronicity": 1.0,
            "argument_of_periastron": c.HALF_PI * u.rad, "gamma": 0.0, "period": 5.0,
            "eccentricity": 0.0, "inclination": c.HALF_PI * u.rad, "primary_minimum_time": 0.0,
            "phase_shift": 0.0,
            "primary_t_eff": 6500, "secondary_t_eff": 6500,
            "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
            "primary_albedo": 1.0, "secondary_albedo": 1.0,
        },
        'over-contact': {
            "primary_mass": 2.0, "secondary_mass": 1.0,
            "primary_surface_potential": 2.7,
            "secondary_surface_potential": 2.7,
            "primary_synchronicity": 1.0, "secondary_synchronicity": 1.0,
            "argument_of_periastron": 90 * u.deg, "gamma": 0.0, "period": 1.0,
            "eccentricity": 0.0, "inclination": 90.0 * u.deg, "primary_minimum_time": 0.0,
            "phase_shift": 0.0,
            "primary_t_eff": 6000, "secondary_t_eff": 6000,
            "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
            "primary_albedo": 1.0, "secondary_albedo": 1.0
        },
        'eccentric': {
            "primary_mass": 1.0, "secondary_mass": 1.0,
            "primary_surface_potential": 8,
            "secondary_surface_potential": 8,
            "primary_synchronicity": 1.0, "secondary_synchronicity": 1.0,
            "argument_of_periastron": 223 * u.deg, "gamma": 0.0, "period": 3.0,
            "eccentricity": 0.3, "inclination": 90.0 * u.deg, "primary_minimum_time": 0.0,
            "phase_shift": 0.0,
            "primary_t_eff": 6000, "secondary_t_eff": 6000,
            "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
            "primary_albedo": 1.0, "secondary_albedo": 1.0
        }
    }

    def setUp(self):
        # raise unittest.SkipTest(message)
        self.base_path = op.join(op.dirname(op.abspath(__file__)), "data", "light_curves")
        self.law = config.LIMB_DARKENING_LAW

        config.VAN_HAMME_LD_TABLES = op.join(self.base_path, "limbdarkening")
        config.CK04_ATM_TABLES = op.join(self.base_path, "atmosphere")
        config.ATM_ATLAS = "ck04"
        config._update_atlas_to_base_dir()

    def tearDown(self):
        config.LIMB_DARKENING_LAW = self.law


    def test_eccentric_synchronous_detached_system_approximation_one(self):
        config.POINTS_ON_ECC_ORBIT = 5
        config.MAX_RELATIVE_D_R_POINT = 0.0

        bs = prepare_binary_system(self.params["eccentric"])
        o = Observer(passband=['Generic.Bessell.V'], system=bs)

        start_phs, stop_phs, step = -0.2, 1.2, 0.1

        obtained = o.observe(from_phase=start_phs, to_phase=stop_phs, phase_step=step)
        obtained_phases = obtained[0]
        obtained_flux = normalize_lc_for_unittests(obtained[1]["Generic.Bessell.V"])

        expected_exact = load_light_curve("detached.ecc.sync.generic.bessell.v.json")
        expected_phases_exact = expected_exact[0]
        expected_flux_exact = normalize_lc_for_unittests(expected_exact[1]["Generic.Bessell.V"])

        # todo: for now, it is OK if phase are equal but fixme
        # fixme: alter approximation one/all methods to be computet with enforced significant phases like (minima, etc.)
        self.assertTrue(np.all(abs(np.round(expected_phases_exact, 4) == np.round(obtained_phases, 4))))

    def test_eccentric_synchronous_detached_system_approximation_two(self):
        config.POINTS_ON_ECC_ORBIT = int(1e6)
        config.MAX_RELATIVE_D_R_POINT = 0.05
        config.MAX_SUPPLEMENTAR_D_DISTANCE = 0.05

        bs = prepare_binary_system(self.params["eccentric"])
        o = Observer(passband=['Generic.Bessell.V'], system=bs)

        start_phs, stop_phs, step = -0.2, 1.2, 0.1

        obtained = o.observe(from_phase=start_phs, to_phase=stop_phs, phase_step=step)
        obtained_phases = obtained[0]
        obtained_flux = normalize_lc_for_unittests(obtained[1]["Generic.Bessell.V"])

        expected_exact = load_light_curve("detached.ecc.sync.generic.bessell.v.json")
        expected_phases_exact = expected_exact[0]
        expected_flux_exact = normalize_lc_for_unittests(expected_exact[1]["Generic.Bessell.V"])

        self.assertTrue(np.all(np.round(expected_phases_exact, 4) == np.round(obtained_phases, 4)))
        self.assertTrue(np.all(abs(obtained_flux - expected_flux_exact) < 1e5))

        # from matplotlib import pyplot as plt
        # plt.scatter(expected_phases_exact, expected_flux_exact, marker="o")
        # plt.show()


class ComputeLightCurveWithSpots(ElisaTestCase):
    spots_metadata = {
        "primary":
            [
                {"longitude": 90,
                 "latitude": 58,
                 "angular_radius": 35,
                 "temperature_factor": 0.95},
            ],

        "secondary":
            [
                {"longitude": 60,
                 "latitude": 45,
                 "angular_radius": 28,
                 "temperature_factor": 0.9},
            ]
    }

    params = {
        "detached": {
            "primary_mass": 2.0, "secondary_mass": 1.0,
            "primary_surface_potential": 10.0, "secondary_surface_potential": 10.0,
            "primary_synchronicity": 1.0, "secondary_synchronicity": 1.0,
            "argument_of_periastron": c.HALF_PI * u.rad, "gamma": 0.0, "period": 5.0,
            "eccentricity": 0.0, "inclination": c.HALF_PI * units.deg, "primary_minimum_time": 0.0,
            "phase_shift": 0.0,
            "primary_t_eff": 5000, "secondary_t_eff": 5000,
            "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
            "primary_albedo": 0.6, "secondary_albedo": 0.6,
        },
        "over-contact": {
            "primary_mass": 2.0, "secondary_mass": 1.0,
            "primary_surface_potential": 2.7,
            "secondary_surface_potential": 2.7,
            "primary_synchronicity": 1.0, "secondary_synchronicity": 1.0,
            "argument_of_periastron": 90 * units.deg, "gamma": 0.0, "period": 1.0,
            "eccentricity": 0.0, "inclination": 90.0 * units.deg, "primary_minimum_time": 0.0,
            "phase_shift": 0.0,
            "primary_t_eff": 5000, "secondary_t_eff": 5000,
            "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
            "primary_albedo": 0.6, "secondary_albedo": 0.6
        }
    }

    def setUp(self):
        self.base_path = op.join(op.dirname(op.abspath(__file__)), "data", "light_curves")

        config.VAN_HAMME_LD_TABLES = op.join(self.base_path, "limbdarkening")
        config.CK04_ATM_TABLES = op.join(self.base_path, "atmosphere")
        config.ATM_ATLAS = "ck04"
        config._update_atlas_to_base_dir()

    @skip
    def test_circular_synchronous_detached_system_w_spots(self):
        s = prepare_binary_system(self.params["detached"],
                                  spots_primary=self.spots_metadata["primary"],
                                  spots_secondary=self.spots_metadata["secondary"])

        o = Observer(passband=['Generic.Bessell.V'], system=s)
        start_phs, stop_phs, step = -0.2, 1.2, 0.1
        obtained = o.observe(from_phase=start_phs, to_phase=stop_phs, phase_step=step)
