import os.path as op
import unittest

import numpy as np
from astropy import units as u
from numpy.testing import assert_array_equal

from elisa import const as c
from elisa.binary_system import lc
from elisa.binary_system.geo import OrbitalSupplements
from elisa.conf import config
from elisa.observer.observer import Observer
from unittests.utils import prepare_binary_system, load_light_curve, normalize_lc_for_unittests, ElisaTestCase


class MockSelf(object):
    @staticmethod
    def has_spots():
        return False

    @staticmethod
    def has_pulsations():
        return False

    @staticmethod
    def calculate_all_forward_radii(*args, **kwargs):
        return {
            "primary": np.array([0.4, 0.45, 0.48, 0.34, 0.6]),
            "secondary": np.array([0.2, 0.22, 0.19, 0.4, 0.33])
        }


class SupportMethodsTestCase(ElisaTestCase):
    def _test_find_apsidally_corresponding_positions(self, arr1, arr2, expected, tol=1e-10):
        obtained = lc.find_apsidally_corresponding_positions(arr1[:, 0], arr1, arr2[:, 0], arr2, tol, [np.nan] * 2)
        self.assertTrue(expected == obtained)

    def test_find_apsidally_corresponding_positions_full_match(self):
        arr1 = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5.0, 50]])
        arr2 = np.array([[1, 10], [3, 30], [2, 20], [4, 40], [5.0, 50]])
        expected = OrbitalSupplements([[1, 10], [3, 30], [2, 20], [4, 40], [5.0, 50]],
                                      [[1, 10], [3, 30], [2, 20], [4, 40], [5.0, 50]])

        self._test_find_apsidally_corresponding_positions(arr1, arr2, expected)

    def test_find_apsidally_corresponding_positions_mixed_first_longer(self):
        arr1 = np.array([[1, 10], [2, 20], [5, 50], [6, 60], [7, 70]])
        arr2 = np.array([[1, 10], [2, 20], [5.5, 50.50], [7, 70]])
        nan = [np.nan, np.nan]
        expected = OrbitalSupplements([[1.0, 10.], [2.0, 20.0], [5.5, 50.5], [7.0, 70], [5, 50], [6, 60]],
                                      [[1, 10], [2, 20], nan, [7.0, 70], nan, nan])
        self._test_find_apsidally_corresponding_positions(arr1, arr2, expected)

    def test_find_apsidally_corresponding_positions_mixed_second_longer(self):
        arr1 = np.array([[1, 10], [2, 20], [5.5, 50.50], [7, 70]])
        arr2 = np.array([[1, 10], [2, 20], [5, 50], [6, 60], [7, 70]])
        nan = [np.nan, np.nan]
        expected = OrbitalSupplements([[1., 10.], [2., 20.], [5., 50.], [6., 60.], [7., 70.], [5.5, 50.5]],
                                      [[1., 10.], [2., 20.], nan, nan, [7., 70.], nan])
        self._test_find_apsidally_corresponding_positions(arr1, arr2, expected)

    def test_find_apsidally_corresponding_positions_mixed_under_tolerance(self):
        arr1 = np.array([[1, 10], [2, 20], [3, 30], [4, 40]])
        arr2 = np.array([[1, 10], [1.01, 10.10], [2, 20], [2.02, 20.02], [4, 40]])
        expected = OrbitalSupplements([[1, 10], [1, 10], [2, 20], [2, 20], [4, 40], [3, 30]],
                                      [[1, 10], [1.01, 10.1], [2, 20], [2.02, 20.02], [4, 40], [np.nan, np.nan]])
        self._test_find_apsidally_corresponding_positions(arr1, arr2, expected, tol=0.1)

    def test_find_apsidally_corresponding_positions_total_mixed(self):
        arr1 = np.array([[1, 10], [2, 20]])
        arr2 = np.array([[3, 30], [4, 40]])
        nan = [np.nan, np.nan]
        expected = OrbitalSupplements([[3, 30], [4, 40], [1, 10], [2, 20]], [nan, nan, nan, nan])
        self._test_find_apsidally_corresponding_positions(arr1, arr2, expected)

    def test_find_apsidally_corresponding_positions_body_not_empty(self):
        arr1 = np.array([[1, 10], [2, 20]])
        arr2 = np.array([[3, 30], [4, 40]])
        obtained = lc.find_apsidally_corresponding_positions(arr1[:, 0], arr1, arr2[:, 0], arr2, as_empty=[np.nan] * 2)
        self.assertTrue(np.all(~np.isnan(obtained.body)))

    def test__resolve_geometry_update(self):
        val_backup = config.MAX_RELATIVE_D_R_POINT
        config.MAX_RELATIVE_D_R_POINT = 0.1
        rel_d_radii = np.array([
            [0.05, 0.04, 0.02, 0.01, 0.1, 1.1, 98, 0.00001],
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.12]
        ])

        expected = np.array([True, False, False, True, False, True, True, True, True], dtype=bool)
        obtained = lc._resolve_geometry_update(MockSelf, rel_d_radii.shape[1] + 1, rel_d_radii)
        config.MAX_RELATIVE_D_R_POINT = val_backup

        self.assertTrue(np.all(expected == obtained))

    def test__compute_rel_d_radii(self):
        mock_supplements = OrbitalSupplements([[1., 10.]], [[1., 10.]])
        expected = np.array([[0.1111, 0.0625, 0.4118, 0.4333], [0.0909, 0.1579, 0.525, 0.2121]])
        obtained = np.round(lc._compute_rel_d_radii(MockSelf, mock_supplements), 4)
        self.assertTrue(np.all(expected == obtained))

    def test_get_visible_projection(self):
        obj = MockSelf()
        obj.faces = np.array([[0, 1, 2], [2, 3, 0]])
        obj.indices = np.array([0, 1])
        obj.points = np.array([[-1, -1, -2], [0., 1, 1], [1, 1, 2], [2, 3, 4]])

        obtained = lc.get_visible_projection(obj)
        expected = np.vstack((obj.points[:, 1], obj.points[:, 2])).T
        self.assertTrue(np.all(expected == obtained))

    def test_partial_visible_faces_surface_coverage(self):
        points = np.array([[-1, 0.5], [1, 0.5], [0, 1.5]])
        faces = np.array([[0, 1, 2]])
        normals = np.array([[1, -1, 0]]) / np.linalg.norm(np.array([-1, 1, 0]))
        hull = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])

        obtained = np.round(lc.partial_visible_faces_surface_coverage(points, faces, normals, hull), 10)
        expected = np.round(0.5 / np.cos(np.pi / 4.0), 10)

        self.assertTrue(np.all(obtained == expected))

    def test__phase_crv_symmetry(self):
        phase = np.arange(0, 1.2, 0.2)
        obtained = lc._phase_crv_symmetry(MockSelf, phase)
        expected = np.array([0, 0.2, 0.4]), np.array([0, 1, 2, 2, 1, 0], dtype=int)
        self.assertTrue(np.all(obtained[0] == expected[0]))
        self.assertTrue(np.all(obtained[1] == expected[1]))

        # is imutable???
        self.assertTrue(np.all(np.arange(0, 1.2, 0.2) == phase))


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

    def test_light_curve_pass_on_all_ld_law(self):
        """
        no assert here, it just has to pass without error
        """
        bs = prepare_binary_system(self.params["detached"])
        start_phs, stop_phs, step = -0.2, 1.2, 0.1

        laws = config.LD_LAW_TO_FILE_PREFIX.keys()
        for law in laws:
            config.LIMB_DARKENING_LAW = law
            o = Observer(passband=['Generic.Bessell.V'], system=bs)
            o.observe(from_phase=start_phs, to_phase=stop_phs, phase_step=step)

    def test_circular_synchronous_detached_system(self):
        config.LIMB_DARKENING_LAW = "linear"

        bs = prepare_binary_system(self.params["detached"])
        o = Observer(passband=['Generic.Bessell.V'], system=bs)

        start_phs, stop_phs, step = -0.2, 1.2, 0.01

        expected = load_light_curve("detached.circ.sync.generic.bessel.v.json")
        expected_phases = expected[0]
        expected_flux = normalize_lc_for_unittests(expected[1]["Generic.Bessell.V"])

        obtained = o.observe(from_phase=start_phs, to_phase=stop_phs, phase_step=step)
        obtained_phases = obtained[0]
        obtained_flux = normalize_lc_for_unittests(obtained[1]["Generic.Bessell.V"])

        self.assertTrue(np.all(np.round(obtained_phases, 4) == np.round(expected_phases, 4)))
        self.assertTrue(np.all(np.round(obtained_flux, 4) == np.round(expected_flux, 4)))

    def test_circular_synchronous_overcontact_system(self):
        config.LIMB_DARKENING_LAW = "linear"

        bs = prepare_binary_system(self.params["over-contact"])
        o = Observer(passband=['Generic.Bessell.V'], system=bs)

        start_phs, stop_phs, step = -0.2, 1.2, 0.01

        expected = load_light_curve("overcontact.circ.sync.generic.bessel.v.json")
        expected_phases = expected[0]
        expected_flux = normalize_lc_for_unittests(expected[1]["Generic.Bessell.V"])

        obtained = o.observe(from_phase=start_phs, to_phase=stop_phs, phase_step=step)
        obtained_phases = obtained[0]
        obtained_flux = normalize_lc_for_unittests(obtained[1]["Generic.Bessell.V"])

        self.assertTrue(np.all(np.round(obtained_phases, 4) == np.round(expected_phases, 4)))
        self.assertTrue(np.all(np.round(obtained_flux, 4) - np.round(expected_flux, 4) <= 1e-3))

    def test_eccentric_synchronous_detached_system_no_approximation(self):
        config.POINTS_ON_ECC_ORBIT = int(1e6)
        config.MAX_RELATIVE_D_R_POINT = 0.0

        bs = prepare_binary_system(self.params["eccentric"])
        o = Observer(passband=['Generic.Bessell.V'], system=bs)

        start_phs, stop_phs, step = -0.2, 1.2, 0.1

        obtained = o.observe(from_phase=start_phs, to_phase=stop_phs, phase_step=step)
        obtained_phases = obtained[0]
        obtained_flux = normalize_lc_for_unittests(obtained[1]["Generic.Bessell.V"])

        expected = load_light_curve("detached.ecc.sync.generic.bessell.v.json")
        expected_phases = expected[0]
        expected_flux = normalize_lc_for_unittests(expected[1]["Generic.Bessell.V"])

        self.assertTrue(np.all(np.round(obtained_phases, 4) == np.round(expected_phases, 4)))
        assert_array_equal(np.round(obtained_flux, 4), np.round(expected_flux, 4))

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
