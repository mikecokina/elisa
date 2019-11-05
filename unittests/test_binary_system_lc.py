import os.path as op
import numpy as np

from importlib import reload
from unittest import mock
from numpy.testing import assert_array_equal
from pypex.poly2d import polygon

from elisa.binary_system.container import OrbitalPositionContainer
from elisa.binary_system.curves import lc
from elisa.binary_system.system import BinarySystem
from elisa.conf import config
from elisa.observer.observer import Observer
from elisa.orbit.container import OrbitalSupplements

from elisa.binary_system import (
    utils as bsutils,
    dynamic
)
from elisa.utils import is_empty
from unittests.utils import (
    ElisaTestCase,
    prepare_binary_system,
    load_light_curve,
    normalize_lc_for_unittests,
    SPOTS_META,
    BINARY_SYSTEM_PARAMS
)
from elisa import (
    umpy as up,
    units,
    const
)

TOL = 1e-3


class MockSelf(object):
    class StarMock(object):
        synchronicity = 1.0
        surface_potential = 10.0

    primary = StarMock()
    secondary = StarMock()
    mass_ratio = 0.5

    @staticmethod
    def has_spots():
        return False

    @staticmethod
    def has_pulsations():
        return False

    @staticmethod
    def calculate_forward_radii(*args, **kwargs):
        c = args[-1]
        return np.array([0.4, 0.45, 0.48, 0.34, 0.6]) if c == "primary" else np.array([0.2, 0.22, 0.19, 0.4, 0.33])


class SupportMethodsTestCase(ElisaTestCase):
    def test_compute_filling_factor(self):
        potential, l_points = 100.0, [2.4078, 2.8758, 2.5772]
        expected = -325.2652
        obtained = round(BinarySystem.compute_filling_factor(potential, l_points), 4)
        self.assertEqual(expected, obtained)

    def test_darkside_filter(self):
        normals = np.array([[1, 1, 1], [0.3, 0.1, -5], [-2, -3, -4.1]])
        normals = np.array([a / b for a, b in zip(normals, np.linalg.norm(normals, axis=1))])
        los = [1, 0, 0]
        obtained = dynamic.darkside_filter(los, normals)
        expected = np.array([0, 1], dtype=int)
        self.assertTrue(np.all(obtained == expected))

    def test_calculate_spot_longitudes(self):
        class MockSpot(object):
            def __init__(self):
                self.longitude = 1.123

        class MockStar(object):
            def __init__(self, s):
                self.spots = {0: MockSpot()}
                self.synchronicity = s

        class MockBinaryInstance(object):
            def __init__(self):
                self.primary = MockStar(1.1)
                self.secondary = MockStar(0.9)

        phases = np.array([-0.9, 0.0, 0.3, 1.2])

        expected = np.round(np.array(
            [np.array([0.55751332, 1.123, 1.31149556, 1.87698224]),
             np.array([1.68848668, 1.123, 0.93450444, 0.36901776])]
        ), 5)

        obtained = dynamic.calculate_spot_longitudes(MockBinaryInstance(), phases)
        obtained = np.round(np.array([obtained["primary"][0], obtained["secondary"][0]]), 5)
        assert_array_equal(expected, obtained)

    def test_surface_area_coverage_not_partial(self):
        size = 5
        visible = [0, 2]
        coverage = [10, 20]

        obtained = lc.surface_area_coverage(size, visible, coverage)
        expected = np.array([10., 0., 20., 0., 0.])
        self.assertTrue(np.all(obtained == expected))

    def test_surface_area_coverage_partial(self):
        size = 5
        visible = [0, 2]
        partial = [1]
        coverage = [10, 20]
        partial_coverage = [30]

        obtained = lc.surface_area_coverage(size, visible, coverage, partial, partial_coverage)
        expected = np.array([10., 30., 20., 0., 0.])
        self.assertTrue(np.all(obtained == expected))

    def test_faces_to_pypex_poly(self):
        points = np.array([[1, 1], [0.3, 0.1], [-2, -3]])
        faces = np.array([points[[0, 1, 2]]])
        obtained = bsutils.faces_to_pypex_poly(faces)[0]
        expected = polygon.Polygon(points)
        self.assertTrue(obtained == expected)

    @staticmethod
    def test_pypex_poly_surface_area():
        points = np.array([[1, 1], [0.3, 0.1], [-2, -3]])
        polygons = [None, polygon.Polygon(points), None]
        obtained = np.round(bsutils.pypex_poly_surface_area(polygons), 5)
        expected = [0.0, 0.05, 0.0]
        assert_array_equal(obtained, expected)

    def test_hull_to_pypex_poly(self):
        hull = np.array([[0, 0], [0, 1], [1, 1]])
        obtained = bsutils.hull_to_pypex_poly(hull)
        self.assertTrue(isinstance(obtained, polygon.Polygon))

    def test__compute_rel_d_radii(self):
        mock_supplements = OrbitalSupplements([[1., 10.]], [[1., 10.]])
        expected = np.array([[0.1111, 0.0625, 0.4118, 0.4333], [0.0909, 0.1579, 0.525, 0.2121]])
        with mock.patch('elisa.binary_system.radius.calculate_forward_radii', MockSelf.calculate_forward_radii):
            obtained = np.round(lc._compute_rel_d_radii(MockSelf, mock_supplements), 4)
        self.assertTrue(np.all(expected == obtained))

    def _test_find_apsidally_corresponding_positions(self, arr1, arr2, expected, tol=1e-10):
        obtained = dynamic.find_apsidally_corresponding_positions(arr1[:, 0], arr1, arr2[:, 0], arr2, tol, [np.nan] * 2)
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
        obtained = dynamic.find_apsidally_corresponding_positions(arr1[:, 0], arr1, arr2[:, 0], arr2, as_empty=[np.nan] * 2)
        self.assertTrue(np.all(~up.isnan(obtained.body)))

    def test_resolve_object_geometry_update(self):
        val_backup = config.MAX_RELATIVE_D_R_POINT
        config.MAX_RELATIVE_D_R_POINT = 0.1
        rel_d_radii = np.array([
            [0.05, 0.04, 0.02, 0.01, 0.1, 1.1, 98, 0.00001],
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.12]
        ])

        expected = np.array([True, False, False, True, False, True, True, True, True], dtype=bool)
        obtained = dynamic.resolve_object_geometry_update(False, rel_d_radii.shape[1] + 1, rel_d_radii)
        config.MAX_RELATIVE_D_R_POINT = val_backup

        self.assertTrue(np.all(expected == obtained))

    def test_get_visible_projection(self):
        obj = MockSelf()
        obj.faces = np.array([[0, 1, 2], [2, 3, 0]])
        obj.indices = np.array([0, 1])
        obj.points = np.array([[-1, -1, -2], [0., 1, 1], [1, 1, 2], [2, 3, 4]])

        obtained = bsutils.get_visible_projection(obj)
        expected = np.vstack((obj.points[:, 1], obj.points[:, 2])).T
        self.assertTrue(np.all(expected == obtained))

    def test_partial_visible_faces_surface_coverage(self):
        points = np.array([[-1, 0.5], [1, 0.5], [0, 1.5]])
        faces = np.array([[0, 1, 2]])
        normals = np.array([[1, -1, 0]]) / np.linalg.norm(np.array([-1, 1, 0]))
        hull = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])

        obtained = np.round(lc.partial_visible_faces_surface_coverage(points, faces, normals, hull), 10)
        expected = np.round(0.5 / up.cos(up.pi / 4.0), 10)

        self.assertTrue(np.all(obtained == expected))

    @staticmethod
    def test_resolve_spots_geometry_update():
        config.MAX_SPOT_D_LONGITUDE = 0.06
        spots_longitudes = {
            'primary': {
                0: np.array([0.5, 0.55, 0.6, 0.75, 0.76, 0.77]),
                1: np.array([0.5, 0.55, 0.6, 0.75, 0.76, 0.77]) - 0.1
            },
            'secondary': {}
        }

        obtained = np.array(dynamic.resolve_spots_geometry_update(spots_longitudes), dtype=bool)
        expected = np.array([[True, False, True, True, False, False], [True] + [False] * 5], dtype=bool)
        assert_array_equal(expected, obtained)

    def test_phase_crv_symmetry(self):
        phase = up.arange(0, 1.2, 0.2)
        obtained = dynamic.phase_crv_symmetry(MockSelf, phase)
        expected = np.array([0, 0.2, 0.4]), np.array([0, 1, 2, 2, 1, 0], dtype=int)
        self.assertTrue(np.all(obtained[0] == expected[0]))
        self.assertTrue(np.all(obtained[1] == expected[1]))

        # is imutable???
        self.assertTrue(np.all(up.arange(0, 1.2, 0.2) == phase))

    def test_visible_indices_when_darkside_filter_apply(self):
        bs = prepare_binary_system(BINARY_SYSTEM_PARAMS['detached-physical'])
        from_this = dict(binary_system=bs, position=const.BINARY_POSITION_PLACEHOLDER(0, 1.0, 0.0, 0.0, 0.0))
        system = OrbitalPositionContainer.from_binary_system(**from_this)
        system.build(components_distance=1.0)
        system.apply_darkside_filter()
        self.assertTrue((not is_empty(system.primary.indices)) and (not is_empty(system.secondary.indices)))


class ComputeLightCurvesTestCase(ElisaTestCase):
    params = {
        'detached': {
            "primary_mass": 2.0, "secondary_mass": 1.0,
            "primary_surface_potential": 5.0, "secondary_surface_potential": 5.0,
            "primary_synchronicity": 1.0, "secondary_synchronicity": 1.0,
            "argument_of_periastron": const.HALF_PI * units.rad, "gamma": 0.0, "period": 5.0,
            "eccentricity": 0.0, "inclination": const.HALF_PI * units.rad, "primary_minimum_time": 0.0,
            "phase_shift": 0.0,
            "primary_t_eff": 6500, "secondary_t_eff": 6500,
            "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
            "primary_albedo": 1.0, "secondary_albedo": 1.0,
        },
        'detached-async-ecc': {
            "primary_mass": 2.0, "secondary_mass": 1.0,
            "primary_surface_potential": 5.0, "secondary_surface_potential": 5.0,
            "primary_synchronicity": 0.8, "secondary_synchronicity": 1.2,
            "argument_of_periastron": const.HALF_PI * units.rad, "gamma": 0.0, "period": 5.0,
            "eccentricity": 0.3, "inclination": const.HALF_PI * units.rad, "primary_minimum_time": 0.0,
            "phase_shift": 0.0,
            "primary_t_eff": 6500, "secondary_t_eff": 6500,
            "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
            "primary_albedo": 1.0, "secondary_albedo": 1.0,
        },
        'detached-async': {
            "primary_mass": 2.0, "secondary_mass": 1.0,
            "primary_surface_potential": 5.0, "secondary_surface_potential": 5.0,
            "primary_synchronicity": 0.8, "secondary_synchronicity": 1.2,
            "argument_of_periastron": const.HALF_PI * units.rad, "gamma": 0.0, "period": 5.0,
            "eccentricity": 0.0, "inclination": const.HALF_PI * units.rad, "primary_minimum_time": 0.0,
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
            "argument_of_periastron": 90 * units.deg, "gamma": 0.0, "period": 1.0,
            "eccentricity": 0.0, "inclination": 90.0 * units.deg, "primary_minimum_time": 0.0,
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
            "argument_of_periastron": 223 * units.deg, "gamma": 0.0, "period": 3.0,
            "eccentricity": 0.3, "inclination": 90.0 * units.deg, "primary_minimum_time": 0.0,
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
        reload(lc)

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
            o.lc(from_phase=start_phs, to_phase=stop_phs, phase_step=step)

    def test_circular_synchronous_detached_system(self):
        config.LIMB_DARKENING_LAW = "linear"
        reload(lc)

        bs = prepare_binary_system(self.params["detached"])
        o = Observer(passband=['Generic.Bessell.V'], system=bs)

        start_phs, stop_phs, step = -0.2, 1.2, 0.01

        expected = load_light_curve("detached.circ.sync.generic.bessel.v.json")
        expected_phases = expected[0]
        expected_flux = normalize_lc_for_unittests(expected[1]["Generic.Bessell.V"])

        obtained = o.lc(from_phase=start_phs, to_phase=stop_phs, phase_step=step)
        obtained_phases = obtained[0]
        obtained_flux = normalize_lc_for_unittests(obtained[1]["Generic.Bessell.V"])

        self.assertTrue(np.all(up.abs(np.round(obtained_phases, 3) - np.round(expected_phases, 3)) < TOL))
        self.assertTrue(np.all(up.abs(np.round(obtained_flux, 3) - np.round(expected_flux, 3)) < TOL))

    def test_circular_synchronous_overcontact_system(self):
        config.LIMB_DARKENING_LAW = "linear"
        reload(lc)

        bs = prepare_binary_system(self.params["over-contact"])
        o = Observer(passband=['Generic.Bessell.V'], system=bs)

        start_phs, stop_phs, step = -0.2, 1.2, 0.01

        expected = load_light_curve("overcontact.circ.sync.generic.bessel.v.json")
        expected_phases = expected[0]
        expected_flux = normalize_lc_for_unittests(expected[1]["Generic.Bessell.V"])

        obtained = o.lc(from_phase=start_phs, to_phase=stop_phs, phase_step=step)
        obtained_phases = obtained[0]
        obtained_flux = normalize_lc_for_unittests(obtained[1]["Generic.Bessell.V"])

        self.assertTrue(np.all(up.abs(np.round(obtained_phases, 3) - np.round(expected_phases, 3)) < TOL))
        self.assertTrue(np.all(up.abs(np.round(obtained_flux, 3) - np.round(expected_flux, 3)) < TOL))

    def test_eccentric_synchronous_detached_system_no_approximation(self):
        config.POINTS_ON_ECC_ORBIT = -1
        config.MAX_RELATIVE_D_R_POINT = 0.0
        reload(lc)

        bs = prepare_binary_system(self.params["eccentric"])
        o = Observer(passband=['Generic.Bessell.V'], system=bs)

        start_phs, stop_phs, step = -0.2, 1.2, 0.1

        obtained = o.lc(from_phase=start_phs, to_phase=stop_phs, phase_step=step)

        obtained_phases = obtained[0]
        obtained_flux = normalize_lc_for_unittests(obtained[1]["Generic.Bessell.V"])

        expected = load_light_curve("detached.ecc.sync.generic.bessell.v.json")
        expected_phases = expected[0]
        expected_flux = normalize_lc_for_unittests(expected[1]["Generic.Bessell.V"])

        self.assertTrue(np.all(up.abs(np.round(obtained_phases, 3) - np.round(expected_phases, 3)) < TOL))
        self.assertTrue(np.all(up.abs(np.round(obtained_flux, 3) - np.round(expected_flux, 3)) < TOL))

    def test_eccentric_synchronous_detached_system_approximation_one(self):
        config.POINTS_ON_ECC_ORBIT = 5
        config.MAX_RELATIVE_D_R_POINT = 0.0
        reload(lc)

        bs = prepare_binary_system(self.params["eccentric"])
        o = Observer(passband=['Generic.Bessell.V'], system=bs)

        start_phs, stop_phs, step = -0.2, 1.2, 0.1

        obtained = o.lc(from_phase=start_phs, to_phase=stop_phs, phase_step=step)
        obtained_phases = obtained[0]
        obtained_flux = normalize_lc_for_unittests(obtained[1]["Generic.Bessell.V"])

        expected = load_light_curve("detached.ecc.sync.generic.bessell.v.appx_one.json")
        expected_phases = expected[0]
        expected_flux = normalize_lc_for_unittests(expected[1]["Generic.Bessell.V"])

        self.assertTrue(np.all(up.abs(np.round(obtained_phases, 3) - np.round(expected_phases, 3)) < TOL))
        self.assertTrue(np.all(up.abs(np.round(obtained_flux, 3) - np.round(expected_flux, 3)) < TOL))

    def test_eccentric_synchronous_detached_system_approximation_two(self):
        config.POINTS_ON_ECC_ORBIT = int(1e6)
        config.MAX_RELATIVE_D_R_POINT = 0.05
        config.MAX_SUPPLEMENTAR_D_DISTANCE = 0.05
        reload(lc)

        bs = prepare_binary_system(self.params["eccentric"])
        o = Observer(passband=['Generic.Bessell.V'], system=bs)

        start_phs, stop_phs, step = -0.2, 1.2, 0.1

        obtained = o.lc(from_phase=start_phs, to_phase=stop_phs, phase_step=step)
        obtained_phases = obtained[0]
        obtained_flux = normalize_lc_for_unittests(obtained[1]["Generic.Bessell.V"])

        expected = load_light_curve("detached.ecc.sync.generic.bessell.v.appx_two.json")
        expected_phases = expected[0]
        expected_flux = normalize_lc_for_unittests(expected[1]["Generic.Bessell.V"])

        self.assertTrue(np.all(np.round(obtained_phases, 3) - np.round(expected_phases, 3) < TOL))
        self.assertTrue(np.all(np.round(obtained_flux, 3) - np.round(expected_flux, 3) < TOL))

        expected_exact = load_light_curve("detached.ecc.sync.generic.bessell.v.json")
        expected_flux_exact = normalize_lc_for_unittests(expected_exact[1]["Generic.Bessell.V"])
        self.assertTrue(np.all(up.abs(np.round(obtained_flux, 3) - np.round(expected_flux_exact, 3)) < 5e-3))

        # from matplotlib import pyplot as plt
        # plt.scatter(expected_phases_exact, expected_flux_exact, marker="o")
        # plt.show()

    def test_eccentric_asynchronous_detached_system(self):
        config.LIMB_DARKENING_LAW = "linear"
        reload(lc)

        bs = prepare_binary_system(self.params["detached-async-ecc"])
        o = Observer(passband=['Generic.Bessell.V'], system=bs)

        start_phs, stop_phs, step = -0.2, 1.2, 0.1

        obtained = o.lc(from_phase=start_phs, to_phase=stop_phs, phase_step=step)
        obtained_phases = obtained[0]
        obtained_flux = normalize_lc_for_unittests(obtained[1]["Generic.Bessell.V"])

        expected = load_light_curve("detached.ecc.async.generic.bessel.v.json")
        expected_phases = expected[0]
        expected_flux = normalize_lc_for_unittests(expected[1]["Generic.Bessell.V"])

        self.assertTrue(np.all(np.round(obtained_phases, 3) - np.round(expected_phases, 3) < TOL))
        self.assertTrue(np.all(np.round(obtained_flux, 3) - np.round(expected_flux, 3) < TOL))

    def test_circular_spotty_synchronous_detached_system(self):
        bs = prepare_binary_system(self.params["detached"],
                                   spots_primary=SPOTS_META["primary"],
                                   spots_secondary=SPOTS_META["secondary"])
        o = Observer(passband=['Generic.Bessell.V'], system=bs)

        start_phs, stop_phs, step = -0.2, 1.2, 0.01

        obtained = o.lc(from_phase=start_phs, to_phase=stop_phs, phase_step=step)
        obtained_phases = obtained[0]
        obtained_flux = normalize_lc_for_unittests(obtained[1]["Generic.Bessell.V"])

        expected = load_light_curve("detached.circ.spotty.sync.generic.bessel.v.json")
        expected_phases = expected[0]
        expected_flux = normalize_lc_for_unittests(expected[1]["Generic.Bessell.V"])

        self.assertTrue(np.all(np.round(obtained_phases, 3) - np.round(expected_phases, 3) < TOL))
        self.assertTrue(np.all(np.round(obtained_flux, 3) - np.round(expected_flux, 3) < TOL))

    def test_circular_spotty_synchronous_overcontact_system(self):
        bs = prepare_binary_system(self.params["over-contact"],
                                   spots_primary=SPOTS_META["primary"],
                                   spots_secondary=SPOTS_META["secondary"])
        o = Observer(passband=['Generic.Bessell.V'], system=bs)

        start_phs, stop_phs, step = -0.2, 1.2, 0.01

        obtained = o.lc(from_phase=start_phs, to_phase=stop_phs, phase_step=step)
        obtained_phases = obtained[0]
        obtained_flux = normalize_lc_for_unittests(obtained[1]["Generic.Bessell.V"])

        expected = load_light_curve("overcontact.circ.spotty.sync.generic.bessel.v.json")
        expected_phases = expected[0]
        expected_flux = normalize_lc_for_unittests(expected[1]["Generic.Bessell.V"])

        self.assertTrue(np.all(np.round(obtained_phases, 3) - np.round(expected_phases, 3) < TOL))
        self.assertTrue(np.all(np.round(obtained_flux, 3) - np.round(expected_flux, 3) < TOL))

    def test_cicular_spotty_asynchronous_detached_system(self):
        config.MAX_SPOT_D_LONGITUDE = up.pi / 45.0
        reload(lc)

        bs = prepare_binary_system(self.params["detached-async"],
                                   spots_primary=SPOTS_META["primary"])
        o = Observer(passband=['Generic.Bessell.V'], system=bs)
        start_phs, stop_phs, step = -0.2, 1.2, 0.05
        obtained = o.lc(from_phase=start_phs, to_phase=stop_phs, phase_step=step)
        obtained_phases = obtained[0]
        obtained_flux = normalize_lc_for_unittests(obtained[1]["Generic.Bessell.V"])

        expected = load_light_curve("detached.circ.spotty.async.generic.bessel.v.json")
        expected_phases = expected[0]
        expected_flux = normalize_lc_for_unittests(expected[1]["Generic.Bessell.V"])

        self.assertTrue(np.all(abs(np.round(expected_phases, 3) == np.round(obtained_phases, 3))))
        assert_array_equal(np.round(obtained_flux, 3), np.round(expected_flux, 3))

    def test_eccentric_spotty_asynchronous_detached_system(self):
        bs = prepare_binary_system(self.params["detached-async-ecc"],
                                   spots_primary=SPOTS_META["primary"])
        o = Observer(passband=['Generic.Bessell.V'], system=bs)
        start_phs, stop_phs, step = -0.2, 1.2, 0.1
        obtained = o.lc(from_phase=start_phs, to_phase=stop_phs, phase_step=step)
        obtained_phases = obtained[0]
        obtained_flux = normalize_lc_for_unittests(obtained[1]["Generic.Bessell.V"])

        expected = load_light_curve("detached.ecc.spotty.async.generic.bessel.v.json")
        expected_phases = expected[0]
        expected_flux = normalize_lc_for_unittests(expected[1]["Generic.Bessell.V"])

        self.assertTrue(np.all(np.round(obtained_phases, 3) - np.round(expected_phases, 3) < TOL))
        self.assertTrue(np.all(np.round(obtained_flux, 3) - np.round(expected_flux, 3) < TOL))
