import os.path as op
import numpy as np

from unittest import mock, skip
from numpy.testing import assert_array_equal, assert_allclose
from pypex.poly2d import polygon
from copy import deepcopy

from elisa.binary_system.container import OrbitalPositionContainer
from elisa.binary_system.system import BinarySystem
from elisa import settings
from elisa.observer.observer import Observer
from elisa.binary_system.orbit.container import OrbitalSupplements
from elisa.binary_system import surface
from elisa.binary_system.curves import utils as crv_utils
from elisa.base.surface import coverage
from elisa.base.container import PositionContainer

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
    BINARY_SYSTEM_PARAMS,
    APPROX_SETTINGS
)
from elisa import (
    umpy as up,
    units as u,
    const,
    utils)


TOL = 5e-3

PARAMS = {
        'detached': {
            "primary_mass": 2.0, "secondary_mass": 1.0,
            "primary_surface_potential": 5.0, "secondary_surface_potential": 5.0,
            "primary_synchronicity": 1.0, "secondary_synchronicity": 1.0,
            "argument_of_periastron": const.HALF_PI * u.rad, "gamma": 0.0, "period": 5.0,
            "eccentricity": 0.0, "inclination": const.HALF_PI * u.rad, "primary_minimum_time": 0.0,
            "phase_shift": 0.0,
            "primary_t_eff": 6500, "secondary_t_eff": 6500,
            "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
            "primary_albedo": 1.0, "secondary_albedo": 1.0,
        },
        'detached-async-ecc': {
            "primary_mass": 2.0, "secondary_mass": 1.0,
            "primary_surface_potential": 5.0, "secondary_surface_potential": 5.0,
            "primary_synchronicity": 0.8, "secondary_synchronicity": 1.2,
            "argument_of_periastron": const.HALF_PI * u.rad, "gamma": 0.0, "period": 5.0,
            "eccentricity": 0.3, "inclination": const.HALF_PI * u.rad, "primary_minimum_time": 0.0,
            "phase_shift": 0.0,
            "primary_t_eff": 6500, "secondary_t_eff": 6500,
            "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
            "primary_albedo": 1.0, "secondary_albedo": 1.0,
        },
        'detached-async': {
            "primary_mass": 2.0, "secondary_mass": 1.0,
            "primary_surface_potential": 5.0, "secondary_surface_potential": 5.0,
            "primary_synchronicity": 0.8, "secondary_synchronicity": 1.2,
            "argument_of_periastron": const.HALF_PI * u.rad, "gamma": 0.0, "period": 5.0,
            "eccentricity": 0.0, "inclination": const.HALF_PI * u.rad, "primary_minimum_time": 0.0,
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
    def setUp(self):
        self.lc_base_path = op.join(op.dirname(op.abspath(__file__)), "data", "light_curves")

    def test_compute_filling_factor(self):
        potential, l_points = 100.0, [2.4078, 2.8758, 2.5772]
        expected = -325.2652
        obtained = round(BinarySystem.compute_filling_factor(potential, l_points), 4)
        self.assertEqual(expected, obtained)

    def test_darkside_filter(self):
        normals = np.array([[1, 1, 1], [0.3, 0.1, -5], [-2, -3, -4.1]])
        normals = np.array([a / b for a, b in zip(normals, np.linalg.norm(normals, axis=1))])
        los = [1, 0, 0]
        cosines = PositionContainer.return_cosines(normals=normals, line_of_sight=los)
        obtained = PositionContainer.darkside_filter(cosines)
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

        obtained = dynamic.calculate_spot_longitudes(MockBinaryInstance(), phases, correct_libration=False)
        obtained = np.round(np.array([obtained["primary"][0], obtained["secondary"][0]]), 5)
        assert_array_equal(expected, obtained)

    def test_surface_area_coverage_not_partial(self):
        size = 5
        visible = [0, 2]
        result_coverage = [10, 20]

        obtained = coverage.surface_area_coverage(size, visible, result_coverage)
        expected = np.array([10., 0., 20., 0., 0.])
        self.assertTrue(np.all(obtained == expected))

    def test_surface_area_coverage_partial(self):
        size = 5
        visible = [0, 2]
        partial = [1]
        result_coverage = [10, 20]
        partial_coverage = [30]

        obtained = coverage.surface_area_coverage(size, visible, result_coverage, partial, partial_coverage)
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

    def _test_find_apsidally_corresponding_positions(self, args, expected, tol=1e-5):
        obtained = dynamic.find_apsidally_corresponding_positions(*args)
        assert_allclose(expected.body, obtained.body, atol=tol)
        assert_allclose(expected.mirror, obtained.mirror, atol=tol)

    def prep_for_test(self, eccentricity, num=20, omega=90):
        params = deepcopy(PARAMS["eccentric"])
        params['eccentricity'] = eccentricity
        params['argument_of_periastron'] = omega
        phases = np.linspace(0, 1, num=num, endpoint=False)

        bs = prepare_binary_system(params)
        positions = bs.calculate_orbital_motion(phases, return_nparray=True, calculate_from='phase')
        potentials = bs.correct_potentials(phases, component="all", iterations=2)
        # calculating components radii for each orbital position
        radii = crv_utils.forward_radii_from_distances(bs, positions[:, 1], potentials)

        _, _, reduced_phase_mask = \
            crv_utils.prepare_apsidaly_symmetric_orbit(bs, positions[:, 2], phases)

        base_arr, supplement_arr = crv_utils.split_orbit_by_apse_line(positions, reduced_phase_mask)
        return bs, radii, base_arr, supplement_arr

    def test_find_apsidally_corresponding_positions_full_match(self):
        args = self.prep_for_test(eccentricity=0.001, num=6, omega=110)

        expected = OrbitalSupplements(
            np.array(
            [[1.0, 0.99923481, 2.61996529, 0.70010311, 0.16666667],
             [3.0, 1.00094004, 4.71375449, 2.79389232, 0.5],
             [2.0, 1.00017529, 3.66784394, 1.74798176, 0.33333333]]
        ),
            np.array(
                [[0., 0.99906019, 1.57079633, 5.93411946, 0.],
                 [4., 1.00076602, 5.7589847, 3.83912252, 0.66666667],
                 [5., 0.99982665, 0.52231253, 4.88563566, 0.83333333]]
        ))

        self._test_find_apsidally_corresponding_positions(args, expected)

    def test_find_apsidally_corresponding_positions_mixed_first_longer(self):
        args = self.prep_for_test(eccentricity=0.001, num=6, omega=90)

        expected = OrbitalSupplements(
            np.array(
                [[2.0, 1.00050075e+00, 3.66692240e+00, 2.09612607e+00, 3.33333333e-01],
                 [2.0, 1.00050075e+00, 3.66692240e+00, 2.09612607e+00, 3.33333333e-01],
                 [1.0, 9.99500751e-01, 2.61972701e+00, 1.04893068e+00, 1.66666667e-01],
                 [0.0, 9.99000000e-01, 1.57079633e+00, 9.63928419e-35, 0.00000000e+00]]
            ),
            np.array(
                [[3.0, 1.00100000, 4.71238898, 3.14159265, 0.5],
                 [4.0, 1.00050075, 5.75785556, 4.18705924, 0.66666667],
                 [5.0, 0.99950075, 0.52186564, 5.23425462, 0.83333333],
                 [np.nan, np.nan, np.nan, np.nan, np.nan]]
            ))

        self._test_find_apsidally_corresponding_positions(args, expected)

    def test_find_apsidally_corresponding_positions_total_mixed(self):
        args = self.prep_for_test(eccentricity=0.1, num=2, omega=90)
        expected = OrbitalSupplements(
            np.array(
                [[1.0, 1.1, 4.71238898, 3.14159265, 0.5],
                 [0.0, 0.9, 1.57079633, 0.00000000, 0.0]]
            ),
            np.array(
                [[np.nan, np.nan, np.nan, np.nan, np.nan],
                 [np.nan, np.nan, np.nan, np.nan, np.nan]]
            ))
        self._test_find_apsidally_corresponding_positions(args, expected)

    def test_find_apsidally_corresponding_positions_body_not_empty(self):
        args = self.prep_for_test(eccentricity=0.001)

        obtained = dynamic.find_apsidally_corresponding_positions(*args)
        self.assertTrue(np.all(~up.isnan(obtained.body)))

    def test_resolve_object_geometry_update(self):
        settings.configure(**{"MAX_RELATIVE_D_R_POINT": 0.1})
        rel_d_radii = np.array([
            [0.05, 0.04, 0.02, 0.01, 0.1, 1.1, 98, 0.00001],
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.12]
        ])
        expected = np.array([True, False, False, True, False, True, True, True, True], dtype=bool)
        obtained = dynamic.resolve_object_geometry_update(False, rel_d_radii.shape[1] + 1, rel_d_radii)
        self.assertTrue(np.all(expected == obtained))

    def test_get_visible_projection(self):
        obj = MockSelf()
        obj.faces = np.array([[0, 1, 2], [2, 3, 0]])
        obj.indices = np.array([0, 1])
        obj.points = np.array([[-1, -1, -2], [0., 1, 1], [1, 1, 2], [2, 3, 4]])

        obtained = utils.get_visible_projection(obj)
        expected = np.vstack((obj.points[:, 1], obj.points[:, 2])).T
        self.assertTrue(np.all(expected == obtained))

    def test_partial_visible_faces_surface_coverage(self):
        points = np.array([[-1, 0.5], [1, 0.5], [0, 1.5]])
        faces = np.array([[0, 1, 2]])
        normals = const.LINE_OF_SIGHT[0] * np.array([[1, -1, 0]]) / np.linalg.norm(np.array([-1, 1, 0]))
        hull = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])

        obtained = np.round(surface.coverage.partial_visible_faces_surface_coverage(points, faces, normals, hull), 10)
        expected = np.round(0.5 / up.cos(up.pi / 4.0), 10)
        self.assertTrue(np.all(obtained == expected))

    @staticmethod
    def test_resolve_spots_geometry_update():
        settings.configure(**{"MAX_SPOT_D_LONGITUDE": 0.06})
        spots_longitudes = {
            'primary': {
                0: np.array([0.5, 0.55, 0.6, 0.75, 0.76, 0.77]),
                1: np.array([0.5, 0.55, 0.6, 0.75, 0.76, 0.77]) - 0.1
            },
            'secondary': {}
        }

        pulsation_tests = {'primary': False, 'secondary': False}
        obtained = np.array(dynamic.resolve_spots_geometry_update(spots_longitudes, 6, pulsation_tests), dtype=bool)
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
        settings.configure(**{
            "LD_TABLES": op.join(self.lc_base_path, "limbdarkening"),
            "CK04_ATM_TABLES": op.join(self.lc_base_path, "atmosphere")
        })

        bs = prepare_binary_system(BINARY_SYSTEM_PARAMS['detached-physical'])
        from_this = dict(binary_system=bs, position=const.Position(0, 1.0, 0.0, 0.0, 0.0))
        system = OrbitalPositionContainer.from_binary_system(**from_this)
        system.build(components_distance=1.0)
        system.calculate_face_angles(line_of_sight=const.LINE_OF_SIGHT)
        system.apply_darkside_filter()
        self.assertTrue((not is_empty(system.primary.indices)) and (not is_empty(system.secondary.indices)))


class ComputeLightCurvesTestCase(ElisaTestCase):
    def setUp(self):
        super(ComputeLightCurvesTestCase, self).setUp()
        self.lc_base_path = op.join(op.dirname(op.abspath(__file__)), "data", "light_curves")
        settings.configure(**{
            "LD_TABLES": op.join(self.lc_base_path, "limbdarkening"),
            "CK04_ATM_TABLES": op.join(self.lc_base_path, "atmosphere")
        })

    def test_light_curve_pass_on_all_ld_law(self):
        """
        no assert here, it just has to pass without error
        """
        bs = prepare_binary_system(PARAMS["detached"])
        start_phs, stop_phs, step = -0.2, 1.2, 0.1

        # idea is to update configuration content for problematic values in default configuration file
        content_tempalte = "[physics]limb_darkening_law={ld_law}"

        for law in settings.LD_LAW_TO_FILE_PREFIX.keys():
            settings.configure(**{"LIMB_DARKENING_LAW": law})
            self.write_default_support(ld_tables=settings.LD_TABLES, atm_tables=settings.CK04_ATM_TABLES)
            with open(self.CONFIG_FILE, "a") as f:
                f.write(content_tempalte.format(ld_law=law))

            o = Observer(passband=['Generic.Bessell.V'], system=bs)
            o.lc(from_phase=start_phs, to_phase=stop_phs, phase_step=step)

    def do_comparison(self, system, filename, f_tol, start_phs, stop_phs, step):
        o = Observer(passband=['Generic.Bessell.V'], system=system)

        obtained = o.lc(from_phase=start_phs, to_phase=stop_phs, phase_step=step)
        obtained_phases = obtained[0]
        obtained_flux = normalize_lc_for_unittests(obtained[1]["Generic.Bessell.V"])

        expected = load_light_curve(filename)
        expected_phases = expected[0]
        expected_flux = normalize_lc_for_unittests(expected[1]["Generic.Bessell.V"])

        # import matplotlib.pyplot as plt
        # plt.plot(expected_phases, expected_flux, label='expected')
        # plt.plot(obtained_phases, obtained_flux, label='obtained')
        # plt.legend()
        # plt.show()

        self.assertTrue(np.all(up.abs(obtained_phases - expected_phases) < TOL))
        self.assertTrue(np.all(up.abs(obtained_flux - expected_flux) < f_tol))

    def test_circular_synchronous_detached_system(self):
        bs = prepare_binary_system(PARAMS["detached"])
        self.do_comparison(bs, "detached.circ.sync.generic.bessel.v.json", TOL, -0.2, 1.2, 0.01)

    def test_circular_synchronous_overcontact_system(self):
        bs = prepare_binary_system(PARAMS["over-contact"])
        self.do_comparison(bs, "overcontact.circ.sync.generic.bessel.v.json", TOL, -0.2, 1.2, 0.01)

    def test_eccentric_synchronous_system_no_approximation(self):
        settings.configure(**APPROX_SETTINGS["no_approx"])
        bs = prepare_binary_system(PARAMS["eccentric"])
        self.do_comparison(bs, "detached.ecc.sync.generic.bessell.v.json", TOL, -0.2, 1.2, 0.1)

    def test_eccentric_system_approximation_one(self):
        settings.configure(**APPROX_SETTINGS["approx_one"])
        bs = prepare_binary_system(PARAMS["eccentric"])
        self.do_comparison(bs, "detached.ecc.sync.generic.bessell.v.appx_one.json", TOL, 0.0, 1.0, 0.007)

    def test_eccentric_system_approximation_two(self):
        settings.configure(**APPROX_SETTINGS["approx_two"])
        ec_params = PARAMS["eccentric"].copy()
        ec_params["argument_of_periastron"] = 90 * u.deg
        bs = prepare_binary_system(ec_params)
        self.do_comparison(bs, "detached.ecc.sync.generic.bessell.v.appx_two.json", TOL, -0.2, 1.15, 0.05)

    def test_eccentric_system_approximation_three(self):
        settings.configure(**APPROX_SETTINGS["approx_three"])
        bs = prepare_binary_system(PARAMS["eccentric"])
        self.do_comparison(bs, "detached.ecc.sync.generic.bessell.v.appx_three.json", TOL, -0.0, 0.01, 0.002)

    def test_circular_spotty_synchronous_detached_system(self):
        bs = prepare_binary_system(PARAMS["detached"],
                                   spots_primary=SPOTS_META["primary"],
                                   spots_secondary=SPOTS_META["secondary"])
        self.do_comparison(bs, "detached.circ.spotty.sync.generic.bessel.v.json", TOL, -0.2, 1.2, 0.01)

    def test_circular_spotty_synchronous_overcontact_system(self):
        bs = prepare_binary_system(PARAMS["over-contact"],
                                   spots_primary=SPOTS_META["primary"],
                                   spots_secondary=SPOTS_META["secondary"])
        self.do_comparison(bs, "overcontact.circ.spotty.sync.generic.bessel.v.json", TOL, -0.2, 1.2, 0.01)

    def test_cicular_spotty_asynchronous_detached_system(self):
        settings.configure(**{"MAX_SPOT_D_LONGITUDE": up.pi / 45.0})
        bs = prepare_binary_system(PARAMS["detached-async"], spots_primary=SPOTS_META["primary"])
        self.do_comparison(bs, "detached.circ.spotty.async.generic.bessel.v.json", TOL, -0.2, 1.2, 0.2)

    def test_eccentric_spotty_asynchronous_detached_system(self):
        bs = prepare_binary_system(PARAMS["detached-async-ecc"], spots_primary=SPOTS_META["primary"])
        self.do_comparison(bs, "detached.ecc.spotty.async.generic.bessel.v.json", TOL, -0.2, 1.2, 0.1)


class CompareSingleVsMultiprocess(ElisaTestCase):
    def setUp(self):
        super(CompareSingleVsMultiprocess, self).setUp()
        self.lc_base_path = op.join(op.dirname(op.abspath(__file__)), "data", "light_curves")

        settings.configure(**{
            "LD_TABLES": op.join(self.lc_base_path, "limbdarkening"),
            "CK04_ATM_TABLES": op.join(self.lc_base_path, "atmosphere")
        })
        self.write_default_support(ld_tables=settings.LD_TABLES, atm_tables=settings.CK04_ATM_TABLES)

    def tearDown(self):
        super(CompareSingleVsMultiprocess, self).tearDown()

    def do_comparison(self, system, start_phs=-0.2, stop_phs=1.2, step=0.1, tol=1e-8):
        o = Observer(passband=['Generic.Bessell.V'], system=system)
        sp_res = o.lc(from_phase=start_phs, to_phase=stop_phs, phase_step=step)
        sp_flux = normalize_lc_for_unittests(sp_res[1]["Generic.Bessell.V"])\

        settings.configure(**{"NUMBER_OF_PROCESSES":  2})

        mp_res = o.lc(from_phase=start_phs, to_phase=stop_phs, phase_step=step)
        mp_flux = normalize_lc_for_unittests(mp_res[1]["Generic.Bessell.V"])

        # print(np.max(sp_flux - mp_flux))

        # import matplotlib.pyplot as plt
        # plt.plot(sp_res[0], sp_flux, label='single')
        # plt.plot(mp_res[0], mp_flux, label='multi')
        # plt.legend()
        # plt.show()

        self.assertTrue(np.all(sp_flux - mp_flux < tol))

    def test_circulcar_sync_lc(self):
        bs = prepare_binary_system(PARAMS["detached"])
        self.do_comparison(bs)

    def test_circulcar_spotty_async_lc(self):
        settings.configure(**{"MAX_SPOT_D_LONGITUDE": up.pi / 45.0})

        # write config for stupid windows multiprocessing
        with open(self.CONFIG_FILE, "a") as f:
            f.write(f"[computational]"
                    f"\nmax_spot_d_longitude={settings.MAX_SPOT_D_LONGITUDE}"
                    f"\n")

        bs = prepare_binary_system(PARAMS["detached-async"], spots_primary=SPOTS_META["primary"])
        self.do_comparison(bs)

    def test_eccentric_system_no_approximation(self):
        settings.configure(**APPROX_SETTINGS["no_approx"])
        with open(self.CONFIG_FILE, "a") as f:
            f.write(f"[computational]"
                    f"\nmax_nu_separation={settings.MAX_NU_SEPARATION}"
                    f"\nmax_relative_d_r_point={settings.MAX_RELATIVE_D_R_POINT}"
                    f"\n")
        bs = prepare_binary_system(PARAMS["eccentric"])
        self.do_comparison(bs)

    def test_eccentric_system_approximation_one(self):
        settings.configure(**APPROX_SETTINGS["approx_one"])
        with open(self.CONFIG_FILE, "a") as f:
            f.write(f"[computational]"
                    f"\nmax_nu_separation={settings.MAX_NU_SEPARATION}"
                    f"\nmax_relative_d_r_point={settings.MAX_RELATIVE_D_R_POINT}"
                    f"\n")
        bs = prepare_binary_system(PARAMS["eccentric"])
        self.do_comparison(bs, start_phs=-0.2, stop_phs=1.2, step=0.1)

    def test_eccentric_system_approximation_two(self):
        settings.configure(**APPROX_SETTINGS["approx_two"])
        settings.configure(**{"MAX_RELATIVE_D_R_POINT": 0.0})

        with open(self.CONFIG_FILE, "a") as f:
            f.write(f"[computational]"
                    f"\nmax_nu_separation={settings.MAX_NU_SEPARATION}"
                    f"\nmax_relative_d_r_point={settings.MAX_RELATIVE_D_R_POINT}"
                    f"\n")
        bs = prepare_binary_system(PARAMS["eccentric"])
        self.do_comparison(bs, start_phs=-0.2, stop_phs=1.2, step=0.1)

    def test_eccentric_system_approximation_three(self):
        settings.configure(**APPROX_SETTINGS["approx_three"])

        with open(self.CONFIG_FILE, "a") as f:
            f.write(f"[computational]"
                    f"\nmax_nu_separation={settings.MAX_NU_SEPARATION}"
                    f"\nmax_relative_d_r_point={settings.MAX_RELATIVE_D_R_POINT}")
        bs = prepare_binary_system(PARAMS["eccentric"])
        self.do_comparison(bs, 0.0, 0.01, 0.002, tol=1e-4)

    def test_eccentric_spotty_asynchronous_detached_system(self):
        bs = prepare_binary_system(PARAMS["detached-async-ecc"], spots_primary=SPOTS_META["primary"])
        self.do_comparison(bs)


class CompareApproxVsExact(ElisaTestCase):
    def setUp(self):
        super(CompareApproxVsExact, self).setUp()
        self.lc_base_path = op.join(op.dirname(op.abspath(__file__)), "data", "light_curves")
        settings.configure(**{
            "LD_TABLES": op.join(self.lc_base_path, "limbdarkening"),
            "CK04_ATM_TABLES": op.join(self.lc_base_path, "atmosphere")
        })

    @skip
    def test_approximation_one(self):
        bs = prepare_binary_system(PARAMS["eccentric"])

        phase_step = 1.0/150
        settings.configure(**APPROX_SETTINGS["approx_one"])
        settings.configure(**{"NUMBER_OF_PROCESSES": 4})
        o = Observer(passband=['Generic.Bessell.V'], system=bs)
        ap_res = o.lc(from_phase=-0.1, to_phase=1.1, phase_step=phase_step)
        ap_flux = normalize_lc_for_unittests(ap_res[1]["Generic.Bessell.V"])
        # ap_flux = ap_res[1]["Generic.Bessell.V"]

        settings.configure(**APPROX_SETTINGS["no_approx"])
        o = Observer(passband=['Generic.Bessell.V'], system=bs)
        ex_res = o.lc(from_phase=-0.1, to_phase=1.1, phase_step=phase_step)
        ex_flux = normalize_lc_for_unittests(ex_res[1]["Generic.Bessell.V"])
        # ex_flux = ex_res[1]["Generic.Bessell.V"]

        import matplotlib.pyplot as plt
        plt.plot(ex_res[0], ex_flux, label='exact')
        plt.plot(ap_res[0], ap_flux, label='approx')
        plt.plot(ap_res[0], ex_flux-ap_flux+1, label='diff')
        plt.legend()
        plt.show()

        self.assertTrue(np.all(ex_flux - ap_flux < 3e-3))

    @skip
    def test_approximation_two(self):
        bs = prepare_binary_system(PARAMS["eccentric"])

        settings.configure(**APPROX_SETTINGS["approx_two"])
        settings.configure(**{"NUMBER_OF_PROCESSES": 4})
        o = Observer(passband=['Generic.Bessell.V'], system=bs)
        ap_res = o.lc(from_phase=-0.1, to_phase=1.1, phase_step=0.01)
        ap_flux = normalize_lc_for_unittests(ap_res[1]["Generic.Bessell.V"])

        settings.configure(**APPROX_SETTINGS["no_approx"])
        o = Observer(passband=['Generic.Bessell.V'], system=bs)
        ex_res = o.lc(from_phase=-0.1, to_phase=1.1, phase_step=0.01)
        ex_flux = normalize_lc_for_unittests(ex_res[1]["Generic.Bessell.V"])

        import matplotlib.pyplot as plt
        plt.plot(ex_res[0], ex_flux, label='exact')
        plt.plot(ap_res[0], ap_flux, label='approx')
        plt.plot(ap_res[0], ex_flux-ap_flux+1, label='diff')
        plt.legend()
        plt.show()

        self.assertTrue(np.all(ex_flux - ap_flux < 3e-3))

    @skip
    def test_approximation_three(self):
        bs = prepare_binary_system(PARAMS["eccentric"])

        settings.configure(**APPROX_SETTINGS["approx_three"])
        settings.configure(**{"NUMBER_OF_PROCESSES": 4})
        o = Observer(passband=['Generic.Bessell.V'], system=bs)
        ap_res = o.lc(from_phase=-0.1, to_phase=0.6, phase_step=0.01)
        ap_flux = normalize_lc_for_unittests(ap_res[1]["Generic.Bessell.V"])
        # ap_flux = ap_res[1]["Generic.Bessell.V"]

        settings.configure(**APPROX_SETTINGS["no_approx"])
        o = Observer(passband=['Generic.Bessell.V'], system=bs)
        ex_res = o.lc(from_phase=-0.1, to_phase=0.6, phase_step=0.01)
        ex_flux = normalize_lc_for_unittests(ex_res[1]["Generic.Bessell.V"])
        # ex_flux = ex_res[1]["Generic.Bessell.V"]

        # import matplotlib.pyplot as plt
        # plt.plot(ex_res[0], ex_flux, label='exact')
        # plt.plot(ap_res[0], ap_flux, label='approx')
        # plt.plot(ap_res[0], ex_flux-ap_flux+1, label='diff')
        # plt.legend()
        # plt.show()

        self.assertTrue(np.all(ex_flux - ap_flux < 3e-3))
