import os.path as op

import numpy as np
from astropy import units as u
from numpy.testing import assert_array_equal
from pypex.poly2d import polygon

from elisa import const as c, umpy as up, const
from elisa.base.star import Star
from elisa.binary_system import geo, utils as bsutils, static
from elisa.binary_system.system import BinarySystem
from elisa.conf import config
from elisa.utils import is_empty
from unittests.utils import ElisaTestCase


class SupportMethodsTestCase(ElisaTestCase):
    def test_compute_filling_factor(self):
        potential, l_points = 100.0, [2.4078, 2.8758, 2.5772]
        expected = -325.2652
        obtained = round(static.compute_filling_factor(potential, l_points), 4)
        self.assertEqual(expected, obtained)

    def test_darkside_filter(self):
        normals = np.array([[1, 1, 1], [0.3, 0.1, -5], [-2, -3, -4.1]])
        normals = np.array([a / b for a, b in zip(normals, np.linalg.norm(normals, axis=1))])
        los = [1, 0, 0]
        obtained = geo.darkside_filter(los, normals)
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

        # class MockStarNoSpot(object):
        #     def __init__(self, s):
        #         self.spots = {}
        #         self.synchronicity = s

        class MockBinaryInstance(object):
            def __init__(self):
                self.primary = MockStar(1.1)
                self.secondary = MockStar(0.9)

        phases = np.array([-0.9, 0.0, 0.3, 1.2])

        expected = np.round(np.array(
            [np.array([0.55751332, 1.123, 1.31149556, 1.87698224]),
             np.array([1.68848668, 1.123, 0.93450444, 0.36901776])]
        ), 5)

        obtained = geo.calculate_spot_longitudes(MockBinaryInstance(), phases)
        obtained = np.round(np.array([obtained["primary"][0], obtained["secondary"][0]]), 5)
        assert_array_equal(expected, obtained)

    def test_surface_area_coverage_not_partial(self):
        size = 5
        visible = [0, 2]
        coverage = [10, 20]

        obtained = geo.surface_area_coverage(size, visible, coverage)
        expected = np.array([10., 0., 20., 0., 0.])
        self.assertTrue(np.all(obtained == expected))

    def test_surface_area_coverage_partial(self):
        size = 5
        visible = [0, 2]
        partial = [1]
        coverage = [10, 20]
        partial_coverage = [30]

        obtained = geo.surface_area_coverage(size, visible, coverage, partial, partial_coverage)
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


class SystemOrbitalPositionTestCase(ElisaTestCase):
    orbital_motion = [
        c.BINARY_POSITION_PLACEHOLDER(1, 1.0, const.PI / 2.0, 0.0, 0.0),
        c.BINARY_POSITION_PLACEHOLDER(2, 1.0, c.FULL_ARC * (3. / 4.), const.PI, 0.0)
    ]

    params_combination = {
        "primary_mass": 2.0, "secondary_mass": 1.0,
        "primary_surface_potential": 10.0, "secondary_surface_potential": 10.0,
        "primary_synchronicity": 1.0, "secondary_synchronicity": 1.0,
        "argument_of_periastron": c.HALF_PI * u.rad, "gamma": 0.0, "period": 2.0,
        "eccentricity": 0.0, "inclination": c.HALF_PI * u.rad, "primary_minimum_time": 0.0,
        "phase_shift": 0.0,
        "primary_t_eff": 5000, "secondary_t_eff": 5000,
        "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
        "primary_albedo": 0.6, "secondary_albedo": 0.6,
    }

    def setUp(self):
        path = op.join(op.dirname(op.abspath(__file__)), "data", "light_curves")
        config.VAN_HAMME_LD_TABLES = op.join(path, "limbdarkening")
        config._update_atlas_to_base_dir()

    def _prepare_system(self):
        combo = self.params_combination
        primary = Star(mass=combo["primary_mass"], surface_potential=combo["primary_surface_potential"],
                       synchronicity=combo["primary_synchronicity"],
                       t_eff=combo["primary_t_eff"], gravity_darkening=combo["primary_gravity_darkening"],
                       albedo=combo['primary_albedo'], metallicity=0.0)

        secondary = Star(mass=combo["secondary_mass"], surface_potential=combo["secondary_surface_potential"],
                         synchronicity=combo["secondary_synchronicity"],
                         t_eff=combo["secondary_t_eff"], gravity_darkening=combo["secondary_gravity_darkening"],
                         albedo=combo['secondary_albedo'], metallicity=0.0)

        primary.discretization_factor, secondary.discretization_factor = up.radians(10), up.radians(10)

        return BinarySystem(primary=primary,
                            secondary=secondary,
                            argument_of_periastron=combo["argument_of_periastron"],
                            gamma=combo["gamma"],
                            period=combo["period"],
                            eccentricity=combo["eccentricity"],
                            inclination=combo["inclination"],
                            primary_minimum_time=combo["primary_minimum_time"],
                            phase_shift=combo["phase_shift"])

    def test_initialize_SystemOrbitalPosition(self):
        bs = self._prepare_system()
        bs.build(components_distance=1.0)
        position = geo.SystemOrbitalPosition(bs.primary, bs.secondary, const.PI / 2.0, self.orbital_motion)
        k_attrs = ["points", "normals", "faces", "face_centres", "log_g", "temperatures"]

        for component in config.BINARY_COUNTERPARTS:
            c_instance = getattr(position.init_data, component)
            for k_attr in k_attrs:
                attr = getattr(c_instance, k_attr)
                self.assertTrue(np.any(attr))

            self.assertTrue(is_empty(getattr(c_instance, "coverage")))
            self.assertTrue(is_empty(getattr(c_instance, "indices")))

    def test_iterate_SystemOrbitalPosition_and_init_data_persists(self):

        def serialize_geo_attrs():
            attrs = list()
            for component in config.BINARY_COUNTERPARTS:
                c_instance = getattr(position.init_data, component)
                for k_attr in ["points", "normals", "faces", "face_centres"]:
                    attr = getattr(c_instance, k_attr)
                    attrs.append(attr)
            return attrs

        bs = self._prepare_system()
        bs.build(components_distance=1.0)

        position = geo.SystemOrbitalPosition(bs.primary, bs.secondary, const.PI / 2.0, self.orbital_motion)

        initial_data = serialize_geo_attrs()

        for _ in position:
            current_initial_data = serialize_geo_attrs()
            for a, b in zip(initial_data, current_initial_data):
                assert_array_equal(a, b)

    def test_SystemOrbitalPosition_iteration(self):
        bs = self._prepare_system()
        bs.build(components_distance=1.0)

        test_values = np.array([[1.0, 0, 0], [-2, 1, 1.5]])

        bs.primary.points = test_values.copy()
        bs.primary.normals = test_values.copy()
        bs.secondary.points = test_values.copy()
        bs.secondary.normals = test_values.copy()

        position = geo.SystemOrbitalPosition(bs.primary, bs.secondary, const.PI / 4.0, self.orbital_motion)

        expected = [np.array([[0.7071, 0., -0.7071],
                              [-0.3536, 1., 2.4749]]),
                    np.array([[-0.7071, 0., 0.7071],
                              [2.4749, -1., -0.3536]])]

        for i, p in enumerate(position):
            assert_array_equal(np.round(p.primary.points, 4), expected[i])
            assert_array_equal(np.round(p.secondary.points, 4), expected[i])

    def test_visible_indices_when_darkside_filter_apply(self):
        bs = self._prepare_system()
        bs.build(components_distance=1.0)

        orbital_motion = [c.BINARY_POSITION_PLACEHOLDER(1, 1.0, const.PI / 2.0, const.PI, 0.0)]
        position = geo.SystemOrbitalPosition(bs.primary, bs.secondary, const.PI / 2.0, orbital_motion)
        position = position.darkside_filter()
        nxt = next(iter(position))

        self.assertTrue((not is_empty(nxt.primary.indices)) and (not is_empty(nxt.secondary.indices)))
