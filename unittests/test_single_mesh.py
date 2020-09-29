import numpy as np

from numpy.testing import assert_array_equal

from unittests import utils as testutils
from unittests.utils import ElisaTestCase, prepare_single_system

from elisa.single_system import model
from elisa.utils import is_empty, find_nearest_dist_3d
from elisa import umpy as up, units as u


class BuildMeshSpotsFreeTestCase(ElisaTestCase):
    def test_mesh_for_duplicate_points(self):
        for params in testutils.SINGLE_SYSTEM_PARAMS.values():
            s = prepare_single_system(params)
            # reducing runtime of the test
            s.star.discretization_factor = up.radians(7)
            s.init()

            system_container = testutils.prepare_single_system_container(s)

            system_container.build_mesh()

            star_instance = getattr(system_container, 'star')
            distance = round(find_nearest_dist_3d(list(star_instance.points)), 10)

            if distance < 1e-10:
                bad_points = []
                for i, point in enumerate(star_instance.points):
                    for j, xx in enumerate(star_instance.points[i + 1:]):
                        dist = np.linalg.norm(point - xx)
                        if dist <= 1e-10:
                            print(f'Match: {point}, {i}, {j}')
                            bad_points.append(point)

            self.assertFalse(distance < 1e-10)


class BuildSpottyMeshTestCase(ElisaTestCase):
    def generator_test_mesh(self, key, d):
        s = prepare_single_system(testutils.SINGLE_SYSTEM_PARAMS[key],
                                  spots=testutils.SPOTS_META["primary"])
        s.star.discretization_factor = d
        system_container = testutils.prepare_single_system_container(s)
        system_container.build_mesh()

        self.assertTrue(len(system_container.star.spots) == 1)
        self.assertTrue(not is_empty(system_container.star.spots[0].points))

    def test_build_mesh(self):
        for key in testutils.SINGLE_SYSTEM_PARAMS.keys():
            self.generator_test_mesh(key=key, d=up.radians(5))

    @staticmethod
    def test_build_mesh_detached_with_overlapped_like_umbra():
        s = prepare_single_system(testutils.SINGLE_SYSTEM_PARAMS['spherical'],
                                  spots=list(reversed(testutils.SPOTS_OVERLAPPED)))
        s.star.discretization_factor = up.radians(5)
        system_container = testutils.prepare_single_system_container(s)
        system_container.build_mesh()


class MeshUtilsTestCase(ElisaTestCase):
    def setUp(self):
        super(MeshUtilsTestCase, self).setUp()
        self.params_combination = [
            # solar model
            {
                "mass": 1.0,
                "t_eff": 5772 * u.K,
                "gravity_darkening": 0.32,
                "polar_log_g": 4.43775,
                "gamma": 0.0,
                "inclination": 90.0 * u.deg,
                "rotation_period": 25.38 * u.d,
            },
        ]

        self._singles = self.prepare_systems()

    def prepare_systems(self):
        return [prepare_single_system(combo) for combo in self.params_combination]

    def test_surface_potential_from_polar_log_g(self):
        expected = np.round(np.array([-1.90691272573e+12]), -7)
        obtained = list()

        for ss in self._singles:
            obtained.append(model.surface_potential_from_polar_log_g(ss.star.polar_log_g, ss.star.mass))

        obtained = np.round(np.array(obtained), -7)
        assert_array_equal(expected, obtained)

    def test_potential(self):
        expected = np.round(np.array([-190761017680.0]), -7)
        obtained = list()
        radii = [695700000.0]

        for ii, ss in enumerate(self._singles):
            p_args = (ss.star.mass, ss.angular_velocity, 0.0)
            args = model.pre_calculate_for_potential_value(*p_args)
            obtained.append(model.potential(radii[ii], *args))

        obtained = np.round(np.array(obtained), -7)
        assert_array_equal(expected, obtained)

    def test_radial_potential_derivative(self):
        expected = np.round(np.array([274.200111657]), 3)  # this value should be always true for solar model since it
        # is surface gravity acceleration
        obtained = list()
        radii = [695700000.0]

        for ii, ss in enumerate(self._singles):
            p_args = (ss.star.mass, ss.angular_velocity, 0.0)
            args = (model.pre_calculate_for_potential_value(*p_args),)
            obtained.append(model.radial_potential_derivative(radii[ii], *args))

        obtained = np.round(np.array(obtained), 3)
        assert_array_equal(expected, obtained)

