import numpy as np

from numpy.testing import assert_array_equal
from elisa.binary_system.surface import gravity
from elisa.utils import is_empty
from unittests import utils as testutils
from unittests.utils import ElisaTestCase, prepare_binary_system, polar_gravity_acceleration
from elisa import umpy as up, units as u, const


class BuildSpotlessGravityTestCase(ElisaTestCase):
    def generator_test_gravity(self, key, over):
        s = testutils.prepare_binary_system(testutils.BINARY_SYSTEM_PARAMS[key])
        s.primary.discretization_factor = up.radians(10)
        s.secondary.discretization_factor = up.radians(10)

        orbital_position_container = testutils.prepare_orbital_position_container(s)
        orbital_position_container.build_mesh(components_distance=1.0)
        orbital_position_container.build_faces(components_distance=1.0)
        orbital_position_container.build_surface_areas()
        orbital_position_container.build_faces_orientation(components_distance=1.0)
        orbital_position_container.build_surface_gravity(components_distance=1.0)

        self.assertTrue(np.all(orbital_position_container.primary.log_g > over))
        self.assertTrue(np.all(orbital_position_container.secondary.log_g > over))

    def test_build_gravity_detached(self):
        self.generator_test_gravity('detached', over=0.0)

    def test_build_gravity_semi_detached(self):
        self.generator_test_gravity('semi-detached', over=0.0)

    def test_build_gravity_overcontact(self):
        self.generator_test_gravity('over-contact', over=0.0)


class BuildSpotGravityTestCase(ElisaTestCase):
    def generator_test_gravity(self, key):
        s = testutils.prepare_binary_system(testutils.BINARY_SYSTEM_PARAMS[key],
                                            spots_primary=testutils.SPOTS_META["primary"],
                                            spots_secondary=testutils.SPOTS_META["secondary"])
        s.primary.discretization_factor = up.radians(10)
        s.secondary.discretization_factor = up.radians(10)
        orbital_position_container = testutils.prepare_orbital_position_container(s)
        orbital_position_container.build_mesh(components_distance=1.0)
        orbital_position_container.build_faces(components_distance=1.0)
        orbital_position_container.build_surface_areas()
        orbital_position_container.build_faces_orientation(components_distance=1.0)
        orbital_position_container.build_surface_gravity(components_distance=1.0)

        self.assertTrue(hasattr(orbital_position_container.primary.spots[0], "potential_gradient_magnitudes"))
        self.assertTrue(hasattr(orbital_position_container.secondary.spots[0], "potential_gradient_magnitudes"))

        self.assertTrue(hasattr(orbital_position_container.primary.spots[0], "log_g"))
        self.assertTrue(hasattr(orbital_position_container.secondary.spots[0], "log_g"))

        self.assertTrue(not is_empty(orbital_position_container.primary.spots[0].log_g))
        self.assertTrue(not is_empty(orbital_position_container.secondary.spots[0].log_g))

    def test_build_gravity_detached(self):
        self.generator_test_gravity('detached')

    def test_build_gravity_semi_detached(self):
        self.generator_test_gravity('semi-detached')

    def test_build_gravity_overcontact(self):
        self.generator_test_gravity('over-contact')


class GravityUtilsTestCase(ElisaTestCase):
    def setUp(self):
        super(GravityUtilsTestCase, self).setUp()
        self.params_combination = [
            {"primary_mass": 2.0, "secondary_mass": 1.0,
             "primary_surface_potential": 100.0, "secondary_surface_potential": 100.0,
             "primary_synchronicity": 1.0, "secondary_synchronicity": 1.0,
             "argument_of_periastron": const.HALF_PI * u.rad, "gamma": 0.0, "period": 1.0,
             "eccentricity": 0.0, "inclination": const.HALF_PI * u.deg, "primary_minimum_time": 0.0,
             "phase_shift": 0.0,
             "primary_t_eff": 5000, "secondary_t_eff": 5000,
             "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
             "primary_albedo": 0.6, "secondary_albedo": 0.6,
             },
            # compact spherical components on circular orbit

            {"primary_mass": 2.0, "secondary_mass": 1.0,
             "primary_surface_potential": 4.8, "secondary_surface_potential": 4.0,
             "primary_synchronicity": 1.5, "secondary_synchronicity": 1.2,
             "argument_of_periastron": const.HALF_PI * u.rad, "gamma": 0.0, "period": 1.0,
             "eccentricity": 0.3, "inclination": 90.0 * u.deg, "primary_minimum_time": 0.0,
             "phase_shift": 0.0,
             "primary_t_eff": 5000, "secondary_t_eff": 5000,
             "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
             "primary_albedo": 0.6, "secondary_albedo": 0.6
             }  # close tidally deformed components with asynchronous rotation on eccentric orbit
        ]

        self._binaries = self.prepare_systems()

    def prepare_systems(self):
        return [prepare_binary_system(combo) for combo in self.params_combination]

    def test_calculate_potential_gradient_primary(self):
        points = np.array([[0.1, 0.1, 0.1], [-0.1, 0.0, 0.3]])
        distance = 0.95

        expected = np.round(np.array(
            [
                [[18.9847562, 19.17315831, 19.32315831], [-2.86141573, 0., 9.60202004]],
                [[18.7972562, 18.98565831, 19.32315831], [-2.67391573, 0., 9.60202004]]
            ]), 4)
        obtained = list()

        for bs in self._binaries:
            gradient = gravity.calculate_potential_gradient(distance, "primary", points,
                                                            bs.primary.synchronicity, bs.mass_ratio)
            obtained.append(gradient)
        obtained = np.round(obtained, 4)
        assert_array_equal(expected, obtained)

    def test_calculate_potential_gradient_secondary(self):
        points = np.array([[0.1, 0.1, 0.1], [-0.1, 0.0, 0.3]])
        distance = 0.95
        expected = np.round(np.array(
            [
                [[18.74770634, 19.17315831, 19.32315831], [-3.0984656,   0.,          9.60202004]],
                [[19.30870634, 19.10715831, 19.3231583], [-2.4054656,   0.,          9.60202004]]
            ]), 4)
        obtained = list()
        for bs in self._binaries:
            gradient = gravity.calculate_potential_gradient(distance, "secondary", points,
                                                            bs.secondary.synchronicity, bs.mass_ratio)
            obtained.append(gradient)
        obtained = np.round(obtained, 4)
        assert_array_equal(expected, obtained)

    def test_calculate_polar_gravity_acceleration_circular(self):
        bs = self._binaries[0]
        orbital_position_container = testutils.prepare_orbital_position_container(bs)

        expected_g_cgs_primary = polar_gravity_acceleration(bs, ["primary"], 1.0)
        expected_g_cgs_secondary = polar_gravity_acceleration(bs, ["secondary"], 1.0)

        obtained_g_cgs_primary = gravity.calculate_polar_gravity_acceleration(orbital_position_container.primary,
                                                                              1.0, bs.mass_ratio, "primary",
                                                                              bs.semi_major_axis,
                                                                              bs.primary.synchronicity,
                                                                              logg=False) * 100
        obtained_g_cgs_secondary = gravity.calculate_polar_gravity_acceleration(orbital_position_container.secondary,
                                                                                1.0, bs.mass_ratio, "secondary",
                                                                                bs.semi_major_axis,
                                                                                bs.secondary.synchronicity,
                                                                                logg=False) * 100

        self.assertEqual(round(expected_g_cgs_primary, 4), round(obtained_g_cgs_primary, 4))
        self.assertEqual(round(expected_g_cgs_secondary, 4), round(obtained_g_cgs_secondary, 4))

    def test_calculate_polar_gravity_acceleration_eccentric(self):
        bs = self._binaries[1]
        distance = bs.orbit.orbital_motion([0.34])[0][0]
        orbital_position_container = testutils.prepare_orbital_position_container(bs)

        expected_g_cgs_primary = polar_gravity_acceleration(bs, ["primary"], distance)
        expected_g_cgs_secondary = polar_gravity_acceleration(bs, ["secondary"], distance)

        obtained_g_cgs_primary = gravity.calculate_polar_gravity_acceleration(orbital_position_container.primary,
                                                                              distance, bs.mass_ratio,
                                                                              "primary", bs.semi_major_axis,
                                                                              bs.primary.synchronicity,
                                                                              logg=False) * 100.0
        obtained_g_cgs_secondary = gravity.calculate_polar_gravity_acceleration(orbital_position_container.secondary,
                                                                                distance, bs.mass_ratio,
                                                                                "secondary", bs.semi_major_axis,
                                                                                bs.secondary.synchronicity,
                                                                                logg=False) * 100.0

        print(expected_g_cgs_primary, obtained_g_cgs_primary)
        print(expected_g_cgs_secondary, obtained_g_cgs_secondary)
        print(distance)

        # raise Exception("Unfinished test")
