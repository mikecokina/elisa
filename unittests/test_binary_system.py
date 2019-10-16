import os.path as op
import numpy as np

from unittest import skip
from copy import copy
from astropy import units as u
from mpl_toolkits.mplot3d import Axes3D
from numpy.testing import assert_array_equal

from elisa import const as c, umpy as up, units
from elisa.base.star import Star
from elisa.binary_system.system import BinarySystem
from elisa.conf import config
from elisa.utils import is_empty, find_nearest_dist_3d
from unittests.utils import ElisaTestCase, prepare_binary_system, plot_points, plot_faces, polar_gravity_acceleration

ax3 = Axes3D


class TestBinarySystemInit(ElisaTestCase):
    def setUp(self):
        self.params_combination = [
            {"primary_mass": 2.0, "secondary_mass": 1.0,
             "primary_surface_potential": 100.0, "secondary_surface_potential": 100.0,
             "primary_synchronicity": 1.0, "secondary_synchronicity": 1.0,
             "argument_of_periastron": c.HALF_PI * u.rad, "gamma": 0.0, "period": 2.0,
             "eccentricity": 0.0, "inclination": c.HALF_PI * u.rad, "primary_minimum_time": 0.0,
             "phase_shift": 0.0,
             "primary_t_eff": 5000, "secondary_t_eff": 5000,
             "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
             "primary_albedo": 0.6, "secondary_albedo": 0.6,
             },  # compact spherical components on circular orbit

            {"primary_mass": 2.0, "secondary_mass": 1.0,
             "primary_surface_potential": 100.0, "secondary_surface_potential": 80.0,
             "primary_synchronicity": 400, "secondary_synchronicity": 550,
             "argument_of_periastron": c.HALF_PI * u.rad, "gamma": 0.0, "period": 3.0,
             "eccentricity": 0.0, "inclination": 90.0 * units.deg, "primary_minimum_time": 0.0,
             "phase_shift": 0.0,
             "primary_t_eff": 5000, "secondary_t_eff": 5000,
             "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
             "primary_albedo": 0.6, "secondary_albedo": 0.6
             },  # rotationally squashed compact spherical components

            {"primary_mass": 2.0, "secondary_mass": 1.0,
             "primary_surface_potential": 3.5, "secondary_surface_potential": 3.0,
             "primary_synchronicity": 1.5, "secondary_synchronicity": 1.2,
             "argument_of_periastron": c.HALF_PI, "gamma": 0.0, "period": 1.0,
             "eccentricity": 0.0, "inclination": 90.0 * units.deg, "primary_minimum_time": 0.0,
             "phase_shift": 0.0,
             "primary_t_eff": 5000, "secondary_t_eff": 5000,
             "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
             "primary_albedo": 0.6, "secondary_albedo": 0.6
             },  # close tidally deformed components with asynchronous rotation on circular orbit

            {"primary_mass": 2.0, "secondary_mass": 1.0,
             "primary_surface_potential": 4.8, "secondary_surface_potential": 4.0,
             "primary_synchronicity": 1.5, "secondary_synchronicity": 1.2,
             "argument_of_periastron": c.HALF_PI * u.rad, "gamma": 0.0, "period": 1.0,
             "eccentricity": 0.3, "inclination": 90.0 * units.deg, "primary_minimum_time": 0.0,
             "phase_shift": 0.0,
             "primary_t_eff": 5000, "secondary_t_eff": 5000,
             "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
             "primary_albedo": 0.6, "secondary_albedo": 0.6
             },  # close tidally deformed components with asynchronous rotation on eccentric orbit

            {"primary_mass": 2.0, "secondary_mass": 1.0,
             "primary_surface_potential": 2.875844632141054,
             "secondary_surface_potential": 2.875844632141054,
             "primary_synchronicity": 1.0, "secondary_synchronicity": 1.0,
             "argument_of_periastron": c.HALF_PI * u.rad, "gamma": 0.0, "period": 1.0,
             "eccentricity": 0.0, "inclination": 90.0 * units.deg, "primary_minimum_time": 0.0,
             "phase_shift": 0.0,
             "primary_t_eff": 5000, "secondary_t_eff": 5000,
             "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
             "primary_albedo": 0.6, "secondary_albedo": 0.6
             },  # synchronous contact system

            {"primary_mass": 2.0, "secondary_mass": 1.0,
             "primary_surface_potential": 3.159639848886489,
             "secondary_surface_potential": 3.229240544834036,
             "primary_synchronicity": 1.5, "secondary_synchronicity": 2.0,
             "argument_of_periastron": c.HALF_PI * u.rad, "gamma": 0.0, "period": 1.0,
             "eccentricity": 0.0, "inclination": 90.0 * units.deg, "primary_minimum_time": 0.0,
             "phase_shift": 0.0,
             "primary_t_eff": 5000, "secondary_t_eff": 5000,
             "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
             "primary_albedo": 0.6, "secondary_albedo": 0.6
             },  # asynchronous contact system (improbable but whatever...)

            {"primary_mass": 2.0, "secondary_mass": 1.0,
             "primary_surface_potential": 2.7,
             "secondary_surface_potential": 2.7,
             "primary_synchronicity": 1.0, "secondary_synchronicity": 1.0,
             "argument_of_periastron": 90 * units.deg, "gamma": 0.0, "period": 1.0,
             "eccentricity": 0.0, "inclination": 90.0 * units.deg, "primary_minimum_time": 0.0,
             "phase_shift": 0.0,
             "primary_t_eff": 5000, "secondary_t_eff": 5000,
             "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
             "primary_albedo": 0.6, "secondary_albedo": 0.6
             }  # over-contact system
        ]

    def _prepare_systems(self):
        return [prepare_binary_system(combo) for combo in self.params_combination]

    def test_calculate_semi_major_axis(self):
        expected = [6702758048.0, 8783097736.0, 4222472978.0, 4222472978.0, 4222472978.0, 4222472978.0, 4222472978.0]
        obtained = list()

        for bs in self._prepare_systems():
            obtained.append(np.round(bs.semi_major_axis, 0))
        assert_array_equal(expected, obtained)

    def test__setup_periastron_critical_potential(self):
        expected_potentials = np.round(np.array([
            [2.875844632141054, 2.875844632141054],
            [93.717106763853593, 73.862399105365014],
            [3.159639848886489, 2.935086409515319],
            [4.027577786299736, 3.898140726941630],
            [2.875844632141054, 2.875844632141054],
            [3.159639848886489, 3.229240544834036],
            [2.875844632141054, 2.875844632141054]
        ]), 5)
        obtained_potentials = list()
        for bs in self._prepare_systems():
            obtained_potentials.append([bs.primary.critical_surface_potential, bs.secondary.critical_surface_potential])
        obtained_potentials = np.round(np.array(obtained_potentials), 5)
        assert_array_equal(expected_potentials, obtained_potentials)

    def test__setup_morphology(self):
        expected = ['detached', 'detached', 'detached', 'detached', 'semi-detached', 'double-contact', 'over-contact']
        obtained = list()

        for bs in self._prepare_systems():
            obtained.append(bs.morphology)
        assert_array_equal(expected, obtained)

    # fixme: change method to compute radii anytime necessary
    # def test_setup_components_radii(self):
    #     radii = ["forward_radius", "side_radius", "equatorial_radius", "backward_radius", "polar_radius"]
    #     obtained = dict()
    #
    #     expected = {
    #         "primary": {
    #             "forward_radius": [0.01005, 0.01229, 0.3783, 0.26257, 0.57075, 0.53029, np.nan],
    #             "side_radius": [0.01005, 0.01229, 0.35511, 0.24859, 0.43994, 0.41553, 0.48182],
    #             "equatorial_radius": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    #             "backward_radius": [0.01005, 0.01229, 0.36723, 0.25608, 0.46794, 0.44207, 0.52573],
    #             "polar_radius": [0.01005, 0.01005, 0.33055, 0.24242, 0.41427, 0.37162, 0.44577]
    #         },
    #         "secondary": {
    #             "forward_radius": [0.00506, 0.00763, 0.34966, 0.25693, 0.42925, 0.36743, np.nan],
    #             "side_radius": [0.00506, 0.00763, 0.29464, 0.21419, 0.31288, 0.28094, 0.35376],
    #             "equatorial_radius": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    #             "backward_radius": [0.00506, 0.00763, 0.32018, 0.23324, 0.34537, 0.30842, 0.41825],
    #             "polar_radius": [0.00506, 0.00635, 0.2798, 0.20994, 0.29977, 0.2489, 0.33306]
    #         }
    #     }
    #
    #     for component in ["primary", "secondary"]:
    #         obtained[component] = dict()
    #         for radius in radii:
    #             obtained[component][radius] = list()
    #             for i, bs in enumerate(self._prepare_systems()):
    #                 value = np.round(getattr(getattr(bs, component), radius), 5)
    #                 assert_array_equal([value], [expected[component][radius][i]])

    def test_lagrangian_points(self):

        expected_points = np.round(np.array([
            [-0.80302796065835425, 0.57075157151852673, 1.5823807222136623],
            [-0.80302796065835425, 0.57075157151852673, 1.5823807222136623],
            [-0.80302796065835425, 0.57075157151852673, 1.5823807222136623],
            [-0.7308068505479407, 0.41566688133312363, 1.4990376377419574],
            [-0.80302796065835425, 0.57075157151852673, 1.5823807222136623],
            [-0.80302796065835425, 0.57075157151852673, 1.5823807222136623],
            [-0.80302796065835425, 0.57075157151852673, 1.5823807222136623]
        ]), 5)

        obtained_points = []
        for bs in self._prepare_systems():
            obtained_points.append(bs.lagrangian_points())

        obtained_points = np.round(np.array(obtained_points), 5)
        assert_array_equal(expected_points, obtained_points)


class TestValidity(ElisaTestCase):
    MANDATORY_KWARGS = ['gamma', 'inclination', 'period', 'eccentricity', 'argument_of_periastron', 'phase_shift']

    def setUp(self):
        self._initial_params = {
            "primary_mass": 2.0, "secondary_mass": 1.0,
            "primary_surface_potential": 100.0, "secondary_surface_potential": 100.0,
            "primary_synchronicity": 1.0, "secondary_synchronicity": 1.0,
            "argument_of_periastron": c.HALF_PI * u.rad, "gamma": 0.0, "period": 1.0,
            "eccentricity": 0.0, "inclination": c.HALF_PI * units.deg, "primary_minimum_time": 0.0,
            "phase_shift": 0.0,
            "primary_t_eff": 5000, "secondary_t_eff": 5000,
            "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
            "primary_albedo": 1.0, "secondary_albedo": 1.0,
        }

        self._primary = Star(mass=self._initial_params["primary_mass"],
                             surface_potential=self._initial_params["primary_surface_potential"],
                             synchronicity=self._initial_params["primary_synchronicity"],
                             t_eff=self._initial_params["primary_t_eff"],
                             gravity_darkening=self._initial_params["primary_gravity_darkening"],
                             albedo=self._initial_params['primary_albedo'],
                             metallicity=0.0)

        self._secondary = Star(mass=self._initial_params["secondary_mass"],
                               surface_potential=self._initial_params["secondary_surface_potential"],
                               synchronicity=self._initial_params["secondary_synchronicity"],
                               t_eff=self._initial_params["secondary_t_eff"],
                               gravity_darkening=self._initial_params["secondary_gravity_darkening"],
                               albedo=self._initial_params['secondary_albedo'],
                               metallicity=0.0)

    def test__star_params_validity_check(self):
        with self.assertRaises(Exception) as context:
            BinarySystem(primary=69,
                         secondary=self._secondary,
                         argument_of_periastron=self._initial_params["argument_of_periastron"],
                         gamma=self._initial_params["gamma"],
                         period=self._initial_params["period"],
                         eccentricity=self._initial_params["eccentricity"],
                         inclination=self._initial_params["inclination"],
                         primary_minimum_time=self._initial_params["primary_minimum_time"],
                         phase_shift=self._initial_params["phase_shift"])

        self.assertTrue('Component `primary` is not instance of class' in str(context.exception))

        star_mandatory_kwargs = ['mass', 'surface_potential', 'synchronicity',
                                 'albedo', 'metallicity', 'gravity_darkening']

        for kw in star_mandatory_kwargs[:1]:
            p = copy(self._primary)
            setattr(p, f"{kw}", None)

            with self.assertRaises(Exception) as context:
                BinarySystem(primary=p,
                             secondary=self._secondary,
                             argument_of_periastron=self._initial_params["argument_of_periastron"],
                             gamma=self._initial_params["gamma"],
                             period=self._initial_params["period"],
                             eccentricity=self._initial_params["eccentricity"],
                             inclination=self._initial_params["inclination"],
                             primary_minimum_time=self._initial_params["primary_minimum_time"],
                             phase_shift=self._initial_params["phase_shift"])

            self.assertTrue(f'Mising argument(s): `{kw}`' in str(context.exception))

    def test_invalid_kwarg_checker(self):
        with self.assertRaises(Exception) as context:
            BinarySystem(primary=self._primary,
                         secondary=self._secondary,
                         argument_of_periastron=self._initial_params["argument_of_periastron"],
                         gamma=self._initial_params["gamma"],
                         period=self._initial_params["period"],
                         eccentricity=self._initial_params["eccentricity"],
                         inclination=self._initial_params["inclination"],
                         primary_minimum_time=self._initial_params["primary_minimum_time"],
                         phase_shift=self._initial_params["phase_shift"],
                         adhoc_param="xxx")
        self.assertTrue('Invalid keyword argument(s): adhoc_param' in str(context.exception))

    def test_check_missing_kwargs(self):
        initial_kwargs = dict(primary=self._primary,
                              secondary=self._secondary,
                              argument_of_periastron=self._initial_params["argument_of_periastron"],
                              gamma=self._initial_params["gamma"],
                              period=self._initial_params["period"],
                              eccentricity=self._initial_params["eccentricity"],
                              inclination=self._initial_params["inclination"],
                              primary_minimum_time=self._initial_params["primary_minimum_time"],
                              phase_shift=self._initial_params["phase_shift"])

        for kw in self.MANDATORY_KWARGS:
            kwargs = copy(initial_kwargs)
            del (kwargs[kw])
            with self.assertRaises(Exception) as context:
                BinarySystem(**kwargs)
            self.assertTrue(f'Missing argument(s): `{kw}`' in str(context.exception))


class TestMethods(ElisaTestCase):
    MANDATORY_KWARGS = ['gamma', 'inclination', 'period', 'eccentricity', 'argument_of_periastron',
                        'primary_minimum_time', 'phase_shift']
    OPTIONAL_KWARGS = []
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    def setUp(self):
        self.params_combination = [
            {"primary_mass": 2.0, "secondary_mass": 1.0,
             "primary_surface_potential": 100.0, "secondary_surface_potential": 100.0,
             "primary_synchronicity": 1.0, "secondary_synchronicity": 1.0,
             "argument_of_periastron": c.HALF_PI * u.rad, "gamma": 0.0, "period": 1.0,
             "eccentricity": 0.0, "inclination": c.HALF_PI * units.deg, "primary_minimum_time": 0.0,
             "phase_shift": 0.0,
             "primary_t_eff": 5000, "secondary_t_eff": 5000,
             "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
             "primary_albedo": 0.6, "secondary_albedo": 0.6,
             },
            # compact spherical components on circular orbit

            {"primary_mass": 2.0, "secondary_mass": 1.0,
             "primary_surface_potential": 4.8, "secondary_surface_potential": 4.0,
             "primary_synchronicity": 1.5, "secondary_synchronicity": 1.2,
             "argument_of_periastron": c.HALF_PI * u.rad, "gamma": 0.0, "period": 1.0,
             "eccentricity": 0.3, "inclination": 90.0 * units.deg, "primary_minimum_time": 0.0,
             "phase_shift": 0.0,
             "primary_t_eff": 5000, "secondary_t_eff": 5000,
             "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
             "primary_albedo": 0.6, "secondary_albedo": 0.6
             }  # close tidally deformed components with asynchronous rotation on eccentric orbit
        ]

        self._binaries = self._prepare_systems()

    def _prepare_systems(self):
        return [prepare_binary_system(combo) for combo in self.params_combination]

    def test__kwargs_serializer(self):
        bs = self._binaries[-1]
        obtained = bs._kwargs_serializer()

        expected = dict(
            gamma=0.0,
            inclination=c.HALF_PI,
            period=1.0,
            eccentricity=0.3,
            argument_of_periastron=c.HALF_PI,
            primary_minimum_time=0.0,
            phase_shift=0.0
        )

        obtained_array, expectedd_array = list(), list()

        for o_key, e_key in zip(obtained, expected):
            if isinstance(obtained[o_key], u.Quantity):
                obtained[o_key] = obtained[o_key].value
            obtained_array.append(round(obtained[o_key], 5))
            expectedd_array.append(round(expected[e_key], 5))

        assert_array_equal(expectedd_array, obtained_array)

    def test_primary_potential_derivative_x(self):
        d = 1.1,
        x = 0.13
        expected = np.round(np.array([-58.8584146731, -58.6146646731]), 4)
        obtained = list()

        for bs in self._binaries:
            obtained.append(bs.primary_potential_derivative_x(x, *d))
        obtained = np.round(obtained, 4)
        assert_array_equal(expected, obtained)

    def test_secondary_potential_derivative_x(self):
        d = 1.1,
        x = 0.13
        expected = np.round(np.array([-59.268745, -59.908945]), 4)
        obtained = list()

        for bs in self._binaries:
            obtained.append(bs.secondary_potential_derivative_x(x, *d))
        obtained = np.round(obtained, 4)
        assert_array_equal(expected, obtained)

    def test_pre_calculate_for_potential_value_primary(self):
        # single
        distance, phi, theta = 1.1, c.HALF_PI, c.HALF_PI / 2.0
        args = (distance, phi, theta)

        obtained = list()
        expected = [[1.21, 0., 0., 0.375],
                    [1.21, 0., 0., 0.8438]]

        for bs in self._binaries:
            obtained.append(bs.pre_calculate_for_potential_value_primary(*args))

        obtained = np.round(obtained, 4)
        assert_array_equal(expected, obtained)

    def test_potential_value_primary(self):
        # tested by critical_potential test
        pass

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
            gradient = bs.calculate_potential_gradient(component="primary", components_distance=distance, points=points)
            obtained.append(gradient)
        obtained = np.round(obtained, 4)

        assert_array_equal(expected, obtained)

    def test_calculate_potential_gradient_secondary(self):
        points = np.array([[0.1, 0.1, 0.1], [-0.1, 0.0, 0.3]])
        distance = 0.95

        expected = np.round(np.array(
            [
                [[17.60021, 19.17316, 19.32316], [-4.83097, 0., 9.60202]],
                [[17.65631, 19.10716, 19.32316], [-4.90027, 0., 9.60202]]
            ]), 4)
        obtained = list()

        for bs in self._binaries:
            gradient = bs.calculate_potential_gradient("secondary", distance, points)
            obtained.append(gradient)
        obtained = np.round(obtained, 4)

        assert_array_equal(expected, obtained)

    def test_calculate_polar_gravity_acceleration_circular(self):
        bs = self._binaries[0]
        expected_g_cgs_primary = polar_gravity_acceleration(bs, ["primary"], 1.0)
        expected_g_cgs_secondary = polar_gravity_acceleration(bs, ["secondary"], 1.0)

        obtained_g_cgs_primary = bs.calculate_polar_gravity_acceleration("primary", 1.0, logg=False) * 100
        obtained_g_cgs_secondary = bs.calculate_polar_gravity_acceleration("secondary", 1.0, logg=False) * 100

        self.assertEqual(round(expected_g_cgs_primary, 4), round(obtained_g_cgs_primary, 4))
        self.assertEqual(round(expected_g_cgs_secondary, 4), round(obtained_g_cgs_secondary, 4))

    @skip("requires attention - why this doesn't work for eccentric orbit")
    def test_calculate_polar_gravity_acceleration_eccentric(self):
        bs = self._binaries[1]
        distance = bs.orbit.orbital_motion([0.34])[0][0]

        expected_g_cgs_primary = polar_gravity_acceleration(bs, ["primary"], distance)
        expected_g_cgs_secondary = polar_gravity_acceleration(bs, ["secondary"], distance)

        obtained_g_cgs_primary = bs.calculate_polar_gravity_acceleration("primary", distance, logg=False) * 100
        obtained_g_cgs_secondary = bs.calculate_polar_gravity_acceleration("secondary", distance, logg=False) * 100

        print(expected_g_cgs_primary, obtained_g_cgs_primary)
        print(expected_g_cgs_secondary, obtained_g_cgs_secondary)
        print(distance)

        raise Exception("Unfinished test")

    def test_angular_velocity(self):
        expected = np.round([7.27220521664e-05, 4.64936429032e-05], 8)
        obtained = list()
        for bs in self._binaries:
            avcs = bs.angular_velocity(components_distance=bs.orbit.orbital_motion([0.35])[0][0])
            obtained.append(round(avcs, 8))
        assert_array_equal(expected, obtained)


class TestIntegrationNoSpots(ElisaTestCase):
    params = {
        "detached": {
            "primary_mass": 2.0, "secondary_mass": 1.0,
            "primary_surface_potential": 100.0, "secondary_surface_potential": 100.0,
            "primary_synchronicity": 1.0, "secondary_synchronicity": 1.0,
            "argument_of_periastron": c.HALF_PI * u.rad, "gamma": 0.0, "period": 1.0,
            "eccentricity": 0.0, "inclination": c.HALF_PI * units.deg, "primary_minimum_time": 0.0,
            "phase_shift": 0.0,
            "primary_t_eff": 5000, "secondary_t_eff": 5000,
            "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
            "primary_albedo": 0.6, "secondary_albedo": 0.6,
        },  # compact spherical components on circular orbit

        "detached.ecc": {
            "primary_mass": 2.0, "secondary_mass": 1.0,
            "primary_surface_potential": 4.8, "secondary_surface_potential": 4.0,
            "primary_synchronicity": 1.5, "secondary_synchronicity": 1.2,
            "argument_of_periastron": c.HALF_PI * u.rad, "gamma": 0.0, "period": 1.0,
            "eccentricity": 0.3, "inclination": 90.0 * units.deg, "primary_minimum_time": 0.0,
            "phase_shift": 0.0,
            "primary_t_eff": 5000, "secondary_t_eff": 5000,
            "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
            "primary_albedo": 0.6, "secondary_albedo": 0.6
        },  # close tidally deformed components with asynchronous rotation on eccentric orbit

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
        },  # over-contact system

        "semi-detached": {
            "primary_mass": 2.0, "secondary_mass": 1.0,
            "primary_surface_potential": 2.875844632141054,
            "secondary_surface_potential": 2.875844632141054,
            "primary_synchronicity": 1.0, "secondary_synchronicity": 1.0,
            "argument_of_periastron": c.HALF_PI * u.rad, "gamma": 0.0, "period": 1.0,
            "eccentricity": 0.0, "inclination": 90.0 * units.deg, "primary_minimum_time": 0.0,
            "phase_shift": 0.0,
            "primary_t_eff": 5000, "secondary_t_eff": 5000,
            "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
            "primary_albedo": 0.6, "secondary_albedo": 0.6
        }
    }

    def _test_build_mesh(self, _key, _d, _length, plot=False, single=False):
        s = prepare_binary_system(self.params[_key])
        s.primary.discretization_factor = _d
        s.secondary.discretization_factor = _d
        s.build_mesh(components_distance=1.0)

        obtained_primary = np.round(s.primary.points, 4)
        obtained_secondary = np.round(s.secondary.points, 4)

        if plot:
            if not single:
                plot_points(points_1=obtained_primary, points_2=obtained_secondary, label="obtained")
            else:
                plot_points(points_1=obtained_primary, points_2=[[]], label="obtained.primary")
                plot_points(points_1=obtained_secondary, points_2=[[]], label="obtained.secondary")

        assert_array_equal([len(obtained_primary), len(obtained_secondary)], _length)

    def test_build_mesh_detached(self):
        self._test_build_mesh(_key="detached", _d=up.radians(10), _length=[426, 426], plot=False, single=True)

    def test_build_mesh_overcontact(self):
        self._test_build_mesh(_key="over-contact", _d=up.radians(10), _length=[413, 401], plot=False, single=True)

    def test_build_mesh_semi_detached(self):
        self._test_build_mesh(_key="semi-detached", _d=up.radians(10), _length=[426, 426], plot=False, single=True)

    def _test_build_faces(self, _key, _d, _max_s=10.0, plot=False):
        s = prepare_binary_system(self.params[_key])
        s.primary.discretization_factor = _d
        s.secondary.discretization_factor = _d
        s.build_mesh(components_distance=1.0)
        s.build_faces(components_distance=1.0)

        obtained_primary_f = np.round(s.primary.faces, 4)
        obtained_secondary_f = np.round(s.secondary.faces, 4)

        if plot:
            if _key in ["over-contact"]:
                obtained_p = up.concatenate((s.primary.points, s.secondary.points))
                obtained_f = up.concatenate((obtained_secondary_f, obtained_secondary_f + obtained_primary_f.max() + 1))

                plot_faces(obtained_p, obtained_f, label="obtained.overcontact")

            else:
                plot_faces(s.primary.points, obtained_primary_f, label="obtained.primary")
                plot_faces(s.secondary.points, obtained_secondary_f, label="obtained.secondary")

        s.build_surface_areas()
        self.assertTrue(s.primary.areas.max() < _max_s)
        self.assertTrue(s.secondary.areas.max() < _max_s)

    def test_build_faces_detached(self):
        self._test_build_faces("detached", up.radians(10), _max_s=2e-6, plot=False)

    def test_build_faces_semi_detached(self):
        self._test_build_faces("semi-detached", up.radians(10), _max_s=6e-3, plot=False)

    def test_build_faces_overcontact(self):
        self._test_build_faces("over-contact", up.radians(10), _max_s=7e-3, plot=False)


class TestIntegrationWithSpots(ElisaTestCase):
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

    spots_to_rasie = [
        {"longitude": 60,
         "latitude": 45,
         "angular_radius": 28,
         "temperature_factor": 0.1},
    ]

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

    def test_spots_are_presented_after_mesh_build_in_detached(self):
        s = prepare_binary_system(self.params["detached"],
                                  spots_primary=self.spots_metadata["primary"],
                                  spots_secondary=self.spots_metadata["secondary"])
        s.build_mesh(components_distance=1.0)
        self.assertTrue(len(s.primary.spots) == 1 and len(s.secondary.spots) == 1)

    def test_spots_are_presented_after_mesh_build_in_overcontact(self):
        s = prepare_binary_system(self.params["over-contact"],
                                  spots_primary=self.spots_metadata["primary"],
                                  spots_secondary=self.spots_metadata["secondary"])
        s.build_mesh(components_distance=1.0)
        self.assertTrue(len(s.primary.spots) == 1 and len(s.secondary.spots) == 1)

    def test_spots_contain_all_params_after_build(self):
        s = prepare_binary_system(self.params["detached"],
                                  spots_primary=self.spots_metadata["primary"],
                                  spots_secondary=self.spots_metadata["secondary"])
        s.build(components_distance=1.0)

        self.assertTrue(np.all(np.round(s.primary.spots[0].log_g, 0) == 2.0))
        self.assertFalse(is_empty(s.primary.spots[0].points))
        self.assertFalse(is_empty(s.primary.spots[0].faces))
        self.assertFalse(is_empty(s.primary.spots[0].areas))
        self.assertFalse(is_empty(s.primary.spots[0].normals))
        self.assertTrue(np.all(np.round(s.primary.spots[0].temperatures, -2) == 4800.0))

        self.assertTrue(np.all(np.round(s.secondary.spots[0].log_g, 0) == 2.0))
        self.assertFalse(is_empty(s.secondary.spots[0].points))
        self.assertFalse(is_empty(s.secondary.spots[0].faces))
        self.assertFalse(is_empty(s.secondary.spots[0].areas))
        self.assertFalse(is_empty(s.secondary.spots[0].normals))
        self.assertTrue(np.all(np.round(s.secondary.spots[0].temperatures, -2) == 4500.0))

    def test_raise_valueerror_due_to_limb_darkening(self):
        s = prepare_binary_system(self.params["over-contact"], spots_secondary=self.spots_to_rasie)
        with self.assertRaises(ValueError) as context:
            s.build(components_distance=1.0)
        self.assertTrue('interpolation lead to np.nan' in str(context.exception))

    @skip
    def test_mesh_for_duplicate_points(self):
        for params in self.params.values():
            bs = prepare_binary_system(params)
            components_distance = bs.orbit.orbital_motion(phase=0.0)[0][0]

            bs.build_mesh(components_distance=components_distance)

            distance1 = round(find_nearest_dist_3d(list(bs.primary.points)), 10)
            distance2 = round(find_nearest_dist_3d(list(bs.secondary.points)), 10)

            if distance1 < 1e-10:
                bad_points = []
                for i, point in enumerate(bs.primary.points):
                    for j, xx in enumerate(bs.primary.points[i+1:]):
                        dist = np.linalg.norm(point-xx)
                        if dist <= 1e-10:
                            print(f'Match: {point}, {i}, {j}')
                            bad_points.append(point)

            self.assertFalse(distance1 < 1e-10)
            self.assertFalse(distance2 < 1e-10)
