# import random
# import sys
import unittest
from copy import copy

import numpy as np
from astropy import units as u
# from os.path import dirname
# from os.path import join as pjoin
#
# import numpy as np
# import pandas as pd
from numpy.testing import assert_array_equal

from elisa import const as c, const
from elisa.base.star import Star
from elisa.binary_system.system import BinarySystem


class TestBinarySystemSetters(unittest.TestCase):
    MANDATORY_KWARGS = ['gamma', 'inclination', 'period', 'eccentricity', 'argument_of_periastron',
                        'primary_minimum_time', 'phase_shift']

    def setUp(self):
        combo = {
            "primary_mass": 2.0, "secondary_mass": 1.0,
            "primary_surface_potential": 100.0, "secondary_surface_potential": 100.0,
            "primary_synchronicity": 1.0, "secondary_synchronicity": 1.0,
            "argument_of_periastron": c.HALF_PI * u.rad, "gamma": 0.0, "period": 1.0,
            "eccentricity": 0.0, "inclination": c.HALF_PI * u.deg, "primary_minimum_time": 0.0,
            "phase_shift": 0.0,
            "primary_t_eff": 5000, "secondary_t_eff": 5000,
            "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
            "primary_albedo": 1.0, "secondary_albedo": 1.0,
        }

        primary = Star(mass=combo["primary_mass"], surface_potential=combo["primary_surface_potential"],
                       synchronicity=combo["primary_synchronicity"],
                       t_eff=combo["primary_t_eff"], gravity_darkening=combo["primary_gravity_darkening"],
                       albedo=combo['primary_albedo'], metallicity=0.0)

        secondary = Star(mass=combo["secondary_mass"], surface_potential=combo["secondary_surface_potential"],
                         synchronicity=combo["secondary_synchronicity"],
                         t_eff=combo["secondary_t_eff"], gravity_darkening=combo["secondary_gravity_darkening"],
                         albedo=combo['secondary_albedo'], metallicity=0.0)

        self._binary = BinarySystem(primary=primary,
                                    secondary=secondary,
                                    argument_of_periastron=combo["argument_of_periastron"],
                                    gamma=combo["gamma"],
                                    period=combo["period"],
                                    eccentricity=combo["eccentricity"],
                                    inclination=combo["inclination"],
                                    primary_minimum_time=combo["primary_minimum_time"],
                                    phase_shift=combo["phase_shift"])

    def test_mass_ratio(self):
        self.assertEqual(0.5, self._binary.mass_ratio)

    def _test_generic_setter(self, input_vals, expected, val_str):
        obtained = list()
        for val in input_vals:
            setattr(self._binary, val_str, val)
            obtained.append(round(getattr(self._binary, val_str), 3))
        assert_array_equal(obtained, expected)

    def test_period(self):
        periods = [0.25 * u.d, 0.65, 86400 * u.s]
        expected = [0.25, 0.65, 1.0]
        self._test_generic_setter(periods, expected, 'period')

    def test_eccentricity_valid(self):
        valid = [0.0, 0.3, 0.998]
        self._test_generic_setter(valid, valid, 'eccentricity')

    def test_eccentricity_invalid(self):
        invalid = [-1, 0.3 * u.deg, 1.0, 3.1]

        for i, e in enumerate(invalid):
            with self.assertRaises(Exception) as context:
                self._binary.eccentricity = e
            self.assertTrue('Input of variable `eccentricity`' in str(context.exception))

    def test_argument_of_periastron(self):
        periastrons = [135 * u.deg, 90.0, 1.56 * u.rad]
        expected = [2.356, 1.571, 1.560]
        self._test_generic_setter(periastrons, expected, 'argument_of_periastron')

    def test_primary_minimum_time(self):
        pmts = [0.25 * u.d, 0.65, 86400 * u.s]
        expected = [0.25, 0.65, 1.0]
        self._test_generic_setter(pmts, expected, 'primary_minimum_time')

    def test_inclination(self):
        inclinations = [135 * u.deg, 90.0, 1.56 * u.rad]
        expected = [2.356, 1.571, 1.56]
        self._test_generic_setter(inclinations, expected, 'inclination')


class TestBinarySystemInit(unittest.TestCase):
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
             "eccentricity": 0.0, "inclination": 90.0 * u.deg, "primary_minimum_time": 0.0,
             "phase_shift": 0.0,
             "primary_t_eff": 5000, "secondary_t_eff": 5000,
             "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
             "primary_albedo": 0.6, "secondary_albedo": 0.6
             },  # rotationally squashed compact spherical components

            {"primary_mass": 2.0, "secondary_mass": 1.0,
             "primary_surface_potential": 3.5, "secondary_surface_potential": 3.0,
             "primary_synchronicity": 1.5, "secondary_synchronicity": 1.2,
             "argument_of_periastron": c.HALF_PI, "gamma": 0.0, "period": 1.0,
             "eccentricity": 0.0, "inclination": 90.0 * u.deg, "primary_minimum_time": 0.0,
             "phase_shift": 0.0,
             "primary_t_eff": 5000, "secondary_t_eff": 5000,
             "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
             "primary_albedo": 0.6, "secondary_albedo": 0.6
             },  # close tidally deformed components with asynchronous rotation on circular orbit

            {"primary_mass": 2.0, "secondary_mass": 1.0,
             "primary_surface_potential": 4.8, "secondary_surface_potential": 4.0,
             "primary_synchronicity": 1.5, "secondary_synchronicity": 1.2,
             "argument_of_periastron": c.HALF_PI * u.rad, "gamma": 0.0, "period": 1.0,
             "eccentricity": 0.3, "inclination": 90.0 * u.deg, "primary_minimum_time": 0.0,
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
             "eccentricity": 0.0, "inclination": 90.0 * u.deg, "primary_minimum_time": 0.0,
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
             "eccentricity": 0.0, "inclination": 90.0 * u.deg, "primary_minimum_time": 0.0,
             "phase_shift": 0.0,
             "primary_t_eff": 5000, "secondary_t_eff": 5000,
             "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
             "primary_albedo": 0.6, "secondary_albedo": 0.6
             },  # asynchronous contact system (improbable but whatever...)

            {"primary_mass": 2.0, "secondary_mass": 1.0,
             "primary_surface_potential": 2.7,
             "secondary_surface_potential": 2.7,
             "primary_synchronicity": 1.0, "secondary_synchronicity": 1.0,
             "argument_of_periastron": 90 * u.deg, "gamma": 0.0, "period": 1.0,
             "eccentricity": 0.0, "inclination": 90.0 * u.deg, "primary_minimum_time": 0.0,
             "phase_shift": 0.0,
             "primary_t_eff": 5000, "secondary_t_eff": 5000,
             "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
             "primary_albedo": 0.6, "secondary_albedo": 0.6
             }  # over-contact system
        ]

    def _prepare_systems(self):
        s = list()
        for i, combo in enumerate(self.params_combination):
            primary = Star(mass=combo["primary_mass"], surface_potential=combo["primary_surface_potential"],
                           synchronicity=combo["primary_synchronicity"],
                           t_eff=combo["primary_t_eff"], gravity_darkening=combo["primary_gravity_darkening"],
                           albedo=combo['primary_albedo'], metallicity=0.0)

            secondary = Star(mass=combo["secondary_mass"], surface_potential=combo["secondary_surface_potential"],
                             synchronicity=combo["secondary_synchronicity"],
                             t_eff=combo["secondary_t_eff"], gravity_darkening=combo["secondary_gravity_darkening"],
                             albedo=combo['secondary_albedo'], metallicity=0.0)

            s.append(BinarySystem(primary=primary,
                                  secondary=secondary,
                                  argument_of_periastron=combo["argument_of_periastron"],
                                  gamma=combo["gamma"],
                                  period=combo["period"],
                                  eccentricity=combo["eccentricity"],
                                  inclination=combo["inclination"],
                                  primary_minimum_time=combo["primary_minimum_time"],
                                  phase_shift=combo["phase_shift"]))
        return s

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

    def test_setup_components_radii(self):
        radii = ["forward_radius", "side_radius", "equatorial_radius", "backward_radius", "polar_radius"]
        obtained = dict()

        expected = {
            "primary": {
                "forward_radius": [0.01005, 0.01229, 0.3783, 0.26257, 0.57075, 0.53029, np.nan],
                "side_radius": [0.01005, 0.01229, 0.35511, 0.24859, 0.43994, 0.41553, 0.48182],
                "equatorial_radius": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                "backward_radius": [0.01005, 0.01229, 0.36723, 0.25608, 0.46794, 0.44207, 0.52573],
                "polar_radius": [0.01005, 0.01005, 0.33055, 0.24242, 0.41427, 0.37162, 0.44577]
            },
            "secondary": {
                "forward_radius": [0.00506, 0.00763, 0.34966, 0.25693, 0.42925, 0.36743, np.nan],
                "side_radius": [0.00506, 0.00763, 0.29464, 0.21419, 0.31288, 0.28094, 0.35376],
                "equatorial_radius": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                "backward_radius": [0.00506, 0.00763, 0.32018, 0.23324, 0.34537, 0.30842, 0.41825],
                "polar_radius": [0.00506, 0.00635, 0.2798, 0.20994, 0.29977, 0.2489, 0.33306]
            }
        }

        for component in ["primary", "secondary"]:
            obtained[component] = dict()
            for radius in radii:
                obtained[component][radius] = list()
                for i, bs in enumerate(self._prepare_systems()):
                    value = np.round(getattr(getattr(bs, component), radius), 5)
                    assert_array_equal([value], [expected[component][radius][i]])

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


class TestValidity(unittest.TestCase):
    MANDATORY_KWARGS = ['gamma', 'inclination', 'period', 'eccentricity', 'argument_of_periastron',
                        'primary_minimum_time', 'phase_shift']

    def setUp(self):
        self._initial_params = {
            "primary_mass": 2.0, "secondary_mass": 1.0,
            "primary_surface_potential": 100.0, "secondary_surface_potential": 100.0,
            "primary_synchronicity": 1.0, "secondary_synchronicity": 1.0,
            "argument_of_periastron": c.HALF_PI * u.rad, "gamma": 0.0, "period": 1.0,
            "eccentricity": 0.0, "inclination": c.HALF_PI * u.deg, "primary_minimum_time": 0.0,
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

        self.assertTrue('Primary component is not instance of class' in str(context.exception))

        star_mandatory_kwargs = ['mass', 'surface_potential', 'synchronicity',
                                 'albedo', 'metallicity', 'gravity_darkening']

        for kw in star_mandatory_kwargs[:1]:
            p = copy(self._primary)
            setattr(p, f"_{kw}", None)

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


class TestMethods(unittest.TestCase):
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
             "eccentricity": 0.0, "inclination": c.HALF_PI * u.deg, "primary_minimum_time": 0.0,
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
             "eccentricity": 0.3, "inclination": 90.0 * u.deg, "primary_minimum_time": 0.0,
             "phase_shift": 0.0,
             "primary_t_eff": 5000, "secondary_t_eff": 5000,
             "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
             "primary_albedo": 0.6, "secondary_albedo": 0.6
             }  # close tidally deformed components with asynchronous rotation on eccentric orbit
        ]

        self._binaries = self._prepare_systems()

    def _prepare_systems(self):
        s = list()
        for i, combo in enumerate(self.params_combination):
            primary = Star(mass=combo["primary_mass"], surface_potential=combo["primary_surface_potential"],
                           synchronicity=combo["primary_synchronicity"],
                           t_eff=combo["primary_t_eff"], gravity_darkening=combo["primary_gravity_darkening"],
                           albedo=combo['primary_albedo'], metallicity=0.0)

            secondary = Star(mass=combo["secondary_mass"], surface_potential=combo["secondary_surface_potential"],
                             synchronicity=combo["secondary_synchronicity"],
                             t_eff=combo["secondary_t_eff"], gravity_darkening=combo["secondary_gravity_darkening"],
                             albedo=combo['secondary_albedo'], metallicity=0.0)

            s.append(BinarySystem(primary=primary,
                                  secondary=secondary,
                                  argument_of_periastron=combo["argument_of_periastron"],
                                  gamma=combo["gamma"],
                                  period=combo["period"],
                                  eccentricity=combo["eccentricity"],
                                  inclination=combo["inclination"],
                                  primary_minimum_time=combo["primary_minimum_time"],
                                  phase_shift=combo["phase_shift"]))
        return s

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
                    [1.21, 0., 0., 0.8437]]

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

    def test_angular_velocity(self):
        pass

    def test_calculate_orbital_motion(self):
        pass


def polar_gravity_acceleration(bs, component=None, components_distance=None):
    for _componet in component:
        components_instance = getattr(bs, _componet)

        mass_ratio = bs.mass_ratio if _componet == "primary" else 1.0 / bs.mass_ratio

        polar_radius = components_instance.polar_radius
        x_com = (mass_ratio * components_distance) / (1.0 + mass_ratio)
        semi_major_axis = bs.semi_major_axis

        primary_mass, secondary_mass = bs.primary.mass, bs.secondary.mass
        if _componet == "secondary":
            primary_mass, secondary_mass = secondary_mass, primary_mass

        r_vector = np.array([0.0, 0.0, polar_radius * semi_major_axis])
        centrifugal_distance = np.array([x_com * semi_major_axis, 0.0, 0.0])
        actual_distance = np.array([components_distance * semi_major_axis, 0., 0.])
        h_vector = r_vector - actual_distance
        angular_velocity = bs.angular_velocity(components_distance=components_distance)

        block_a = - ((const.G * primary_mass) / np.linalg.norm(r_vector) ** 3) * r_vector
        block_b = - ((const.G * secondary_mass) / np.linalg.norm(h_vector) ** 3) * h_vector
        block_c = - (angular_velocity ** 2) * centrifugal_distance

        g = block_a + block_b + block_c

        # magnitude of polar gravity acceleration in physical CGS units
        return np.linalg.norm(g) * 1e2
