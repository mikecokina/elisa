# import random
# import sys
import unittest
from copy import copy

from astropy import units as u
# from os.path import dirname
# from os.path import join as pjoin
#
# import numpy as np
# import pandas as pd
from numpy.testing import assert_array_equal

from elisa import const as c
from elisa.base.star import Star
from elisa.binary_system.system import BinarySystem


# from pandas.testing import assert_frame_equal


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
             "argument_of_periastron": c.HALF_PI * u.rad, "gamma": 0.0, "period": 1.0,
             "eccentricity": 0.0, "inclination": c.HALF_PI * u.rad, "primary_minimum_time": 0.0,
             "phase_shift": 0.0,
             "primary_t_eff": 5000, "secondary_t_eff": 5000,
             "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
             "primary_albedo": 0.6, "secondary_albedo": 0.6,
             },  # compact spherical components on circular orbit

            {"primary_mass": 2.0, "secondary_mass": 1.0,
             "primary_surface_potential": 100.0, "secondary_surface_potential": 80.0,
             "primary_synchronicity": 400, "secondary_synchronicity": 550,
             "argument_of_periastron": c.HALF_PI * u.rad, "gamma": 0.0, "period": 1.0,
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
             }  # contact system
        ]

    def test_calculate_semi_major_axis(self):
        pass

    def test__setup_periastron_critical_potential(self):
        pass

    def test__setup_morphology(self):
        pass

    def test_setup_components_radii(self):
        pass


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
            del(kwargs[kw])
            with self.assertRaises(Exception) as context:
                BinarySystem(**kwargs)
            self.assertTrue(f'Missing argument(s): `{kw}`' in str(context.exception))


class Test(unittest.TestCase):
    def setUp(self):
        # self.params_combination = [
        #     {"primary_mass": 2.0, "secondary_mass": 1.0,
        #      "primary_surface_potential": 100.0, "secondary_surface_potential": 100.0,
        #      "primary_synchronicity": 1.0, "secondary_synchronicity": 1.0,
        #      "argument_of_periastron": c.HALF_PI * u.rad, "gamma": 0.0, "period": 1.0,
        #      "eccentricity": 0.0, "inclination": c.HALF_PI * u.deg, "primary_minimum_time": 0.0,
        #      "phase_shift": 0.0,
        #      "primary_t_eff": 5000, "secondary_t_eff": 5000,
        #      "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
        #      "primary_albedo": 0.6, "secondary_albedo": 0.6,
        #      },
        #     # compact spherical components on circular orbit
        #
        #     {"primary_mass": 2.0, "secondary_mass": 1.0,
        #      "primary_surface_potential": 100.0, "secondary_surface_potential": 80.0,
        #      "primary_synchronicity": 400, "secondary_synchronicity": 550,
        #      "argument_of_periastron": c.HALF_PI * u.rad, "gamma": 0.0, "period": 1.0,
        #      "eccentricity": 0.0, "inclination": 90.0 * u.deg, "primary_minimum_time": 0.0,
        #      "phase_shift": 0.0,
        #      "primary_t_eff": 5000, "secondary_t_eff": 5000,
        #      "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
        #      "primary_albedo": 0.6, "secondary_albedo": 0.6
        #      },  # rotationally squashed compact spherical components
        #
        #     {"primary_mass": 2.0, "secondary_mass": 1.0,
        #      "primary_surface_potential": 3.5, "secondary_surface_potential": 3.0,
        #      "primary_synchronicity": 1.5, "secondary_synchronicity": 1.2,
        #      "argument_of_periastron": c.HALF_PI, "gamma": 0.0, "period": 1.0,
        #      "eccentricity": 0.0, "inclination": 90.0 * u.deg, "primary_minimum_time": 0.0,
        #      "phase_shift": 0.0,
        #      "primary_t_eff": 5000, "secondary_t_eff": 5000,
        #      "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
        #      "primary_albedo": 0.6, "secondary_albedo": 0.6
        #      },  # close tidally deformed components with asynchronous rotation
        #     # on circular orbit
        #
        #     {"primary_mass": 2.0, "secondary_mass": 1.0,
        #      "primary_surface_potential": 4.8, "secondary_surface_potential": 4.0,
        #      "primary_synchronicity": 1.5, "secondary_synchronicity": 1.2,
        #      "argument_of_periastron": c.HALF_PI * u.rad, "gamma": 0.0, "period": 1.0,
        #      "eccentricity": 0.3, "inclination": 90.0 * u.deg, "primary_minimum_time": 0.0,
        #      "phase_shift": 0.0,
        #      "primary_t_eff": 5000, "secondary_t_eff": 5000,
        #      "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
        #      "primary_albedo": 0.6, "secondary_albedo": 0.6
        #      },  # close tidally deformed components with asynchronous rotation
        #     # on eccentric orbit
        #
        #     {"primary_mass": 2.0, "secondary_mass": 1.0,
        #      "primary_surface_potential": 2.875844632141054,
        #      "secondary_surface_potential": 2.875844632141054,
        #      "primary_synchronicity": 1.0, "secondary_synchronicity": 1.0,
        #      "argument_of_periastron": c.HALF_PI * u.rad, "gamma": 0.0, "period": 1.0,
        #      "eccentricity": 0.0, "inclination": 90.0 * u.deg, "primary_minimum_time": 0.0,
        #      "phase_shift": 0.0,
        #      "primary_t_eff": 5000, "secondary_t_eff": 5000,
        #      "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
        #      "primary_albedo": 0.6, "secondary_albedo": 0.6
        #      },  # synchronous contact system
        #
        #     {"primary_mass": 2.0, "secondary_mass": 1.0,
        #      "primary_surface_potential": 3.159639848886489,
        #      "secondary_surface_potential": 3.229240544834036,
        #      "primary_synchronicity": 1.5, "secondary_synchronicity": 2.0,
        #      "argument_of_periastron": c.HALF_PI * u.rad, "gamma": 0.0, "period": 1.0,
        #      "eccentricity": 0.0, "inclination": 90.0 * u.deg, "primary_minimum_time": 0.0,
        #      "phase_shift": 0.0,
        #      "primary_t_eff": 5000, "secondary_t_eff": 5000,
        #      "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
        #      "primary_albedo": 0.6, "secondary_albedo": 0.6
        #      },  # asynchronous contact system (improbable but whatever...)
        #
        #     {"primary_mass": 2.0, "secondary_mass": 1.0,
        #      "primary_surface_potential": 2.7,
        #      "secondary_surface_potential": 2.7,
        #      "primary_synchronicity": 1.0, "secondary_synchronicity": 1.0,
        #      "argument_of_periastron": c.HALF_PI * u.rad, "gamma": 0.0, "period": 1.0,
        #      "eccentricity": 0.0, "inclination": 90.0 * u.deg, "primary_minimum_time": 0.0,
        #      "phase_shift": 0.0,
        #      "primary_t_eff": 5000, "secondary_t_eff": 5000,
        #      "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
        #      "primary_albedo": 0.6, "secondary_albedo": 0.6
        #      }  # contact system
        # ]
        pass
