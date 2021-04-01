import numpy as np
from numpy.testing import assert_array_equal, assert_equal
from copy import copy

from unittests.utils import ElisaTestCase, prepare_single_system
from elisa import units as u, const as c
from elisa import Star
from elisa import SingleSystem


class SingleSystemInitTestCase(ElisaTestCase):
    def setUp(self):
        super(SingleSystemInitTestCase, self).setUp()
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
            {
                "mass": 1.5,
                "t_eff": 6500 * u.K,
                "gravity_darkening": 0.32,
                "polar_log_g": 4.2 * u.dex(u.cm / u.s ** 2),
                "gamma": 10 * u.km / u.s,
                "inclination": 90.0 * u.deg,
                "rotation_period": 25.38 * u.d,
            },
        ]

    def prepare_systems(self):
        return [prepare_single_system(combo) for combo in self.params_combination]

    def test_critical_surface_potential(self):
        expected_potentials = np.round(np.array([-7873152926.098635, -10316748888.181795]), 5)
        obtained_potentials = list()
        for system in self.prepare_systems():
            obtained_potentials.append(system.star.critical_surface_potential)
        obtained_potentials = np.round(np.array(obtained_potentials), 5)
        assert_array_equal(expected_potentials, obtained_potentials)

    def test_setup_radii(self):
        radii = ["equatorial_radius", "polar_radius", "equivalent_radius"]

        expected = {
            "equatorial_radius": [69595445.87082, 1120763694.70205],
            "polar_radius": [69595445.14516, 1120731159.03338],
            "equivalent_radius": [69595445.62893, 1120752849.37421]
        }

        # obtained = {
        #     "equatorial_radius": [],
        #     "polar_radius": [],
        #     "equivalent_radius": []
        # }

        for i, system in enumerate(self.prepare_systems()):
            for radius in radii:
                value = np.round(getattr(system.star, radius), 0) if hasattr(system.star, radius) else np.nan
                # obtained[radius].append(value)
                assert_equal(np.round(expected[radius][i], 0), value)


class ValidityTestCase(ElisaTestCase):
    MANDATORY_KWARGS = ['inclination', 'rotation_period']

    def setUp(self):
        super(ValidityTestCase, self).setUp()
        self._initial_params = {
            "mass": 1.0 * u.solMass,
            "t_eff": 5772 * u.K,
            "gravity_darkening": 0.32,
            "polar_log_g": 4.43775,
            "gamma": 0.0,
            "inclination": 90.0 * u.deg,
            "rotation_period": 25.38 * u.d,
            "phase_shift": 0.1,
            "reference_time": 0.0 * u.d
        }

        self._star = Star(mass=self._initial_params["mass"],
                          t_eff=self._initial_params["t_eff"],
                          gravity_darkening=self._initial_params["gravity_darkening"],
                          polar_log_g=self._initial_params["polar_log_g"]
                          )

    def test__star_params_validity_check(self):
        with self.assertRaises(Exception) as context:
            SingleSystem(star=42,
                         gamma=self._initial_params["gamma"],
                         rotation_period=self._initial_params["rotation_period"],
                         inclination=self._initial_params["inclination"],
                         reference_time=self._initial_params["reference_time"],
                         phase_shift=self._initial_params["phase_shift"])

        self.assertTrue('Component `star` is not instance of class' in str(context.exception))

        star_mandatory_kwargs = ['mass', 't_eff', 'gravity_darkening', 'polar_log_g', 'metallicity']

        for kw in star_mandatory_kwargs[:1]:
            s = copy(self._star)
            setattr(s, f"{kw}", None)

            with self.assertRaises(Exception) as context:
                SingleSystem(star=s,
                             gamma=self._initial_params["gamma"],
                             rotation_period=self._initial_params["rotation_period"],
                             inclination=self._initial_params["inclination"],
                             reference_time=self._initial_params["reference_time"],
                             phase_shift=self._initial_params["phase_shift"])

            self.assertTrue(f'Mising argument(s): `{kw}`' in str(context.exception))

    def test_invalid_kwarg_checker(self):
        with self.assertRaises(Exception) as context:
            SingleSystem(star=self._star,
                         gamma=self._initial_params["gamma"],
                         rotation_period=self._initial_params["rotation_period"],
                         inclination=self._initial_params["inclination"],
                         reference_time=self._initial_params["reference_time"],
                         phase_shift=self._initial_params["phase_shift"],
                         adhoc_param="xxx")

        self.assertTrue('Invalid keyword argument(s): adhoc_param' in str(context.exception))

    def test_check_missing_kwargs(self):
        initial_kwargs = dict(star=self._star,
                              gamma=self._initial_params["gamma"],
                              rotation_period=self._initial_params["rotation_period"],
                              inclination=self._initial_params["inclination"],
                              reference_time=self._initial_params["reference_time"],
                              phase_shift=self._initial_params["phase_shift"])

        for kw in self.MANDATORY_KWARGS:
            kwargs = copy(initial_kwargs)
            del (kwargs[kw])
            with self.assertRaises(Exception) as context:
                SingleSystem(**kwargs)
            self.assertTrue(f'Missing argument(s): `{kw}`' in str(context.exception))


class SingleSystemSerializersTestCase(ElisaTestCase):
    def setUp(self):
        super(SingleSystemSerializersTestCase, self).setUp()
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
            {
                "mass": 1.5,
                "t_eff": 6500 * u.K,
                "gravity_darkening": 0.32,
                "polar_log_g": 4.2 * u.dex(u.cm / u.s ** 2),
                "gamma": 10 * u.km / u.s,
                "inclination": 90.0 * u.deg,
                "rotation_period": 25.38 * u.d,
            },
        ]

        self._singles = self.prepare_systems()

    def prepare_systems(self):
        return [prepare_single_system(combo) for combo in self.params_combination]

    def test_kwargs_serializer(self):
        single = self._singles[-1]
        obtained = single.kwargs_serializer()

        expected = dict(
            inclination=c.HALF_PI,
            rotation_period=25.38,
            reference_time=0,
            phase_shift=0.0,
            additional_light=0.0,
            gamma=10000
        )

        obtained_array, expectedd_array = list(), list()

        for o_key, e_key in zip(obtained, expected):
            if isinstance(obtained[o_key], u.Quantity):
                obtained[o_key] = obtained[o_key].value
            obtained_array.append(round(obtained[o_key], 5))
            expectedd_array.append(round(expected[e_key], 5))

        assert_array_equal(expectedd_array, obtained_array)

    def test_properties_serializer(self):
        single = self._singles[-1]
        obtained = single.properties_serializer()
        expected = ["angular_velocity"]
        for e in expected:
            self.assertTrue(e in obtained)

    @staticmethod
    def _get_std(data):
        return SingleSystem.from_json(data)

    @classmethod
    def test_init_from_json_std(cls):
        data = {
            "system": {
                "inclination": 90.0,
                "rotation_period": 10.1,
                "gamma": 10000,
                "reference_time": 0.5,
                "phase_shift": 0.0
            },
            "star": {
                "mass": 1.0,
                "t_eff": 5772.0,
                "gravity_darkening": 0.32,
                "discretization_factor": 5,
                "metallicity": 0.0,
                "polar_log_g": 2.43775
            }
        }
        cls._get_std(data)

    @classmethod
    def test_init_from_json_radius(cls):
        data = {
            "system": {
                "inclination": 90.0,
                "rotation_period": 10.1,
                "gamma": 10000,
                "reference_time": 0.5,
                "phase_shift": 0.0
            },
            "star": {
                "mass": 1.0,
                "t_eff": 5772.0,
                "gravity_darkening": 0.32,
                "discretization_factor": 5,
                "metallicity": 0.0,
                "equivalent_radius": 1.0
            }
        }
        cls._get_std(data)

    @classmethod
    def test_init_string_repr(cls):
        data = {
            "system": {
                "inclination": "90.0 deg",
                "rotation_period": "10.1 d",
                "gamma": "10 km / s",
                "reference_time": "0. d",
                "phase_shift": 0.0
            },
            "star": {
                "mass": "1.0 solMass",
                "t_eff": "5772.0 K",
                "gravity_darkening": 0.32,
                "discretization_factor": 5,
                "metallicity": 0.0,
                "polar_log_g": "4.43775 dex(cm / s2)"
            }
        }
        cls._get_std(data)
