from copy import copy
from unittest import (
    mock
)
import astropy.units as u

import numpy as np
from numpy.testing import assert_array_equal

from elisa.analytics.binary import params
from elisa.base.error import InitialParamsError
from elisa.binary_system import t_layer
from elisa.conf.config import BINARY_COUNTERPARTS
from unittests.utils import ElisaTestCase
from elisa.analytics.dataset.base import RVData, LCData
from elisa.analytics.base import BinarySystemAnalyticsTask
from elisa.analytics import utils as autils

TOL = 1e-5


class TestParamsTestCase(ElisaTestCase):
    def setUp(self):
        self.x0 = {
            'system': {
                'argument_of_periastron': {
                    'value': 180,
                    'fixed': False
                }
            },
            'primary': {
                'mass': {
                    'value': 2.0,
                    'fixed': False,
                    'min': 1.0,
                    'max': 3.0
                },
                't_eff': {
                    'value': 4000.0,
                    'fixed': True,
                    'min': 3500.0,
                    'max': 4500.0
                },
            },
        }

        params.NORMALIZATION_MAP.update({
            params.PARAMS_KEY_MAP['M1']: (1.0, 3.0),
            params.PARAMS_KEY_MAP['T1']: (3000, 5000),
            params.PARAMS_KEY_MAP['omega']: (0, 360),
        })

    def test_x0_vectorize(self):
        x0, labels = params.x0_vectorize(self.x0)
        expected_x0 = (180, 2.0)
        expected_labels = [params.PARAM_PARSER.join(['system', 'argument_of_periastron']),
                           params.PARAM_PARSER.join(['primary', 'mass'])]

        assert_array_equal(x0, expected_x0)
        assert_array_equal(labels, expected_labels)

    def test_x0_to_kwargs(self):
        obtained = params.x0_to_kwargs(self.x0)
        expected = {params.PARAM_PARSER.join(['primary', 'mass']): 2.0,
                    params.PARAM_PARSER.join(['primary', 't_eff']): 4000.0,
                    params.PARAM_PARSER.join(['system', 'argument_of_periastron']): 180}
        self.assertDictEqual(obtained, expected)

    def test_flatten_fit_params(self):
        obtained = params.flatten_fit_params(self.x0)
        expected = {params.PARAM_PARSER.join(['primary', 'mass']): self.x0['primary']['mass'],
                    params.PARAM_PARSER.join(['primary', 't_eff']): self.x0['primary']['t_eff'],
                    params.PARAM_PARSER.join(['system', 'argument_of_periastron']):
                        self.x0['system']['argument_of_periastron']}
        self.assertDictEqual(obtained, expected)

    def test_dict_to_user_format(self):
        flattened = params.flatten_fit_params(self.x0)
        ret_dict = params.dict_to_user_format(flattened)
        self.assertDictEqual(self.x0, ret_dict)

    def test_x0_to_fixed_kwargs(self):
        obtained = params.x0_to_fixed_kwargs(self.x0)
        expected = {params.PARAM_PARSER.join(['primary', 't_eff']): 4000.0}
        self.assertDictEqual(obtained, expected)

    def test_serialize_param_boundaries(self):
        obtained = params.serialize_param_boundaries(self.x0)
        expected = {params.PARAM_PARSER.join(['primary', 'mass']): (1.0, 3.0),
                    params.PARAM_PARSER.join(['system', 'argument_of_periastron']): (0, 360)}
        self.assertDictEqual(obtained, expected)

    def test_param_normalizer(self):
        boundaries = params.serialize_param_boundaries(self.x0)
        params.update_normalization_map(boundaries)

        x0, labels = params.x0_vectorize(self.x0)
        obtained = params.param_normalizer(x0, labels)
        expected = np.array([0.5, 0.5])

        self.assertTrue(np.all(np.abs(expected - obtained) < TOL))

    def test_param_renormalizer(self):
        values = [0.5, 0.5]
        labels = [params.PARAM_PARSER.join(['primary', 'mass']),
                  'argument_of_periastron']

        obtained = params.param_renormalizer(values, labels)
        expected = np.array([[2.0, 180.0]])
        self.assertTrue(np.all(np.abs(expected - obtained) < TOL))

    def test_extend_result_with_units(self):
        result = {
            params.PARAM_PARSER.join(['primary', 'mass']): {'value': 2.0},
            params.PARAM_PARSER.join(['primary', 't_eff']): {'value': 4000},
            params.PARAM_PARSER.join(['primary', 'spots', 'spot1', 'longitude']): {'value': 30},
        }

        expected = {
            params.PARAM_PARSER.join(['primary', 'mass']): {'value': 2.0, 'unit': u.solMass},
            params.PARAM_PARSER.join(['primary', 't_eff']): {'value': 4000, 'unit': u.K},
            params.PARAM_PARSER.join(['primary', 'spots', 'spot1', 'longitude']): {'value': 30, 'unit': u.degree},
        }

        obtained = params.extend_result_with_units(result)
        assert_array_equal(expected, obtained)

    def test_is_overcontact(self):
        self.assertTrue(params.is_overcontact('over-contact'))
        self.assertFalse(params.is_overcontact('overcontact'))
        self.assertFalse(params.is_overcontact('detached'))

    @skip('Not used')
    def test_adjust_result_constrained_potential(self):
        hash_map = {'p__surface_potential': 0, 's__surface_potential': 1}
        xn = [
            {
                "value": 1.1,
                "param": "p__surface_potential"
            },
            {
                "value": 1.2,
                "param": "s__surface_potential"
            }
        ]

        expected = [{'value': 1.1, 'param': 'p__surface_potential'},
                    {'param': 's__surface_potential', 'value': 1.1}]
        obtained = params.adjust_result_constrained_potential(xn, hash_map)
        self.assertTrue(expected[0]['value'] == obtained[0]['value'])

    def test_initial_x0_validity_check(self):
        xn = {
            'primary': {
                "surface_potential": {
                    "value": 1.1,
                    "min": 1.0,
                    "max": 10.0
                },
                "t_eff": {
                    'fixed': False,
                    "value": 5000,
                    "min": 4000,
                    "max": 6000
                }
            },
            'secondary': {
                "surface_potential": {
                    "value": 1.2,
                    "min": 1.1,
                    "max": 11.0
                },
            }
        }

        params.lc_initial_x0_validity_check(xn, morphology='detached')

        xn['primary']["surface_potential"]["fixed"] = True
        xn['secondary']["surface_potential"]["fixed"] = False
        with self.assertRaises(InitialParamsError) as context:
            params.lc_initial_x0_validity_check(xn, morphology='over-contact')
        self.assertTrue("just one fixed potential" in str(context.exception).lower())

        xn['primary']["surface_potential"]["fixed"] = True
        xn['secondary']["surface_potential"]["fixed"] = True
        with self.assertRaises(InitialParamsError) as context:
            params.lc_initial_x0_validity_check(xn, morphology='over-contact')
        self.assertTrue("different potential" in str(context.exception).lower())

        xn['primary']["surface_potential"]["fixed"] = False
        xn['secondary']["surface_potential"]["fixed"] = False
        params.lc_initial_x0_validity_check(xn, morphology='over-contact')

        self.assertTrue(xn['primary']["surface_potential"]['min'] == xn['secondary']["surface_potential"]['min'])
        self.assertTrue(xn['primary']["surface_potential"]['max'] == xn['secondary']["surface_potential"]['max'])

        self.assertTrue(xn['secondary']["surface_potential"]['constraint'] ==
                        "{primary" + params.PARAM_PARSER + "surface_potential}")
        self.assertFalse(xn['primary']["surface_potential"].get('fixed', False))

    def test_mixed_fixed_and_constraint_raise_error(self):
        xn = {
            'primary': {
                "surface_potential": {
                    "value": 1.1,
                    "min": 1.0,
                    "max": 10.0,
                    "fixed": True,
                    "constraint": "{secondary.surface_potential}"
                },
            },
            'secondary': {
                "surface_potential": {
                    "value": 1.1,
                    'fixed': False
                }
            }
        }

        with self.assertRaises(InitialParamsError) as context:
            params.lc_initial_x0_validity_check(xn, morphology='detached')
        self.assertTrue("to contain `fixed` and `constraint`" in str(context.exception).lower())

    def test_constraints(self):
        x0 = {
            'system': {
                'inclination': {
                    'value': 80,
                    'fixed': True,
                    'units': u.deg
                },
                'semi_major_axis': {
                    "value": 3.5,
                    'fixed': False,
                    'min': 3.0,
                    'max': 4.0,
                    'unit': u.solRad
                },
                'eccentricity': {
                    'constraint': '2 * {semi_major_axis}',
                    'unit': ''
                }
            },
            'primary': {
                'mass':{
                    "value": 10.0,
                    "fixed": True,
                    'unit': u.solMass
                },
                'surface_potential': {
                    'value': 3.5,
                    'fixed': False,
                    'min': 3.0,
                    'max': 4.0,
                    'unit': ''
                },
                'synchronicity': {
                    'constraint': '2 * {semi_major_axis}',
                    'unit': ''
                },
                'spots': {
                    'spot1': {
                        'latitude': {
                            'value': 30,
                            'fixed': True,
                            'unit': u.deg
                        },
                        'longitude': {
                            'value': 3.5,
                            'fixed': False,
                            'min': 3.0,
                            'max': 4.0,
                            'unit': u.deg
                        },
                        'temperature_factor': {
                            'constraint': '{semi_major_axis} + {spot1.longitude}'
                        }
                    }
                }
            }
        }

        x0c = params.x0_to_constrained_kwargs(x0)
        vectorized, labels = params.x0_vectorize(x0)
        x0v = {key: val for key, val in zip(labels, vectorized)}

        evaluated = params.constraints_evaluator(x0v, x0c)
        self.assertTrue(evaluated[params.PARAM_PARSER.join(['primary', 'spots', 'spot1', 'temperature_factor'])] == 7.0)

    def test_xs_reducer_all_same(self):
        a = np.linspace(0, 1, 5)
        xs = {"a": a, "b": a, "c": a}
        new_xs, inverse = params.xs_reducer(xs)
        for band, phases in xs.items():
            assert_array_equal(inverse[band], np.arange(len(phases)))
        assert_array_equal(new_xs, a)

    def test_xs_reducer_random(self):
        a = np.linspace(0, 1, 5)
        b = np.array(np.array(a + 0.01).tolist() + [0.021])
        c = a + 0.02
        xs = {"a": a, "b": b, "c": c}
        new_xs, inverse = params.xs_reducer(xs)
        expected_xs = [0., 0.01, 0.02, 0.021, 0.25, 0.26, 0.27, 0.5, 0.51, 0.52, 0.75, 0.76, 0.77, 1., 1.01, 1.02]
        expected_inverse = {'a': [0, 4, 7, 10, 13], 'b': [1, 5, 8, 11, 14, 3], 'c': [2, 6, 9, 12, 15]}

        for band, phases in xs.items():
            assert_array_equal(inverse[band], expected_inverse[band])
        assert_array_equal(expected_xs, new_xs)


class AbstractFitTestCase(ElisaTestCase):
    def setUp(self):
        self.model_generator = ModelSimulator()

    phases = {'Generic.Bessell.V': np.arange(-0.6, 0.62, 0.02),
              'Generic.Bessell.B': np.arange(-0.6, 0.62, 0.02)}

    flux = {'Generic.Bessell.V': np.array([0.98128349, 0.97901564, 0.9776404, 0.77030991, 0.38623294,
                                           0.32588823, 0.38623294, 0.77030991, 0.9776404, 0.97901564,
                                           0.98128349, 0.9831816, 0.98542223, 0.9880625, 0.99034951,
                                           0.99261368, 0.99453225, 0.99591341, 0.9972921, 0.99865607,
                                           0.99943517, 0.99978567, 1., 0.99970989, 0.99963265,
                                           0.99967025, 0.9990695, 0.99904945, 0.96259235, 0.8771112,
                                           0.83958173, 0.8771112, 0.96259235, 0.99904945, 0.9990695,
                                           0.99967025, 0.99963265, 0.99970989, 1., 0.99978567,
                                           0.99943517, 0.99865607, 0.9972921, 0.99591341, 0.99453225,
                                           0.99261368, 0.99034951, 0.9880625, 0.98542223, 0.9831816,
                                           0.98128349, 0.97901564, 0.9776404, 0.77030991, 0.38623294,
                                           0.32588823, 0.38623294, 0.77030991, 0.9776404, 0.97901564,
                                           0.98128349]),
            'Generic.Bessell.B': np.array([0.80924345, 0.80729325, 0.80604709, 0.60603475, 0.2294959,
                                           0.17384023, 0.2294959, 0.60603475, 0.80604709, 0.80729325,
                                           0.80924345, 0.81088916, 0.81276665, 0.81488617, 0.81664783,
                                           0.81831472, 0.81957938, 0.82037431, 0.82105228, 0.82161889,
                                           0.82171702, 0.82140855, 0.82099437, 0.82019232, 0.81957921,
                                           0.81911052, 0.81821162, 0.81784563, 0.79824012, 0.7489621,
                                           0.72449315, 0.7489621, 0.79824012, 0.81784563, 0.81821162,
                                           0.81911052, 0.81957921, 0.82019232, 0.82099437, 0.82140855,
                                           0.82171702, 0.82161889, 0.82105228, 0.82037431, 0.81957938,
                                           0.81831472, 0.81664783, 0.81488617, 0.81276665, 0.81088916,
                                           0.80924345, 0.80729325, 0.80604709, 0.60603475, 0.2294959,
                                           0.17384023, 0.2294959, 0.60603475, 0.80604709, 0.80729325,
                                           0.80924345])}


class McMcLCTestCase(AbstractFitTestCase):
    """
    Requre just methods to pass.
    """

    def test_mcmc_lc_fit_std_params_detached(self):
        dinit = {
            'system': {
                'inclination': {
                    'value': 90.0,
                    'fixed': True
                },
                'eccentricity': {
                    'value': 0.0,
                    'fixed': True
                },
                'argument_of_periastron': {
                    'value': 0.0,
                    'fixed': True
                },
                'period': {
                    'value': 3.0,
                    'fixed': True
                },
            },
            'primary': {
                'mass': {
                    'value': 1.8,  # 2.0
                    'fixed': False,
                    'min': 1.5,
                    'max': 2.2
                },
                't_eff': {
                    'value': 5000.0,
                    'fixed': True
                },
                'surface_potential': {
                    'value': 5.0,
                    'fixed': True
                },
                'gravity_darkening': {
                    'value': 1.0,
                    'fixed': True
                },
                'albedo': {
                    'value': 1.0,
                    'fixed': True
                },
            },
            'secondary': {
                'mass': {
                    'value': 1.0,
                    'fixed': True
                },
                't_eff': {
                    'value': 6500.0,  # 7000
                    'fixed': False,
                    'min': 5000.0,
                    'max': 10000.0
                },
                'surface_potential': {
                    'value': 5,
                    'fixed': True
                },
                'gravity_darkening': {
                    'value': 1.0,
                    'fixed': True
                },
                'albedo': {
                    'value': 1.0,
                    'fixed': True
                },
            },
        }

        lc_v = LCData(
            x_data=self.phases['Generic.Bessell.V'],
            y_data=self.flux['Generic.Bessell.V'],
            x_unit=u.dimensionless_unscaled,
            y_unit=u.dimensionless_unscaled
        )

        lc_b = LCData(
            x_data=self.phases['Generic.Bessell.B'],
            y_data=self.flux['Generic.Bessell.B'],
            x_unit=u.dimensionless_unscaled,
            y_unit=u.dimensionless_unscaled
        )

        self.model_generator.keep_out = True
        with mock.patch("elisa.analytics.binary.models.synthetic_binary", self.model_generator.lc_generator):
            task = BinarySystemAnalyticsTask(light_curves={'Generic.Bessell.V': lc_v, 'Generic.Bessell.B': lc_b})
            result = task.lc_fit.fit(x0=copy(dinit), method='mcmc', nsteps=10, discretization=20)

    def test_mcmc_lc_fit_community_params_detached(self):
        dinit = {
            'system': {
                'semi_major_axis': {
                    'value': 11.0,  # 12.62
                    'fixed': False,
                    'min': 7.0,
                    'max': 15.0
                },
                'mass_ratio': {
                    'value': 0.7,  # 0.5
                    'fixed': False,
                    'min': 0.3,
                    'max': 2.0
                },
                'inclination': {
                    'value': 90.0,
                    'fixed': True
                },
                'eccentricity': {
                    'value': 0.0,
                    'fixed': True
                },
                'argument_of_periastron': {
                    'value': 0.0,
                    'fixed': True
                },
                'period': {
                    'value': 3.0,
                    'fixed': True
                },
            },
            'primary': {
                't_eff': {
                    'value': 5000.0,
                    'fixed': True
                },
                'surface_potential': {
                    'value': 5.0,
                    'fixed': True
                },
                'gravity_darkening': {
                    'value': 1.0,
                    'fixed': True
                },
                'albedo': {
                    'value': 1.0,
                    'fixed': True
                },
            },
            'secondary': {
                't_eff': {
                    'value': 7000.0,
                    'fixed': True
                },
                'surface_potential': {
                    'value': 5,
                    'fixed': True
                },
                'gravity_darkening': {
                    'value': 1.0,
                    'fixed': True
                },
                'albedo': {
                    'value': 1.0,
                    'fixed': True
                },
            },
        }

        lc_v = LCData(
            x_data=self.phases['Generic.Bessell.V'],
            y_data=self.flux['Generic.Bessell.V'],
            x_unit=u.dimensionless_unscaled,
            y_unit=u.dimensionless_unscaled
        )

        lc_b = LCData(
            x_data=self.phases['Generic.Bessell.B'],
            y_data=self.flux['Generic.Bessell.B'],
            x_unit=u.dimensionless_unscaled,
            y_unit=u.dimensionless_unscaled
        )

        self.model_generator.keep_out = True
        with mock.patch("elisa.analytics.binary.models.synthetic_binary", self.model_generator.lc_generator):
            task = BinarySystemAnalyticsTask(light_curves={'Generic.Bessell.V': lc_v, 'Generic.Bessell.B': lc_b})
            result = task.lc_fit.fit(x0=copy(dinit), method='mcmc', nsteps=10, discretization=20)


class RVTestCase(ElisaTestCase):
    def setUp(self):
        self.model_generator = ModelSimulator()

    rv = {'primary': -1 * np.array([111221.02018955, 102589.40515112, 92675.34114568,
                                    81521.98280508, 69189.28515476, 55758.52165462,
                                    41337.34984718, 26065.23187763, 10118.86370365,
                                    -6282.93249474, -22875.63138097, -39347.75579673,
                                    -55343.14273712, -70467.93174445, -84303.24303593,
                                    -96423.915992, -106422.70195531, -113938.05716509,
                                    -118682.43573797, -120467.13803823, -119219.71866465,
                                    -114990.89801808, -107949.71016039, -98367.77975255,
                                    -86595.51823899, -73034.14124119, -58107.52819161,
                                    -42237.21613808, -25822.61388457, -9227.25025377,
                                    7229.16722243, 23273.77242388, 38679.82691287,
                                    53263.47669152, 66879.02978007, 79413.57620399,
                                    90781.53548261, 100919.51001721, 109781.66096297,
                                    117335.70723602, 123559.57210929, 128438.6567666,
                                    131963.69775175, 134129.15836278, 134932.10727626,
                                    134371.54717101, 132448.1692224, 129164.5242993,
                                    124525.61727603, 118539.94602416, 111221.02018955,
                                    102589.40515112, 92675.34114568, 81521.98280508,
                                    69189.28515476, 55758.52165462, 41337.34984718,
                                    26065.23187763, 10118.86370365, -6282.93249474,
                                    -22875.63138097]),
          'secondary': -1 * np.array([-144197.83633559, -128660.92926642, -110815.61405663,
                                      -90739.56904355, -68540.71327298, -44365.33897272,
                                      -18407.22971932, 9082.58262586, 37786.04533903,
                                      67309.27849613, 97176.13649135, 126825.96043971,
                                      155617.65693242, 182842.27714561, 207745.83747028,
                                      229563.04879121, 247560.86352515, 261088.50290277,
                                      269628.38433395, 272840.84847442, 270595.49360196,
                                      262983.61643814, 250309.4782943, 233062.00356019,
                                      211871.93283578, 187461.45423975, 160593.55075051,
                                      132026.98905414, 102480.70499782, 72609.05046238,
                                      42987.49900522, 14107.20964261, -13623.68843756,
                                      -39874.25803914, -64382.25359853, -86944.43716158,
                                      -107406.76386309, -125655.11802538, -141606.98972774,
                                      -155204.27301923, -166407.22979112, -175189.58217428,
                                      -181534.65594755, -185432.4850474, -186877.79309168,
                                      -185868.78490222, -182406.70459473, -176496.14373313,
                                      -168146.11109126, -157371.90283788, -144197.83633559,
                                      -128660.92926641, -110815.61405663, -90739.56904355,
                                      -68540.71327297, -44365.33897271, -18407.22971932,
                                      9082.58262586, 37786.04533903, 67309.27849613,
                                      97176.13649135])}


class McMcRVTestCase(RVTestCase):
    def test_mcmc_rv_fit_community_params(self):
        phases = np.arange(-0.6, 0.62, 0.02)

        rv_primary = RVData(
            x_data=phases,
            y_data=self.rv['primary'],
            x_unit=u.dimensionless_unscaled,
            y_unit=u.m / u.s
        )

        rv_secondary = RVData(
            x_data=phases,
            y_data=self.rv['secondary'],
            x_unit=u.dimensionless_unscaled,
            y_unit=u.m / u.s
        )

        initial_parameters = \
            {
                'system': {
                    'eccentricity': {
                        'value': 0.1,
                        'fixed': True,
                    },
                    'asini': {
                        'value': 20.0,  # 4.219470628180749
                        'fixed': False,
                        'min': 1.0,
                        'max': 100
                    },
                    'mass_ratio': {
                        'value': 0.8,  # 1.0 / 1.8
                        'fixed': False,
                        'min': 0,
                        'max': 2
                    },
                    'argument_of_periastron': {
                        'value': 0.0,
                        'fixed': True
                    },
                    'gamma': {
                        'value': -20000.0,
                        'fixed': True
                    },
                    'period': {
                        'value': 0.6,
                        'fixed': True
                    }
                }
            }

        self.model_generator.keep_out = True
        with mock.patch("elisa.analytics.binary.models.central_rv_synthetic", self.model_generator.rv_generator):
            task = BinarySystemAnalyticsTask(radial_velocities={'primary': rv_primary, 'secondary': rv_secondary})
            fit_params = task.rv_fit.fit(x0=initial_parameters, method='mcmc', nsteps=100)

        self.assertTrue(1.0 > fit_params['system']["r_squared"]['value'] > 0.9)


class LeastSqaureRVTestCase(RVTestCase):
    def test_least_squares_rv_fit_unknown_phases(self):
        period, t0 = 0.6, 12.0
        phases = np.arange(-0.6, 0.62, 0.02)
        jd = t_layer.phase_to_jd(t0, period, phases)
        xs = {comp: jd for comp in BINARY_COUNTERPARTS}

        model_generator = ModelSimulator()
        model_generator.keep_out = True
        rvs = model_generator.rv_generator()

        rv_primary = RVData(
            x_data=xs['primary'],
            y_data=rvs['primary'],
            x_unit=u.d,
            y_unit=u.m / u.s
        )

        rv_secondary = RVData(
            x_data=xs['secondary'],
            y_data=rvs['secondary'],
            x_unit=u.d,
            y_unit=u.m / u.s
        )

        initial_parameters = \
            {
                'system': {
                    'eccentricity': {
                        'value': 0.1,
                        'fixed': True,
                    },
                    'inclination': {
                        'value': 90.0,
                        'fixed': True,
                    },
                    'argument_of_periastron': {
                        'value': 0.0,
                        'fixed': True
                    },
                    'gamma': {
                        'value': -30000.0,  # 20000.0 is real
                        'fixed': False,
                        'min': -10000,
                        'max': -40000
                    },
                    'period': {
                        'value': 0.68,  # 0.6 is real
                        'fixed': False,
                        'min': 0.5,
                        'max': 0.7
                    },
                    'primary_minimum_time': {
                        'value': 11.5,  # 12 is real
                        'fixed': False,
                        'min': 11.0,
                        'max': 13.0
                    }
                },
                'primary': {
                    'mass': {
                        'value': 1.8,
                        'fixed': True
                    },
                },
                'secondary': {
                    'mass': {
                        'value': 1.0,
                        'fixed': True,
                    },
                },
            }

        self.model_generator.keep_out = True
        with mock.patch("elisa.analytics.binary.models.central_rv_synthetic", self.model_generator.rv_generator):
            task = BinarySystemAnalyticsTask(radial_velocities={'primary': rv_primary, 'secondary': rv_secondary})
            result = task.rv_fit.fit(x0=copy(initial_parameters), method='least_squares')
        self.assertTrue(1.0 > result['system']["r_squared"]['value'] > 0.90)

    def test_least_squares_rv_fit_std_params(self):
        """
        Test has to pass and finis in real time.
        real period = 0.6d
        """
        phases = np.arange(-0.6, 0.62, 0.02)

        model_generator = ModelSimulator()
        model_generator.keep_out = True
        rvs = model_generator.rv_generator()

        rv_primary = RVData(
            x_data=phases,
            y_data=rvs['primary'],
            x_unit=u.dimensionless_unscaled,
            y_unit=u.m / u.s
        )

        rv_secondary = RVData(
            x_data=phases,
            y_data=rvs['secondary'],
            x_unit=u.dimensionless_unscaled,
            y_unit=u.m / u.s
        )

        initial_parameters = {
            'system': {
                'eccentricity': {
                    'value': 0.1,
                    'fixed': True,
                },
                'inclination': {
                    'value': 90.0,
                    'fixed': True,
                },
                'argument_of_periastron': {
                    'value': 0.0,
                    'fixed': True
                },
                'gamma': {
                    'value': -30000.0,  # 20000.0 is real
                    'fixed': False,
                    'min': -10000,
                    'max': -40000
                },
                'period': {
                    'value': 0.6,
                    'fixed': True
                }
            },
            'primary': {
                'mass': {
                    'value': 1.2,  # 1.8 is real
                    'fixed': False,
                    'min': 1,
                    'max': 3
                },
            },
            'secondary': {
                'mass': {
                    'value': 1.0,
                    'fixed': True,
                },
            },
        }
        self.model_generator.keep_out = True
        with mock.patch("elisa.analytics.binary.models.central_rv_synthetic", self.model_generator.rv_generator):
            task = BinarySystemAnalyticsTask(radial_velocities={'primary': rv_primary, 'secondary': rv_secondary})
            result = task.rv_fit.fit(x0=copy(initial_parameters), method='least_squares')

        self.assertTrue(1.0 > result['system']["r_squared"]['value'] > 0.95)

    def test_least_squares_rv_fit_community_params(self):
        phases = np.arange(-0.6, 0.62, 0.02)

        model_generator = ModelSimulator()
        model_generator.keep_out = True
        rvs = model_generator.rv_generator()

        rv_primary = RVData(
            x_data=phases,
            y_data=rvs['primary'],
            x_unit=u.dimensionless_unscaled,
            y_unit=u.m / u.s
        )

        rv_secondary = RVData(
            x_data=phases,
            y_data=rvs['secondary'],
            x_unit=u.dimensionless_unscaled,
            y_unit=u.m / u.s
        )

        initial_parameters = {
            'system': {
                'eccentricity': {
                    'value': 0.1,
                    'fixed': True,
                },
                'asini': {
                    'value': 20.0,  # 4.219470628180749
                    'fixed': False,
                    'min': 1.0,
                    'max': 100
                },
                'mass_ratio': {
                    'value': 0.8,  # 1.0 / 1.8
                    'fixed': False,
                    'min': 0,
                    'max': 2
                },
                'argument_of_periastron': {
                    'value': 0.0,
                    'fixed': True
                },
                'gamma': {
                    'value': -20000.0,
                    'fixed': True
                },
                'period': {
                    'value': 0.6,
                    'fixed': True
                }
            },
        }

        self.model_generator.keep_out = True
        with mock.patch("elisa.analytics.binary.models.central_rv_synthetic", self.model_generator.rv_generator):
            task = BinarySystemAnalyticsTask(radial_velocities={'primary': rv_primary, 'secondary': rv_secondary})
            result = task.rv_fit.fit(x0=copy(initial_parameters), method='least_squares')

        self.assertTrue(1.0 > result['system']["r_squared"]['value'] > 0.95)


class LeastSqaureLCTestCase(AbstractFitTestCase):
    def test_least_squares_lc_fit_std_params(self):
        dinit = {
            'system': {
                'inclination': {
                    'value': 90.0,
                    'fixed': True
                },
                'eccentricity': {
                    'value': 0.0,
                    'fixed': True
                },
                'argument_of_periastron': {
                    'value': 0.0,
                    'fixed': True
                },
                'period': {
                    'value': 3.0,
                    'fixed': True
                },
            },
            'primary': {
                'mass': {
                    'value': 1.8,  # 2.0
                    'fixed': False,
                    'min': 1.5,
                    'max': 2.2
                },
                't_eff': {
                    'value': 5000.0,
                    'fixed': True
                },
                'surface_potential': {
                    'value': 5.0,
                    'fixed': True
                },
                'gravity_darkening': {
                    'value': 1.0,
                    'fixed': True
                },
                'albedo': {
                    'value': 1.0,
                    'fixed': True
                },
            },
            'secondary': {
                'mass': {
                    'value': 1.0,
                    'fixed': True
                },
                't_eff': {
                    'value': 7000,  # 7000
                    'fixed': True,
                    # 'min': 5000.0,
                    # 'max': 10000.0
                },
                'surface_potential': {
                    'value': 5,
                    'fixed': True
                },
                'gravity_darkening': {
                    'value': 1.0,
                    'fixed': True
                },
                'albedo': {
                    'value': 1.0,
                    'fixed': True
                },
            },
        }

        lc_v = LCData(
            x_data=self.phases['Generic.Bessell.V'],
            y_data=self.flux['Generic.Bessell.V'],
            x_unit=u.dimensionless_unscaled,
            y_unit=u.dimensionless_unscaled
        )

        lc_b = LCData(
            x_data=self.phases['Generic.Bessell.B'],
            y_data=self.flux['Generic.Bessell.B'],
            x_unit=u.dimensionless_unscaled,
            y_unit=u.dimensionless_unscaled
        )
        self.model_generator.keep_out = True
        with mock.patch("elisa.analytics.binary.models.synthetic_binary", self.model_generator.lc_generator):
            task = BinarySystemAnalyticsTask(light_curves={'Generic.Bessell.V': lc_v, 'Generic.Bessell.B': lc_b})
            result = task.lc_fit.fit(x0=copy(dinit), method='least_squares', discretization=10)

        self.assertTrue(1.0 > result['system']["r_squared"]['value'] > 0.9)

    def test_least_squares_lc_fit_community_params(self):
        dinit = {
            'system': {
                'semi_major_axis': {
                    'value': 11.0,  # 12.62
                    'fixed': False,
                    'min': 7.0,
                    'max': 15.0
                },
                'mass_ratio': {
                    'value': 0.7,  # 0.5
                    'fixed': False,
                    'min': 0.3,
                    'max': 2.0
                },
                'inclination': {
                    'value': 90.0,
                    'fixed': True
                },
                'eccentricity': {
                    'value': 0.0,
                    'fixed': True
                },
                'argument_of_periastron': {
                    'value': 0.0,
                    'fixed': True
                },
                'period': {
                    'value': 3.0,
                    'fixed': True
                },
            },
            'primary': {
                't_eff': {
                    'value': 5000.0,
                    'fixed': True
                },
                'surface_potential': {
                    'value': 5.0,
                    'fixed': True
                },
                'gravity_darkening': {
                    'value': 1.0,
                    'fixed': True
                },
                'albedo': {
                    'value': 1.0,
                    'fixed': True
                },
            },
            'secondary': {
                't_eff': {
                    'value': 7000.0,
                    'fixed': True
                },
                'surface_potential': {
                    'value': 5,
                    'fixed': True
                },
                'gravity_darkening': {
                    'value': 1.0,
                    'fixed': True
                },
                'albedo': {
                    'value': 1.0,
                    'fixed': True
                },
            },
        }

        lc_v = LCData(
            x_data=self.phases['Generic.Bessell.V'],
            y_data=self.flux['Generic.Bessell.V'],
            x_unit=u.dimensionless_unscaled,
            y_unit=u.dimensionless_unscaled
        )

        lc_b = LCData(
            x_data=self.phases['Generic.Bessell.B'],
            y_data=self.flux['Generic.Bessell.B'],
            x_unit=u.dimensionless_unscaled,
            y_unit=u.dimensionless_unscaled
        )

        with mock.patch("elisa.analytics.binary.models.synthetic_binary", self.model_generator.lc_generator):
            task = BinarySystemAnalyticsTask(light_curves={'Generic.Bessell.V': lc_v, 'Generic.Bessell.B': lc_b})
            result = task.lc_fit.fit(x0=copy(dinit), method='least_squares', discretization=10)

        self.assertTrue(1.0 > result['system']["r_squared"]['value'] > 0.9)


class ModelSimulator(object):
    flux = {'Generic.Bessell.V': np.array([0.98128349, 0.97901564, 0.9776404, 0.77030991, 0.38623294,
                                           0.32588823, 0.38623294, 0.77030991, 0.9776404, 0.97901564,
                                           0.98128349, 0.9831816, 0.98542223, 0.9880625, 0.99034951,
                                           0.99261368, 0.99453225, 0.99591341, 0.9972921, 0.99865607,
                                           0.99943517, 0.99978567, 1., 0.99970989, 0.99963265,
                                           0.99967025, 0.9990695, 0.99904945, 0.96259235, 0.8771112,
                                           0.83958173, 0.8771112, 0.96259235, 0.99904945, 0.9990695,
                                           0.99967025, 0.99963265, 0.99970989, 1., 0.99978567,
                                           0.99943517, 0.99865607, 0.9972921, 0.99591341, 0.99453225,
                                           0.99261368, 0.99034951, 0.9880625, 0.98542223, 0.9831816,
                                           0.98128349, 0.97901564, 0.9776404, 0.77030991, 0.38623294,
                                           0.32588823, 0.38623294, 0.77030991, 0.9776404, 0.97901564,
                                           0.98128349]),
            'Generic.Bessell.B': np.array([0.80924345, 0.80729325, 0.80604709, 0.60603475, 0.2294959,
                                           0.17384023, 0.2294959, 0.60603475, 0.80604709, 0.80729325,
                                           0.80924345, 0.81088916, 0.81276665, 0.81488617, 0.81664783,
                                           0.81831472, 0.81957938, 0.82037431, 0.82105228, 0.82161889,
                                           0.82171702, 0.82140855, 0.82099437, 0.82019232, 0.81957921,
                                           0.81911052, 0.81821162, 0.81784563, 0.79824012, 0.7489621,
                                           0.72449315, 0.7489621, 0.79824012, 0.81784563, 0.81821162,
                                           0.81911052, 0.81957921, 0.82019232, 0.82099437, 0.82140855,
                                           0.82171702, 0.82161889, 0.82105228, 0.82037431, 0.81957938,
                                           0.81831472, 0.81664783, 0.81488617, 0.81276665, 0.81088916,
                                           0.80924345, 0.80729325, 0.80604709, 0.60603475, 0.2294959,
                                           0.17384023, 0.2294959, 0.60603475, 0.80604709, 0.80729325,
                                           0.80924345])}

    rv = {'primary': -1 * np.array([111221.02018955, 102589.40515112, 92675.34114568,
                                    81521.98280508, 69189.28515476, 55758.52165462,
                                    41337.34984718, 26065.23187763, 10118.86370365,
                                    -6282.93249474, -22875.63138097, -39347.75579673,
                                    -55343.14273712, -70467.93174445, -84303.24303593,
                                    -96423.915992, -106422.70195531, -113938.05716509,
                                    -118682.43573797, -120467.13803823, -119219.71866465,
                                    -114990.89801808, -107949.71016039, -98367.77975255,
                                    -86595.51823899, -73034.14124119, -58107.52819161,
                                    -42237.21613808, -25822.61388457, -9227.25025377,
                                    7229.16722243, 23273.77242388, 38679.82691287,
                                    53263.47669152, 66879.02978007, 79413.57620399,
                                    90781.53548261, 100919.51001721, 109781.66096297,
                                    117335.70723602, 123559.57210929, 128438.6567666,
                                    131963.69775175, 134129.15836278, 134932.10727626,
                                    134371.54717101, 132448.1692224, 129164.5242993,
                                    124525.61727603, 118539.94602416, 111221.02018955,
                                    102589.40515112, 92675.34114568, 81521.98280508,
                                    69189.28515476, 55758.52165462, 41337.34984718,
                                    26065.23187763, 10118.86370365, -6282.93249474,
                                    -22875.63138097]),
          'secondary': -1 * np.array([-144197.83633559, -128660.92926642, -110815.61405663,
                                      -90739.56904355, -68540.71327298, -44365.33897272,
                                      -18407.22971932, 9082.58262586, 37786.04533903,
                                      67309.27849613, 97176.13649135, 126825.96043971,
                                      155617.65693242, 182842.27714561, 207745.83747028,
                                      229563.04879121, 247560.86352515, 261088.50290277,
                                      269628.38433395, 272840.84847442, 270595.49360196,
                                      262983.61643814, 250309.4782943, 233062.00356019,
                                      211871.93283578, 187461.45423975, 160593.55075051,
                                      132026.98905414, 102480.70499782, 72609.05046238,
                                      42987.49900522, 14107.20964261, -13623.68843756,
                                      -39874.25803914, -64382.25359853, -86944.43716158,
                                      -107406.76386309, -125655.11802538, -141606.98972774,
                                      -155204.27301923, -166407.22979112, -175189.58217428,
                                      -181534.65594755, -185432.4850474, -186877.79309168,
                                      -185868.78490222, -182406.70459473, -176496.14373313,
                                      -168146.11109126, -157371.90283788, -144197.83633559,
                                      -128660.92926641, -110815.61405663, -90739.56904355,
                                      -68540.71327297, -44365.33897271, -18407.22971932,
                                      9082.58262586, 37786.04533903, 67309.27849613,
                                      97176.13649135])}

    lc_mean = np.mean(np.abs(list(flux.values())))
    rv_mean = np.mean(np.abs(list(rv.values())))

    def __init__(self):
        self.error = 0.05
        self.step = 1
        self.args = []

    def lc_generator(self, *args, **kwargs):
        add = self.lc_mean * self.error
        flux = {band: self.flux[band] + np.random.normal(0, add, len(self.flux[band]))
                for band in self.flux}
        return flux

    def rv_generator(self, *args, **kwargs):
        add = self.rv_mean * self.error
        rv = {component: self.rv[component] + np.random.normal(0, add, len(self.rv[component]))
              for component in BINARY_COUNTERPARTS}
        return rv


class ConstraintsTestCase(ElisaTestCase):
    basic_x0 = {
        'system': {
            'a': {"value": 50, 'fixed': False},
            "b": {"value": 2, 'fixed': False},
        },
        'primary': {
            "c": {"value": 1.25},
            "c1": {"value": None, "constraint": "2 * {a}"},
            'spots': {
                'spot1': {
                    "c2": {
                        "value": 50,
                        "fixed": False
                    },
                    "c3": {
                        "value": None,
                        "constraint": "{spot1.c2} + {b}"
                    },
                    "c4": {
                        "value": None,
                        "constraint": "({a} + {b}) * 2.0"
                    }
                }
            }
        },
    }

    extended_x0 = {
        'system': {
            "a": {"value": 50, 'fixed': False},
            "b": {"value": 2, 'fixed': False},
        },
        'primary': {
            "c1": {
                "value": None,
                "constraint": "{a} * sin({b})"
            },
            "c2": {
                "value": None,
                "constraint": "{a} * log10({b})"
            }
        }
    }

    def test_constraints_validator_basic(self):
        params.constraints_validator(self.basic_x0)

    def test_constraints_evaluator_basic(self):
        expected = {params.PARAM_PARSER.join(['primary', 'c1']): 100,
                    params.PARAM_PARSER.join(['primary', 'spots', 'spot1', 'c3']): 52,
                    params.PARAM_PARSER.join(['primary', 'spots', 'spot1', 'c4']): 104.0}
        constraints = params.x0_to_constrained_kwargs(self.basic_x0)
        floats = params.x0_to_variable_kwargs(self.basic_x0)
        obtained = params.constraints_evaluator(floats, constraints)
        self.assertDictEqual(expected, obtained)

    def test_constraints_validator_extended(self):
        params.constraints_validator(self.extended_x0)

    def test_constraints_evaluator_extended(self):
        expected = {params.PARAM_PARSER.join(['primary', 'c1']): np.round(45.46487134128409, 3),
                    params.PARAM_PARSER.join(['primary', 'c2']): np.round(15.05149978319906, 3)}
        constraints = params.x0_to_constrained_kwargs(self.extended_x0)
        floats = params.x0_to_variable_kwargs(self.extended_x0)
        obtained = params.constraints_evaluator(floats, constraints)
        obtained = {key: np.round(val, 3) for key, val in obtained.items()}
        self.assertDictEqual(expected, obtained)
