from unittest import mock

import numpy as np

from elisa import units as u
from elisa.analytics import LCData, LCBinaryAnalyticsTask, RVData, RVBinaryAnalyticsTask
from elisa.analytics.params.parameters import BinaryInitialParameters
from elisa.binary_system import t_layer
from elisa import settings
from unittests.utils import ElisaTestCase


class AbstractFitTestCase(ElisaTestCase):
    def setUp(self):
        super(AbstractFitTestCase, self).setUp()
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
            "primary": {
                "mass": {
                    'value': 1.8,  # 2.0
                    'fixed': False,
                    'min': 1.5,
                    'max': 2.2
                },
                "t_eff": {
                    'value': 5000.0,
                    'fixed': True
                },
                "surface_potential": {
                    'value': 5.0,
                    'fixed': True
                },
                "gravity_darkening": {
                    'value': 1.0,
                    'fixed': True
                },
                "albedo": {
                    'value': 1.0,
                    'fixed': True
                }
            },

            "secondary": {
                "mass": {
                    'value': 1.0,
                    'fixed': True
                },
                "t_eff": {
                    'value': 6500.0,  # 7000
                    'fixed': False,
                    'min': 5000.0,
                    'max': 10000.0
                },
                "surface_potential": {
                    'value': 5,
                    'fixed': True
                },
                "gravity_darkening": {
                    'value': 1.0,
                    'fixed': True
                },
                "albedo": {
                    'value': 1.0,
                    'fixed': True
                }
            },
            "system": {
                "inclination": {
                    'value': 90.0,
                    'fixed': True
                },
                "eccentricity": {
                    'value': 0.0,
                    'fixed': True
                },
                "argument_of_periastron": {
                    'value': 0.0,
                    'fixed': True
                },
                "period": {
                    'value': 3.0,
                    'fixed': True
                }
            }

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
        with mock.patch("elisa.analytics.models.lc.synthetic_binary", self.model_generator.lc_generator):
            lc_initial = BinaryInitialParameters(**dinit)
            task = LCBinaryAnalyticsTask(data={'Generic.Bessell.V': lc_v, 'Generic.Bessell.B': lc_b},
                                         expected_morphology="detached", method='mcmc')
            task.fit(x0=lc_initial, nsteps=10, discretization=5.0)

    def test_mcmc_lc_fit_community_params_detached(self):
        dinit = {
            "system": {
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
            "primary": {
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
            "secondary": {
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
                }
            }
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
        with mock.patch("elisa.analytics.models.lc.synthetic_binary", self.model_generator.lc_generator):
            lc_initial = BinaryInitialParameters(**dinit)
            task = LCBinaryAnalyticsTask(data={'Generic.Bessell.V': lc_v, 'Generic.Bessell.B': lc_b},
                                         expected_morphology="detached", method='mcmc')
            task.fit(x0=lc_initial, nsteps=10, discretization=5.0)


class RVTestCase(ElisaTestCase):
    def setUp(self):
        super(RVTestCase, self).setUp()
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

        initial_parameters = {
            "system": {
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
                    'min': 0.1,
                    'max': 2.0
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

        rv_initial = BinaryInitialParameters(**initial_parameters)
        task = RVBinaryAnalyticsTask(data={'primary': rv_primary, 'secondary': rv_secondary}, method='mcmc')
        result = task.fit(x0=rv_initial, nsteps=1000, burn_in=100)
        self.assertTrue(1.0 > result["r_squared"]['value'] > 0.9)


class LeastSqaureRVTestCase(RVTestCase):
    def test_least_squares_rv_fit_unknown_phases(self):
        period, t0 = 0.6, 12.0
        phases = np.arange(-0.6, 0.62, 0.02)
        jd = t_layer.phase_to_jd(t0, period, phases)
        xs = {comp: jd for comp in settings.BINARY_COUNTERPARTS}

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

        initial_parameters = {
            "system": {
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
                    'max': -10000,
                    'min': -40000
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
            "primary": {
                'mass': {
                    'value': 1.8,
                    'fixed': True
                }
            },
            "secondary": {
                'mass': {
                    'value': 1.0,
                    'fixed': True,
                }
            }
        }

        rv_initial = BinaryInitialParameters(**initial_parameters)
        task = RVBinaryAnalyticsTask(data={'primary': rv_primary, 'secondary': rv_secondary}, method='least_squares')
        result = task.fit(x0=rv_initial)
        self.assertTrue(1.0 > result["r_squared"]['value'] > 0.9)

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
            "system": {
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
                    'max': -10000,
                    'min': -40000
                },
                'period': {
                    'value': 0.6,
                    'fixed': True
                }
            },
            "primary": {
                'mass': {
                    'value': 1.2,  # 1.8 is real
                    'fixed': False,
                    'min': 1,
                    'max': 3
                },
            },
            "secondary": {
                'mass': {
                    'value': 1.0,
                    'fixed': True,
                },
            }
        }

        rv_initial = BinaryInitialParameters(**initial_parameters)
        task = RVBinaryAnalyticsTask(data={'primary': rv_primary, 'secondary': rv_secondary}, method='least_squares')
        result = task.fit(x0=rv_initial)
        self.assertTrue(1.0 > result["r_squared"]['value'] > 0.95)

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
            "system": {
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
                    'min': 0.1,
                    'max': 2.0
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

        rv_initial = BinaryInitialParameters(**initial_parameters)
        task = RVBinaryAnalyticsTask(data={'primary': rv_primary, 'secondary': rv_secondary}, method='least_squares')
        result = task.fit(x0=rv_initial)
        self.assertTrue(1.0 > result["r_squared"]['value'] > 0.95)


class LeastSqaureLCTestCase(AbstractFitTestCase):
    def test_least_squares_lc_fit_std_params(self):
        dinit = {
            "system": {
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
                }
            },
            "primary": {
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
            "secondary": {
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

            }
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
        with mock.patch("elisa.analytics.models.lc.synthetic_binary", self.model_generator.lc_generator):
            lc_initial = BinaryInitialParameters(**dinit)
            data = {'Generic.Bessell.V': lc_v, 'Generic.Bessell.B': lc_b}
            task = LCBinaryAnalyticsTask(data=data, method='least_squares', expected_morphology='detached')
            result = task.fit(x0=lc_initial)

        self.assertTrue(1.0 > result["r_squared"]['value'] > 0.9)

    def test_least_squares_lc_fit_community_params(self):
        dinit = {
            "system":
                {
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
                    }
                },
            "primary": {
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
                }
            },
            "secondary": {
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
                }
            }
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
        with mock.patch("elisa.analytics.models.lc.synthetic_binary", self.model_generator.lc_generator):
            lc_initial = BinaryInitialParameters(**dinit)
            data = {'Generic.Bessell.V': lc_v, 'Generic.Bessell.B': lc_b}
            task = LCBinaryAnalyticsTask(data=data, method='least_squares', expected_morphology='detached')
            result = task.fit(x0=lc_initial)

        self.assertTrue(1.0 > result["r_squared"]['value'] > 0.9)


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
              for component in settings.BINARY_COUNTERPARTS}
        return rv
