import numpy as np

from copy import copy
from numpy.testing import assert_array_equal
from elisa.analytics.binary import params
from elisa.base.error import InitialParamsError
from unittests.utils import ElisaTestCase
from elisa.analytics.binary.least_squares import central_rv

TOL = 1e-5


class TestParamsTestCase(ElisaTestCase):
    def setUp(self):
        self.x0 = [
            {
                'value': 2.0,
                'param': 'p__mass',
                'fixed': False,
                'min': 1.0,
                'max': 3.0
            },
            {
                'value': 4000.0,
                'param': 'p__t_eff',
                'fixed': True,
                'min': 3500.0,
                'max': 4500.0
            },
            {
                'value': 1.1,
                'param': 'argument_of_periastron',
                'fixed': False
            }
        ]

        params.NORMALIZATION_MAP.update({
            'p__mass': (0.5, 20),
            'p__t_eff': (np.min(params.TEMPERATURES), np.max(params.TEMPERATURES)),
            'argument_of_periastron': (0, 360)
        })

    def test_x0_vectorize(self):
        x0, labels = params.x0_vectorize(self.x0)
        expected_x0 = (2.0, 1.1)
        expected_labels = ['p__mass', 'argument_of_periastron']

        assert_array_equal(x0, expected_x0)
        assert_array_equal(labels, expected_labels)

    def test_x0_to_kwargs(self):
        obtained = params.x0_to_kwargs(self.x0)
        expected = {'p__mass': 2.0, 'p__t_eff': 4000.0, 'argument_of_periastron': 1.1}
        self.assertDictEqual(obtained, expected)

    def test_x0_to_fixed_kwargs(self):
        obtained = params.x0_to_fixed_kwargs(self.x0)
        expected = {'p__t_eff': 4000.0}
        self.assertDictEqual(obtained, expected)

    def test_serialize_param_boundaries(self):
        obtained = params.serialize_param_boundaries(self.x0)
        expected = {'p__mass': (1.0, 3.0), 'argument_of_periastron': (0, 360)}
        self.assertDictEqual(obtained, expected)

    def test_param_normalizer(self):
        boundaries = params.serialize_param_boundaries(self.x0)
        params.update_normalization_map(boundaries)

        x0, labels = params.x0_vectorize(self.x0)
        obtained = params.param_normalizer(x0, labels)
        expected = np.array([0.5, 0.003055555])

        self.assertTrue(np.all(np.abs(expected - obtained) < TOL))

    def test_param_renormalizer(self):
        boundaries = params.serialize_param_boundaries(self.x0)
        params.update_normalization_map(boundaries)

        values = [0.5, 0.1]
        labels = ['p__mass', 'argument_of_periastron']

        obtained = params.param_renormalizer(values, labels)
        expected = np.array([[2.0, 36.0]])
        self.assertTrue(np.all(np.abs(expected - obtained) < TOL))

    def test_extend_result_with_units(self):
        result = [
            {
                'value': 2.0,
                'param': 'p__mass'
            },
            {
                'value': 4000.0,
                'param': 'p__t_eff'
            }]

        obtained = params.extend_result_with_units(result)
        expected = [{'value': 2.0, 'param': 'p__mass', 'unit': 'solMass'},
                    {'value': 4000.0, 'param': 'p__t_eff', 'unit': 'K'}]
        assert_array_equal(expected, obtained)

    def test_is_overcontact(self):
        self.assertTrue(params.is_overcontact('over-contact'))
        self.assertFalse(params.is_overcontact('overcontact'))
        self.assertFalse(params.is_overcontact('detached'))

    def test_adjust_result_constrained_potential(self):
        hash_map = {'p__surface_potential': 0, 's__surface_potential': 1}
        xn = [
            {
                "value": 1.1,
                "param": "p__surface_potential",
                "min": 1.0,
                "max": 10.0
            },
            {
                "value": 1.2,
                "param": "s__surface_potential",
                "min": 1.1,
                "max": 11.0
            }
        ]

        expected = [{'value': 1.1, 'param': 'p__surface_potential', 'min': 1.0, 'max': 10.0},
                    {'param': 's__surface_potential', 'value': 1.1, 'min': 1.0, 'max': 10.0}]
        obtained = params.adjust_result_constrained_potential(xn, hash_map)

        self.assertTrue(expected[0]['value'] == obtained[0]['value'])
        self.assertTrue(expected[0]['min'] == obtained[0]['min'])
        self.assertTrue(expected[0]['max'] == obtained[0]['max'])

    def test_initial_x0_validity_check(self):
        xn = [
            {
                "value": 1.1,
                "param": "p__surface_potential",
                "min": 1.0,
                "max": 10.0
            },
            {
                "value": 1.2,
                "param": "s__surface_potential",
                "min": 1.1,
                "max": 11.0
            }
        ]

        params.initial_x0_validity_check(xn, morphology='over-contact')
        params.initial_x0_validity_check(xn, morphology='detached')

        xn[0]["fixed"] = True
        xn[1]["fixed"] = False
        with self.assertRaises(InitialParamsError) as context:
            params.initial_x0_validity_check(xn, morphology='over-contact')
        self.assertTrue("just one fixed potential" in str(context.exception).lower())

        xn[1]["fixed"] = True
        with self.assertRaises(InitialParamsError) as context:
            params.initial_x0_validity_check(xn, morphology='over-contact')
        self.assertTrue("different potential" in str(context.exception).lower())

        xn[0]["fixed"] = False
        xn[1]["fixed"] = False
        params.initial_x0_validity_check(xn, morphology='over-contact')

        self.assertTrue(xn[0]['min'] == xn[1]['min'])
        self.assertTrue(xn[0]['max'] == xn[1]['max'])

        self.assertTrue(xn[1]['fixed'])
        self.assertFalse(xn[0].get('fixed', False))


class LeastSqaureLCTestCase(ElisaTestCase):
    pass


class McMcLCTestCase(ElisaTestCase):
    pass


class LeastSqaureRVTestCase(ElisaTestCase):
    rv = {'primary': np.array([111221.02018955, 102589.40515112, 92675.34114568,
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
          'secondary': np.array([-144197.83633559, -128660.92926642, -110815.61405663,
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

    def test_least_squares_rv_fit_std_params(self):
        """
        Test has to pass and finis in real time.
        real period = 0.6d
        """
        phases = np.arange(-0.6, 0.62, 0.02)
        initial_parameters = [
            {
                'value': 0.1,
                'param': 'eccentricity',
                'fixed': True,

            },
            {
                'value': 90.0,
                'param': 'inclination',
                'fixed': True,

            },
            {
                'value': 1.2,  # 1.8 is real
                'param': 'p__mass',
                'fixed': False,
                'min': 1,
                'max': 3
            },
            {
                'value': 1.0,
                'param': 's__mass',
                'fixed': True,
            },
            {
                'value': 0.0,
                'param': 'argument_of_periastron',
                'fixed': True
            },
            {
                'value': 30000.0,  # 20000.0 is real
                'param': 'gamma',
                'fixed': False,
                'min': 20000,
                'max': 40000
            }
        ]

        result = central_rv.fit(xs=phases, ys=self.rv, period=0.6, x0=copy(initial_parameters))
        self.assertTrue(1.0 > result[-1]["r_squared"] > 0.9)

    def test_least_squares_rv_fit_community_params(self):
        phases = np.arange(-0.6, 0.62, 0.02)
        initial_parameters = [
            {
                'value': 0.1,
                'param': 'eccentricity',
                'fixed': True,

            },
            {
                'value': 20.0,  # 4.219470628180749
                'param': 'asini',
                'fixed': False,
                'min': 1.0,
                'max': 100

            },
            {
                'value': 0.8,  # 1.0 / 1.8
                'param': 'mass_ratio',
                'fixed': False,
                'min': 0,
                'max': 2
            },
            {
                'value': 0.0,
                'param': 'argument_of_periastron',
                'fixed': True
            },
            {
                'value': 20000.0,
                'param': 'gamma',
                'fixed': True
            }
        ]

        result = central_rv.fit(xs=phases, ys=self.rv, period=0.6, x0=copy(initial_parameters))
        self.assertTrue(1.0 > result[-1]["r_squared"] > 0.9)
