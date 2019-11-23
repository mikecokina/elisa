import numpy as np
from numpy.testing import assert_array_equal

from elisa.analytics.binary import params
from elisa.base.error import InitialParamsError
from unittests.utils import ElisaTestCase

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
