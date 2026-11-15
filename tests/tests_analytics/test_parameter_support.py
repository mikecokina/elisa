# The tests package selects Astropy constants before test-module imports.
import numpy as np
from numpy.testing import assert_array_equal

from elisa.analytics.params import parameters
from elisa.analytics.params.parameters import xs_reducer
from tests import set_astropy_units
from tests.utils import ElisaTestCase

set_astropy_units()

TOL = 1e-5


class ParamsSupportTestCase(ElisaTestCase):
    def test_xs_reducer_all_same(self) -> None:
        a = np.linspace(0, 1, 5)
        xs = {"a": a, "b": a, "c": a}
        new_xs, inverse = xs_reducer(xs)
        for band, phases in xs.items():
            assert_array_equal(inverse[band], np.arange(len(phases)))
        assert_array_equal(new_xs, a)

    def test_xs_reducer_random(self) -> None:
        a = np.linspace(0, 1, 5)
        b = np.array([*np.array(a + 0.01).tolist(), 0.021])
        c = a + 0.02
        xs = {"a": a, "b": b, "c": c}
        new_xs, inverse = xs_reducer(xs)
        expected_xs = [0.0, 0.01, 0.02, 0.021, 0.25, 0.26, 0.27, 0.5, 0.51, 0.52, 0.75, 0.76, 0.77, 1.0, 1.01, 1.02]
        expected_inverse = {"a": [0, 4, 7, 10, 13], "b": [1, 5, 8, 11, 14, 3], "c": [2, 6, 9, 12, 15]}

        for band in xs:
            assert_array_equal(inverse[band], expected_inverse[band])
        assert_array_equal(expected_xs, new_xs)

    def test_vector_renormalizer(self) -> None:
        normalization = {"param1": (0, 10), "param2": (1.0, 2.0)}
        obtained = parameters.vector_renormalizer([0.5, 0.5], ("param1", "param2"), normalization)
        expected = np.array([5.0, 1.5])
        assert np.all(np.abs(expected - obtained) < TOL)

    def test_vector_normalizer(self) -> None:
        normalization = {"param1": (0, 10), "param2": (1.0, 2.0)}
        properties = ["param1", "param2"]
        obtained = parameters.vector_normalizer([5.0, 1.5], properties, normalization)
        expected = np.array([[0.5, 0.5]])
        assert np.all(np.abs(expected - obtained) < TOL)

    def test_serialize_result_rv_community(self) -> None:
        result_dict = {
            "system@eccentricity": {
                "value": 0.1,
                "confidence_interval": {
                    "min": 0.12,
                    "max": 0.08,
                },
                "fixed": False,
                "min": 0.0,
                "max": 0.5,
                "unit": None,
            },
            "system@gamma": {
                "value": 10000,
                "confidence_interval": {
                    "min": 12000,
                    "max": 8000,
                },
                "fixed": False,
                "min": 10000.0,
                "max": 50000.0,
                "unit": "m / s",
            },
            "system@mass_ratio": {
                "value": 0.5,
                "confidence_interval": {
                    "min": 0.00,
                    "max": 0.00,
                },
                "fixed": False,
                "min": 0.1,
                "max": 10.0,
                "unit": None,
            },
            "system@asini": {
                "value": 17.0,
                "confidence_interval": {
                    "min": 18,
                    "max": 16,
                },
                "fixed": False,
                "min": 10.0,
                "max": 20.0,
                "unit": "solRad",
            },
            "system@argument_of_periastron": {
                "value": 0.0,
                "fixed": True,
                "unit": "deg",
            },
            "system@period": {
                "value": 4.5,
                "fixed": True,
                "unit": "d",
            },
            "r_squared": {
                "value": 0.98765,
                "unit": None,
            },
        }
        expected = {
            "system": {
                "eccentricity": {
                    "value": 0.1,
                    "confidence_interval": {
                        "min": 0.12,
                        "max": 0.08,
                    },
                    "fixed": False,
                    "min": 0.0,
                    "max": 0.5,
                    "unit": None,
                },
                "gamma": {
                    "value": 10000,
                    "confidence_interval": {
                        "min": 12000,
                        "max": 8000,
                    },
                    "fixed": False,
                    "min": 10000.0,
                    "max": 50000.0,
                    "unit": "m / s",
                },
                "mass_ratio": {
                    "value": 0.5,
                    "confidence_interval": {
                        "min": 0.0,
                        "max": 0.0,
                    },
                    "fixed": False,
                    "min": 0.1,
                    "max": 10.0,
                    "unit": None,
                },
                "asini": {
                    "value": 17.0,
                    "confidence_interval": {
                        "min": 18,
                        "max": 16,
                    },
                    "fixed": False,
                    "min": 10.0,
                    "max": 20.0,
                    "unit": "solRad",
                },
                "argument_of_periastron": {
                    "value": 0.0,
                    "fixed": True,
                    "unit": "deg",
                },
                "period": {
                    "value": 4.5,
                    "fixed": True,
                    "unit": "d",
                },
            },
            "r_squared": {
                "value": 0.98765,
                "unit": None,
            },
        }
        serialized = parameters.serialize_result(result_dict)
        self.assertDictEqual(serialized, expected)

    def test_serialize(self) -> None:
        result_dict = {
            "system@eccentricity": {
                "value": 0.1,
                "confidence_interval": {
                    "min": 0.12,
                    "max": 0.08,
                },
                "fixed": False,
                "min": 0.0,
                "max": 0.5,
                "unit": None,
            },
            "system@period": {
                "value": 4.5,
                "fixed": True,
                "unit": "d",
            },
            "primary@t_eff": {
                "value": 4.5,
                "fixed": True,
                "unit": "d",
            },
            "primary@surface_potential": {
                "value": 3.5,
                "fixed": False,
                "unit": None,
                "min": 2.0,
                "max": 5.5,
            },
            "secondary@spot@utopic@longitude": {
                "value": 4.5,
                "fixed": False,
                "unit": "d",
                "min": 2.0,
                "max": 5.5,
            },
            "secondary@spot@unicorn@longitude": {
                "value": 2.5,
                "fixed": False,
                "unit": "d",
                "min": 2.0,
                "max": 5.5,
            },
            "secondary@pulsation@unicorn@start_phase": {
                "value": 2.2,
                "fixed": False,
                "unit": "d",
                "min": 2.1,
                "max": 5.0,
            },
        }
        expected = {
            "system": {
                "eccentricity": {
                    "value": 0.1,
                    "confidence_interval": {
                        "min": 0.12,
                        "max": 0.08,
                    },
                    "fixed": False,
                    "min": 0.0,
                    "max": 0.5,
                    "unit": None,
                },
                "period": {
                    "value": 4.5,
                    "fixed": True,
                    "unit": "d",
                },
            },
            "primary": {
                "t_eff": {
                    "value": 4.5,
                    "fixed": True,
                    "unit": "d",
                },
                "surface_potential": {
                    "value": 3.5,
                    "fixed": False,
                    "unit": None,
                    "min": 2.0,
                    "max": 5.5,
                },
            },
            "secondary": {
                "spots": [
                    {
                        "label": "utopic",
                        "longitude": {
                            "value": 4.5,
                            "fixed": False,
                            "unit": "d",
                            "min": 2.0,
                            "max": 5.5,
                        },
                    },
                    {
                        "label": "unicorn",
                        "longitude": {
                            "value": 2.5,
                            "fixed": False,
                            "unit": "d",
                            "min": 2.0,
                            "max": 5.5,
                        },
                    },
                ],
                "pulsations": [
                    {
                        "label": "unicorn",
                        "start_phase": {
                            "value": 2.2,
                            "fixed": False,
                            "unit": "d",
                            "min": 2.1,
                            "max": 5.0,
                        },
                    },
                ],
            },
        }
        serialized = parameters.serialize_result(result_dict)
        self.assertDictEqual(serialized, expected)

    def test_deflate_phenomena(self) -> None:
        phenom_spot = {
            "secondary@spot@utopic@longitude": {
                "value": 4.5,
                "fixed": False,
                "unit": "d",
                "min": 2.0,
                "max": 5.5,
            },
            "secondary@spot@unicorn@longitude": {
                "value": 2.5,
                "fixed": False,
                "unit": "d",
                "min": 2.0,
                "max": 5.5,
            },
        }
        deflated = parameters.deflate_phenomena(phenom_spot)
        expected = {
            "utopic": {
                "label": "utopic",
                "longitude": {
                    "value": 4.5,
                    "fixed": False,
                    "unit": "d",
                    "min": 2.0,
                    "max": 5.5,
                },
            },
            "unicorn": {
                "label": "unicorn",
                "longitude": {
                    "value": 2.5,
                    "fixed": False,
                    "unit": "d",
                    "min": 2.0,
                    "max": 5.5,
                },
            },
        }
        self.assertDictEqual(deflated, expected)

    def test_deserialize_result(self) -> None:
        serialized = expected = {
            "system": {
                "eccentricity": {
                    "value": 0.1,
                    "confidence_interval": {
                        "min": 0.12,
                        "max": 0.08,
                    },
                    "fixed": False,
                    "min": 0.0,
                    "max": 0.5,
                    "unit": None,
                },
                "period": {
                    "value": 4.5,
                    "fixed": True,
                    "unit": "d",
                },
            },
            "primary": {
                "t_eff": {
                    "value": 4.5,
                    "fixed": True,
                    "unit": "d",
                },
                "surface_potential": {
                    "value": 3.5,
                    "fixed": False,
                    "unit": None,
                    "min": 2.0,
                    "max": 5.5,
                },
            },
            "secondary": {
                "spots": [
                    {
                        "label": "utopic",
                        "longitude": {
                            "value": 4.5,
                            "fixed": False,
                            "unit": "d",
                            "min": 2.0,
                            "max": 5.5,
                        },
                    },
                    {
                        "label": "unicorn",
                        "longitude": {
                            "value": 2.5,
                            "fixed": False,
                            "unit": "d",
                            "min": 2.0,
                            "max": 5.5,
                        },
                    },
                ],
                "pulsations": [
                    {
                        "label": "unicorn",
                        "start_phase": {
                            "value": 2.2,
                            "fixed": False,
                            "unit": "d",
                            "min": 2.1,
                            "max": 5.0,
                        },
                    },
                ],
            },
        }
        expected = {
            "primary@t_eff": {
                "value": 4.5,
                "fixed": True,
                "unit": "d",
            },
            "primary@surface_potential": {
                "value": 3.5,
                "fixed": False,
                "unit": None,
                "min": 2.0,
                "max": 5.5,
            },
            "secondary@spot@utopic@longitude": {
                "value": 4.5,
                "fixed": False,
                "unit": "d",
                "min": 2.0,
                "max": 5.5,
            },
            "secondary@spot@unicorn@longitude": {
                "value": 2.5,
                "fixed": False,
                "unit": "d",
                "min": 2.0,
                "max": 5.5,
            },
            "secondary@pulsation@unicorn@start_phase": {
                "value": 2.2,
                "fixed": False,
                "unit": "d",
                "min": 2.1,
                "max": 5.0,
            },
            "system@eccentricity": {
                "value": 0.1,
                "confidence_interval": {
                    "min": 0.12,
                    "max": 0.08,
                },
                "fixed": False,
                "min": 0.0,
                "max": 0.5,
                "unit": None,
            },
            "system@period": {
                "value": 4.5,
                "fixed": True,
                "unit": "d",
            },
        }
        deserialized = parameters.deserialize_result(serialized)
        self.assertDictEqual(deserialized, expected)

    def test_constraints(self) -> None:
        substitution = {
            "system@inclination": 90,
            "primary@surface_potential": 4.5,
        }
        constrained = {
            "secondary@surface_potential": "2.0 * primary@surface_potential",
        }
        evaluated = parameters.constraints_evaluator(substitution, constrained)
        assert evaluated["secondary@surface_potential"] == 9.0

    def test_prepare_properties_set(self) -> None:
        constrained = {
            "secondary@surface_potential": "2.0 * primary@surface_potential",
        }
        fixed = {
            "system@inclination": 90.0,
        }
        properties = ["primary@surface_potential", "primary@t_eff", "system@argument_of_Periastron"]
        xn = [1.5, 2.0, 3.0]
        expected = {
            "primary@surface_potential": 1.5,
            "primary@t_eff": 2.0,
            "system@argument_of_Periastron": 3.0,
            "secondary@surface_potential": 3.0,
            "system@inclination": 90.0,
        }
        properties_set = parameters.prepare_properties_set(xn, properties, constrained, fixed)
        self.assertDictEqual(properties_set, expected)
