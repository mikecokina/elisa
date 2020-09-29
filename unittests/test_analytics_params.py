import numpy as np
from numpy.testing import assert_array_equal
from parameterized import parameterized

from elisa import units as u
from elisa.analytics.params import conf, parameters
from elisa.analytics.params.parameters import xs_reducer
from elisa import settings
from unittests.utils import ElisaTestCase

TOL = 1e-5


class ConfTestCase(ElisaTestCase):
    @staticmethod
    def test_DEFAULT_FLOAT_UNITS():
        # this is important, do not mess up default units in analytics
        expected = {
            'inclination': u.deg,
            'eccentricity': None,
            'argument_of_periastron': u.deg,
            'gamma': u.VELOCITY_UNIT,
            'mass': u.solMass,
            't_eff': u.TEMPERATURE_UNIT,
            'metallicity': None,
            'surface_potential': None,
            'albedo': None,
            'gravity_darkening': None,
            'synchronicity': None,
            'mass_ratio': None,
            'semi_major_axis': u.solRad,
            'asini': u.solRad,
            'period': u.PERIOD_UNIT,
            'primary_minimum_time': u.PERIOD_UNIT,
            'additional_light': None,
            'phase_shift': None,
            # SPOTS
            'latitude': u.deg,
            'longitude': u.deg,
            'angular_radius': u.deg,
            'temperature_factor': None,
            # PULSATIONS
            'l': None,
            'm': None,
            'amplitude': u.VELOCITY_UNIT,
            'frequency': u.FREQUENCY_UNIT,
            'start_phase': u.deg,
            'mode_axis_theta': u.deg,
            'mode_axis_phi': u.deg,
        }
        # in python3.6 >= order is maintain
        assert_array_equal(list(conf.DEFAULT_FLOAT_UNITS.values()), list(expected.values()))
        assert_array_equal(list(conf.DEFAULT_FLOAT_UNITS.keys()), list(expected.keys()))

    @staticmethod
    def test_COMPOSITE_FLAT_PARAMS():
        assert_array_equal(conf.COMPOSITE_FLAT_PARAMS, ['spot', 'pulsation'])

    @staticmethod
    def test_DEFAULT_FLOAT_ANGULAR_UNIT():
        assert conf.DEFAULT_FLOAT_ANGULAR_UNIT, u.deg

    @staticmethod
    def test_DEFAULT_FLOAT_MASS_UNIT():
        assert conf.DEFAULT_FLOAT_MASS_UNIT, u.solMass


class ParamsSupportTestCase(ElisaTestCase):
    def test_xs_reducer_all_same(self):
        a = np.linspace(0, 1, 5)
        xs = {"a": a, "b": a, "c": a}
        new_xs, inverse = xs_reducer(xs)
        for band, phases in xs.items():
            assert_array_equal(inverse[band], np.arange(len(phases)))
        assert_array_equal(new_xs, a)

    def test_xs_reducer_random(self):
        a = np.linspace(0, 1, 5)
        b = np.array(np.array(a + 0.01).tolist() + [0.021])
        c = a + 0.02
        xs = {"a": a, "b": b, "c": c}
        new_xs, inverse = xs_reducer(xs)
        expected_xs = [0., 0.01, 0.02, 0.021, 0.25, 0.26, 0.27, 0.5, 0.51, 0.52, 0.75, 0.76, 0.77, 1., 1.01, 1.02]
        expected_inverse = {'a': [0, 4, 7, 10, 13], 'b': [1, 5, 8, 11, 14, 3], 'c': [2, 6, 9, 12, 15]}

        for band, phases in xs.items():
            assert_array_equal(inverse[band], expected_inverse[band])
        assert_array_equal(expected_xs, new_xs)

    def test_vector_renormalizer(self):
        normalization = {"param1": (0, 10), "param2": (1.0, 2.0)}
        obtained = parameters.vector_renormalizer([0.5, 0.5], ("param1", "param2"), normalization)
        expected = np.array([5.0, 1.5])
        self.assertTrue(np.all(np.abs(expected - obtained) < TOL))

    def test_vector_normalizer(self):
        normalization = {"param1": (0, 10), "param2": (1.0, 2.0)}
        properties = ['param1', 'param2']
        obtained = parameters.vector_normalizer([5.0, 1.5], properties, normalization)
        expected = np.array([[0.5, 0.5]])
        self.assertTrue(np.all(np.abs(expected - obtained) < TOL))

    def test_serialize_result_rv_community(self):
        result_dict = {
            "system@eccentricity": {
                "value": 0.1,
                "confidence_interval": {
                    "min": 0.12,
                    "max": 0.08
                },
                "fixed": False,
                "min": 0.0,
                "max": 0.5,
                "unit": None
            },
            "system@gamma": {
                "value": 10000,
                "confidence_interval": {
                    "min": 12000,
                    "max": 8000
                },
                "fixed": False,
                "min": 10000.0,
                "max": 50000.0,
                "unit": "m / s"
            },
            "system@mass_ratio": {
                "value": 0.5,
                "confidence_interval": {
                    "min": 0.00,
                    "max": 0.00
                },
                "fixed": False,
                "min": 0.1,
                "max": 10.0,
                "unit": None
            },
            "system@asini": {
                "value": 17.0,
                "confidence_interval": {
                    "min": 18,
                    "max": 16
                },
                "fixed": False,
                "min": 10.0,
                "max": 20.0,
                "unit": "solRad"
            },
            "system@argument_of_periastron": {
                "value": 0.0,
                "fixed": True,
                "unit": "deg"
            },
            "system@period": {
                "value": 4.5,
                "fixed": True,
                "unit": "d"
            },
            "r_squared": {
                "value": 0.98765,
                "unit": None
            }
        }
        expected = {
            "system": {
                "eccentricity": {
                    "value": 0.1,
                    "confidence_interval": {
                        "min": 0.12,
                        "max": 0.08
                    },
                    "fixed": False,
                    "min": 0.0,
                    "max": 0.5,
                    "unit": None
                },
                "gamma": {
                    "value": 10000,
                    "confidence_interval": {
                        "min": 12000,
                        "max": 8000
                    },
                    "fixed": False,
                    "min": 10000.0,
                    "max": 50000.0,
                    "unit": "m / s"
                },
                "mass_ratio": {
                    "value": 0.5,
                    "confidence_interval": {
                        "min": 0.0,
                        "max": 0.0
                    },
                    "fixed": False,
                    "min": 0.1,
                    "max": 10.0,
                    "unit": None
                },
                "asini": {
                    "value": 17.0,
                    "confidence_interval": {
                        "min": 18,
                        "max": 16
                    },
                    "fixed": False,
                    "min": 10.0,
                    "max": 20.0,
                    "unit": "solRad"
                },
                "argument_of_periastron": {
                    "value": 0.0,
                    "fixed": True,
                    "unit": "deg"
                },
                "period": {
                    "value": 4.5,
                    "fixed": True,
                    "unit": "d"
                }
            },
            "r_squared": {
                "value": 0.98765,
                "unit": None
            }
        }
        serialized = parameters.serialize_result(result_dict)
        self.assertDictEqual(serialized, expected)

    def test_serialize(self):
        result_dict = {
            "system@eccentricity": {
                "value": 0.1,
                "confidence_interval": {
                    "min": 0.12,
                    "max": 0.08
                },
                "fixed": False,
                "min": 0.0,
                "max": 0.5,
                "unit": None
            },
            "system@period": {
                "value": 4.5,
                "fixed": True,
                "unit": "d"
            },
            "primary@t_eff": {
                "value": 4.5,
                "fixed": True,
                "unit": "d"
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
            }
        }
        expected = {
            "system": {
                "eccentricity": {
                    "value": 0.1,
                    "confidence_interval": {
                        "min": 0.12,
                        "max": 0.08
                    },
                    "fixed": False,
                    "min": 0.0,
                    "max": 0.5,
                    "unit": None
                },
                "period": {
                    "value": 4.5,
                    "fixed": True,
                    "unit": "d"
                }
            },
            "primary": {
                "t_eff": {
                    "value": 4.5,
                    "fixed": True,
                    "unit": "d"
                },
                "surface_potential": {
                    "value": 3.5,
                    "fixed": False,
                    "unit": None,
                    "min": 2.0,
                    "max": 5.5
                }
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
                            "max": 5.5
                        }
                    },
                    {
                        "label": "unicorn",
                        "longitude": {
                            "value": 2.5,
                            "fixed": False,
                            "unit": "d",
                            "min": 2.0,
                            "max": 5.5
                        }
                    }
                ],
                "pulsations": [
                    {
                        "label": "unicorn",
                        "start_phase": {
                            "value": 2.2,
                            "fixed": False,
                            "unit": "d",
                            "min": 2.1,
                            "max": 5.0
                        }
                    }
                ]
            }
        }
        serialized = parameters.serialize_result(result_dict)
        self.assertDictEqual(serialized, expected)

    def test_deflate_phenomena(self):
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
            }
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
                    "max": 5.5
                }
            },
            "unicorn": {
                "label": "unicorn",
                "longitude": {
                    "value": 2.5,
                    "fixed": False,
                    "unit": "d",
                    "min": 2.0,
                    "max": 5.5
                }
            }
        }
        self.assertDictEqual(deflated, expected)

    def test_deserialize_result(self):
        serialized = expected = {
            "system": {
                "eccentricity": {
                    "value": 0.1,
                    "confidence_interval": {
                        "min": 0.12,
                        "max": 0.08
                    },
                    "fixed": False,
                    "min": 0.0,
                    "max": 0.5,
                    "unit": None
                },
                "period": {
                    "value": 4.5,
                    "fixed": True,
                    "unit": "d"
                }
            },
            "primary": {
                "t_eff": {
                    "value": 4.5,
                    "fixed": True,
                    "unit": "d"
                },
                "surface_potential": {
                    "value": 3.5,
                    "fixed": False,
                    "unit": None,
                    "min": 2.0,
                    "max": 5.5
                }
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
                            "max": 5.5
                        }
                    },
                    {
                        "label": "unicorn",
                        "longitude": {
                            "value": 2.5,
                            "fixed": False,
                            "unit": "d",
                            "min": 2.0,
                            "max": 5.5
                        }
                    }
                ],
                "pulsations": [
                    {
                        "label": "unicorn",
                        "start_phase": {
                            "value": 2.2,
                            "fixed": False,
                            "unit": "d",
                            "min": 2.1,
                            "max": 5.0
                        }
                    }
                ]
            }
        }
        expected = {
            "primary@t_eff": {
                "value": 4.5,
                "fixed": True,
                "unit": "d"
            },
            "primary@surface_potential": {
                "value": 3.5,
                "fixed": False,
                "unit": None,
                "min": 2.0,
                "max": 5.5
            },
            "secondary@spot@utopic@longitude": {
                "value": 4.5,
                "fixed": False,
                "unit": "d",
                "min": 2.0,
                "max": 5.5
            },
            "secondary@spot@unicorn@longitude": {
                "value": 2.5,
                "fixed": False,
                "unit": "d",
                "min": 2.0,
                "max": 5.5
            },
            "secondary@pulsation@unicorn@start_phase": {
                "value": 2.2,
                "fixed": False,
                "unit": "d",
                "min": 2.1,
                "max": 5.0
            },
            "system@eccentricity": {
                "value": 0.1,
                "confidence_interval": {
                    "min": 0.12,
                    "max": 0.08
                },
                "fixed": False,
                "min": 0.0,
                "max": 0.5,
                "unit": None
            },
            "system@period": {
                "value": 4.5,
                "fixed": True,
                "unit": "d"
            }
        }
        deserialized = parameters.deserialize_result(serialized)
        self.assertDictEqual(deserialized, expected)

    def test_constraints(self):
        substitution = {
            "system@inclination": 90,
            "primary@surface_potential": 4.5,
        }
        constrained = {
            "secondary@surface_potential": "2.0 * primary@surface_potential"
        }
        evaluated = parameters.constraints_evaluator(substitution, constrained)
        self.assertTrue(evaluated['secondary@surface_potential'] == 9.0)

    def test_prepare_properties_set(self):
        constrained = {
            "secondary@surface_potential": "2.0 * primary@surface_potential"
        }
        fixed = {
            "system@inclination": 90.0
        }
        properties = ["primary@surface_potential", "primary@t_eff", "system@argument_of_Periastron"]
        xn = [1.5, 2.0, 3.0]
        expected = {

            "primary@surface_potential": 1.5,
            "primary@t_eff": 2.0,
            "system@argument_of_Periastron": 3.0,
            "secondary@surface_potential": 3.0,
            "system@inclination": 90.0

        }
        properties_set = parameters.prepare_properties_set(xn, properties, constrained, fixed)
        self.assertDictEqual(properties_set, expected)


class BinaryInitialParametersTestCase(ElisaTestCase):
    lc_initial = {
        "system": {
            "semi_major_axis": {
                "value": 16.515,
                "constraint": "16.515 / sin(radians(system@inclination))"
            },
            "inclination": {
                "value": 85.0,
                "fixed": False,
                "min": 80,
                "max": 90
            },
            "argument_of_periastron": {
                "value": 0.0,
                "fixed": True
            },
            "mass_ratio": {
                "value": 0.5,
                "fixed": True
            },
            "eccentricity": {
                "value": 0.0,
                "fixed": True
            },
            "period": {
                "value": 4.5,
                "fixed": True
            }
        },
        "primary": {
            "t_eff": {
                "value": 8307.0,
                "fixed": False,
                "min": 7800.0,
                "max": 8800.0
            },
            "surface_potential": {
                "value": 3.0,
                "fixed": False,
                "min": 3,
                "max": 5
            },
            "gravity_darkening": {
                "value": 0.32,
                "fixed": True
            },
            "albedo": {
                "value": 0.6,
                "fixed": True
            },
            "pulsations": [
                {
                    "label": "bionic",
                    "l": {
                        "value": 1.0,
                        "fixed": True
                    },
                    "m": {
                        "value": 0.0,
                        "fixed": False,
                    },
                    "amplitude": {
                        "value": 0.0,
                        "fixed": False,
                    },
                    "frequency": {
                        "value": 10,
                        "fixed": False,
                        "min": 1.0,
                        "max": 20.0
                    },
                    "start_phase": {
                        "constraint": "2.0 * primary@pulsation@bionic@frequency",
                    },
                    "mode_axis_theta": {
                        "value": 0.0,
                        "fixed": True
                    },
                    "mode_axis_phi": {
                        "value": 0.0,
                        "fixed": True
                    }
                }
            ]
        },
        "secondary": {
            "t_eff": {
                "value": 4000.0,
                "fixed": False,
                "min": 4000.0,
                "max": 7000.0
            },
            "surface_potential": {
                "value": 5.0,
                "fixed": False,
                "min": 5.0,
                "max": 7.0
            },
            "gravity_darkening": {
                "value": 0.32,
                "fixed": True
            },
            "albedo": {
                "value": 0.6,
                "fixed": True
            },
            "spots": [
                {
                    "label": "utopic",
                    "latitude": {
                        "value": 10,
                        "min": 0,
                        "max": 15,
                        "fixed": False
                    },
                    "longitude": {
                        "value": 20,
                        "min": 0,
                        "max": 30,
                        "fixed": False
                    },
                    "angular_radius": {
                        "value": 12,
                        "fixed": True
                    },
                    "temperature_factor": {
                        "value": 0.95,
                        "fixed": True
                    },
                }
            ]
        }
    }

    def setUp(self):
        super().setUp()
        self.initial_parametres = parameters.BinaryInitialParameters(**self.lc_initial)

    def test_initial_data(self):
        serialized_data_expected = {
            "primary@pulsation@bionic@l": {
                "value": 1,
                "param": "l",
                "min": None,
                "max": None,
                "unit": None,
                "fixed": True
            },
            "primary@pulsation@bionic@m": {
                "value": 0,
                "param": "m",
                "min": -10,
                "max": 10,
                "unit": None,
                "fixed": False
            },
            "primary@pulsation@bionic@amplitude": {
                "value": 0.0,
                "param": "amplitude",
                "min": 0.0,
                "max": 5000.0,
                "unit": "m / s",
                "fixed": False
            },
            "primary@pulsation@bionic@frequency": {
                "value": 10.0,
                "param": "frequency",
                "min": 1.0,
                "max": 20.0,
                "unit": "Hz",
                "fixed": False
            },
            "primary@pulsation@bionic@start_phase": {
                "value": None,
                "param": "start_phase",
                "min": None,
                "max": None,
                "unit": "deg",
                "constraint": "2.0 * primary@pulsation@bionic@frequency"
            },
            "primary@pulsation@bionic@mode_axis_theta": {
                "value": 0.0,
                "param": "mode_axis_theta",
                "min": None,
                "max": None,
                "unit": "deg",
                "fixed": True
            },
            "primary@pulsation@bionic@mode_axis_phi": {
                "value": 0.0,
                "param": "mode_axis_phi",
                "min": None,
                "max": None,
                "unit": "deg",
                "fixed": True
            },
            "primary@t_eff": {
                "value": 8307.0,
                "param": "t_eff",
                "min": 7800.0,
                "max": 8800.0,
                "unit": "K",
                "fixed": False
            },
            "primary@surface_potential": {
                "value": 3.0,
                "param": "surface_potential",
                "min": 3.0,
                "max": 5.0,
                "unit": None,
                "fixed": False
            },
            "primary@albedo": {
                "value": 0.6,
                "param": "albedo",
                "min": None,
                "max": None,
                "unit": None,
                "fixed": True
            },
            "primary@gravity_darkening": {
                "value": 0.32,
                "param": "gravity_darkening",
                "min": None,
                "max": None,
                "unit": None,
                "fixed": True
            },
            "secondary@spot@utopic@longitude": {
                "value": 20.0,
                "param": "longitude",
                "min": 0.0,
                "max": 30.0,
                "unit": "deg",
                "fixed": False
            },
            "secondary@spot@utopic@latitude": {
                "value": 10.0,
                "param": "latitude",
                "min": 0.0,
                "max": 15.0,
                "unit": "deg",
                "fixed": False
            },
            "secondary@spot@utopic@angular_radius": {
                "value": 12.0,
                "param": "angular_radius",
                "min": None,
                "max": None,
                "unit": "deg",
                "fixed": True
            },
            "secondary@spot@utopic@temperature_factor": {
                "value": 0.95,
                "param": "temperature_factor",
                "min": None,
                "max": None,
                "unit": None,
                "fixed": True
            },
            "secondary@t_eff": {
                "value": 4000.0,
                "param": "t_eff",
                "min": 4000.0,
                "max": 7000.0,
                "unit": "K",
                "fixed": False
            },
            "secondary@surface_potential": {
                "value": 5.0,
                "param": "surface_potential",
                "min": 5.0,
                "max": 7.0,
                "unit": None,
                "fixed": False
            },
            "secondary@albedo": {
                "value": 0.6,
                "param": "albedo",
                "min": None,
                "max": None,
                "unit": None,
                "fixed": True
            },
            "secondary@gravity_darkening": {
                "value": 0.32,
                "param": "gravity_darkening",
                "min": None,
                "max": None,
                "unit": None,
                "fixed": True
            },
            "system@eccentricity": {
                "value": 0.0,
                "param": "eccentricity",
                "min": None,
                "max": None,
                "unit": None,
                "fixed": True
            },
            "system@argument_of_periastron": {
                "value": 0.0,
                "param": "argument_of_periastron",
                "min": None,
                "max": None,
                "unit": "deg",
                "fixed": True
            },
            "system@inclination": {
                "value": 85.0,
                "param": "inclination",
                "min": 80.0,
                "max": 90.0,
                "unit": "deg",
                "fixed": False
            },
            "system@period": {
                "value": 4.5,
                "param": "period",
                "min": None,
                "max": None,
                "unit": "d",
                "fixed": True
            },
            "system@mass_ratio": {
                "value": 0.5,
                "param": "mass_ratio",
                "min": None,
                "max": None,
                "unit": None,
                "fixed": True
            },
            "system@semi_major_axis": {
                "value": None,
                "param": "semi_major_axis",
                "min": None,
                "max": None,
                "unit": "solRad",
                "constraint": "16.515 / sin(radians(system@inclination))"
            }
        }

        serialized_in_dicts = {key: val.to_dict() for key, val in self.initial_parametres.data.items()}
        self.assertTrue(len(serialized_data_expected) == len(serialized_data_expected))
        self.assertDictEqual(serialized_data_expected, serialized_in_dicts)

    def test_get_fixed(self):
        fixed = self.initial_parametres.get_fixed()
        expected = ['primary@albedo', 'primary@gravity_darkening', 'primary@pulsation@bionic@l',
                    'primary@pulsation@bionic@mode_axis_phi', 'primary@pulsation@bionic@mode_axis_theta',
                    'secondary@albedo', 'secondary@gravity_darkening', 'secondary@spot@utopic@angular_radius',
                    'secondary@spot@utopic@temperature_factor', 'system@argument_of_periastron', 'system@eccentricity',
                    'system@mass_ratio', 'system@period']
        assert_array_equal(expected, sorted(list(fixed.keys())))

    def test_get_constrained(self):
        constrained = self.initial_parametres.get_constrained()
        expected = ['primary@pulsation@bionic@start_phase', 'system@semi_major_axis']
        assert_array_equal(expected, sorted(list(constrained.keys())))

    def test_get_fitable(self):
        fitable = self.initial_parametres.get_fitable()
        expected = ['primary@pulsation@bionic@amplitude', 'primary@pulsation@bionic@frequency',
                    'primary@pulsation@bionic@m', 'primary@surface_potential', 'primary@t_eff',
                    'secondary@spot@utopic@latitude', 'secondary@spot@utopic@longitude', 'secondary@surface_potential',
                    'secondary@t_eff', 'system@inclination']
        assert_array_equal(expected, sorted(list(fitable.keys())))

    def test_get_substitution_dict(self):
        substitution_dict = self.initial_parametres.get_substitution_dict()
        expected = {
            "primary@pulsation@bionic@l": 1,
            "primary@pulsation@bionic@m": 0,
            "primary@pulsation@bionic@amplitude": 0.0,
            "primary@pulsation@bionic@frequency": 10.0,
            "primary@pulsation@bionic@mode_axis_theta": 0.0,
            "primary@pulsation@bionic@mode_axis_phi": 0.0,
            "primary@t_eff": 8307.0,
            "primary@surface_potential": 3.0,
            "primary@albedo": 0.6,
            "primary@gravity_darkening": 0.32,
            "secondary@spot@utopic@longitude": 20.0,
            "secondary@spot@utopic@latitude": 10.0,
            "secondary@spot@utopic@angular_radius": 12.0,
            "secondary@spot@utopic@temperature_factor": 0.95,
            "secondary@t_eff": 4000.0,
            "secondary@surface_potential": 5.0,
            "secondary@albedo": 0.6,
            "secondary@gravity_darkening": 0.32,
            "system@eccentricity": 0.0,
            "system@argument_of_periastron": 0.0,
            "system@inclination": 85.0,
            "system@period": 4.5,
            "system@mass_ratio": 0.5
        }
        self.assertDictEqual(expected, substitution_dict)

    def test_get_normalization_map(self):
        normalisation = self.initial_parametres.get_normalization_map()
        assert_array_equal((80, 90), normalisation['system@inclination'])
        assert_array_equal((7800, 8800), normalisation['primary@t_eff'])
        assert_array_equal((3, 5), normalisation['primary@surface_potential'])
        assert_array_equal((1, 20), normalisation['primary@pulsation@bionic@frequency'])
        assert_array_equal((4000, 7000), normalisation['secondary@t_eff'])
        assert_array_equal((5, 7), normalisation['secondary@surface_potential'])
        assert_array_equal((0, 15), normalisation['secondary@spot@utopic@latitude'])
        assert_array_equal((0, 30), normalisation['secondary@spot@utopic@longitude'])

    def test_is_overcontact(self):
        self.assertTrue(self.initial_parametres.is_overcontact('over-contact'))
        self.assertFalse(self.initial_parametres.is_overcontact('overcontact'))
        self.assertFalse(self.initial_parametres.is_overcontact('detached'))

    def test_unit_conversion_to_default(self):
        rv_initial = {
            "system": {
                "eccentricity": {
                    "value": 0.2,
                    "fixed": False,
                    "min": 0.0,
                    "max": 0.5,
                    "unit": u.dimensionless_unscaled
                },
                "asini": {
                    "value": 2785360000,
                    "fixed": False,
                    "min": 2785360000 - 10000,
                    "max": 2785360000 + 10000,
                    "unit": u.m
                },
                "mass_ratio": {
                    "value": 3,
                    "fixed": False,
                    "min": 0.1,
                    "max": 10
                },
                "argument_of_periastron": {
                    "value": 0.0,
                    "fixed": True,
                    "unit": u.ARC_UNIT
                },
                "gamma": {
                    "value": 300.0,
                    "fixed": False,
                    "min": 100.0,
                    "max": 500.0,
                    "unit": u.km / u.s
                },
                "period": {
                    "value": 4.5,
                    "fixed": True,
                    "unit": u.PERIOD_UNIT
                }
            }
        }
        initial_parametres = parameters.BinaryInitialParameters(**rv_initial)

        self.assertTrue(initial_parametres.data["system@gamma"].value == 300000.0)
        self.assertTrue(initial_parametres.data["system@period"].value == 4.5)
        self.assertTrue(round(initial_parametres.data["system@asini"].value) == 4.0)
        self.assertTrue(round(initial_parametres.data["system@argument_of_periastron"].value) == 0.0)

        self.assertTrue(initial_parametres.data["system@gamma"].unit == u.m / u.s)
        self.assertTrue(initial_parametres.data["system@period"].unit == u.d)
        self.assertTrue(initial_parametres.data["system@asini"].unit == u.solRad)
        self.assertTrue(initial_parametres.data["system@argument_of_periastron"].unit == u.deg)

    @parameterized.expand([['eccentricity'], ['argument_of_periastron'], ['gamma']])
    def test_validate_mandatory_rv_parameters(self, param):
        mandatory_fit_params = ['eccentricity', 'argument_of_periastron', 'gamma']
        mock_value = {"value": 0.25, "min": 0.00, "max": 0.99}
        mock_data = {"system": {p: mock_value for p in mandatory_fit_params if p not in [param]}}

        with self.assertRaises(Exception) as context:
            parameters.BinaryInitialParameters(**mock_data).validate_rv_parameters()
        self.assertTrue('Missing argument' in str(context.exception))

    def test_validate_require_period(self):
        mandatory_fit_params = ['eccentricity', 'argument_of_periastron', 'gamma']
        mock_value = {"value": 0.25, "min": 0.00, "max": 0.99}
        mock_data = {"system": {p: mock_value for p in mandatory_fit_params}}

        with self.assertRaises(Exception) as context:
            parameters.BinaryInitialParameters(**mock_data).validate_rv_parameters()
        self.assertTrue('Input requires at least period' in str(context.exception))

    @parameterized.expand(['system@eccentricity', 'system@argument_of_periastron',
                           'system@period', 'system@inclination', 'system@period'] + \
                          [f'{component}@{param}'
                           for param in ['t_eff', 'surface_potential', 'gravity_darkening', 'albedo']
                           for component in settings.BINARY_COUNTERPARTS])
    def test_validate_mandatory_lc_parameters(self, param):
        mandatory_fit_params = ['system@eccentricity', 'system@argument_of_periastron',
                                'system@period', 'system@inclination', 'system@period'] + \
                               [f'{component}@{param}'
                                for param in ['t_eff', 'surface_potential', 'gravity_darkening', 'albedo']
                                for component in settings.BINARY_COUNTERPARTS]

        mock_value = {"value": 0.25, "min": 0.00, "max": 0.99}
        mock_data = {p: mock_value for p in mandatory_fit_params if p not in [param]}
        mock_data = parameters.serialize_result(mock_data)

        with self.assertRaises(Exception) as context:
            parameters.BinaryInitialParameters(**mock_data).validate_lc_parameters('detached')
        self.assertTrue('Missing argument' in str(context.exception))

    @parameterized.expand(
        [[{'primary@surface_potential': {"fixed": True, "value": 0.25, "min": 0.2, "max": 1.0},
          'secondary@surface_potential': {"fixed": True, "value": 0.26, "min": 0.2, "max": 1.0}},
         'Different potential'],
         [{'primary@surface_potential': {"fixed": True, "value": 0.25, "min": 0.2, "max": 1.0},
           'secondary@surface_potential': {"fixed": False, "value": 0.25, "min": 0.2, "max": 1.0}},
          'Just one fixed potential']
         ]
    )
    def test_validate_overcontact_lc_parameters(self, potentials, err):
        mandatory_fit_params = ['system@eccentricity', 'system@argument_of_periastron',
                                'system@period', 'system@inclination', 'system@period'] + \
                               [f'{component}@{param}'
                                for param in ['t_eff', 'gravity_darkening', 'albedo']
                                for component in settings.BINARY_COUNTERPARTS]

        mock_value = {"value": 0.25, "min": 0.00, "max": 0.99}
        mock_data = {p: mock_value for p in mandatory_fit_params}
        mock_data.update(potentials)
        mock_data = parameters.serialize_result(mock_data)

        with self.assertRaises(Exception) as context:
            parameters.BinaryInitialParameters(**mock_data).validate_lc_parameters('over-contact')
        self.assertTrue(err in str(context.exception))
