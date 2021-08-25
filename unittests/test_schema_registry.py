# keep it first
# due to stupid astropy units/constants implementation
from unittests import set_astropy_units

from elisa.binary_system.utils import validate_binary_json
from elisa.single_system.utils import validate_single_json
from unittests import utils

set_astropy_units()


class StdBinarySystemmSchemaRegistryTestCase(utils.ElisaTestCase):
    def test_valid_schema(self):
        props = {
            "system": {
                "eccentricity": 0.0,
                "argument_of_periastron": 90,
                "gamma": 0.0,
                "period": 100.0,
                "inclination": 90
            },
            "primary": {
                "mass": 1.0,
                "t_eff": 5774.0,
                "surface_potential": 104.3,
                "synchronicity": 1.0,
                "gravity_darkening": 0.32,
                "discretization_factor": 3,
                "albedo": 0.6,
                "metallicity": 0.0
            },
            "secondary": {
                "mass": 0.5,
                "t_eff": 4000.0,
                "surface_potential": 130.0,
                "synchronicity": 1.0,
                "gravity_darkening": 0.32,
                "discretization_factor": 5,
                "albedo": 0.6,
                "metallicity": 0.0
            }
        }
        self.assertTrue(validate_binary_json(props))

    def test_valid_schema_with_spots_in_secondary(self):
        props = {
            "system": {
                "eccentricity": 0.0,
                "argument_of_periastron": 90,
                "gamma": 0.0,
                "period": 100.0,
                "inclination": 90
            },
            "primary": {
                "mass": 1.0,
                "t_eff": 5774.0,
                "surface_potential": 104.3,
                "synchronicity": 1.0,
                "gravity_darkening": 0.32,
                "discretization_factor": 3,
                "albedo": 0.6,
                "metallicity": 0.0
            },
            "secondary": {
                "mass": 0.5,
                "t_eff": 4000.0,
                "surface_potential": 130.0,
                "synchronicity": 1.0,
                "gravity_darkening": 0.32,
                "discretization_factor": 5,
                "albedo": 0.6,
                "metallicity": 0.0,
                "spots": [
                    {
                        "longitude": 10,
                        "latitude": 5,
                        "angular_radius": 10,
                        "temperature_factor": 0.9,
                        "discretization_factor": 5
                    },
                    {
                        "longitude": 10,
                        "latitude": 5,
                        "angular_radius": 10,
                        "temperature_factor": 0.9
                    }
                ]
            }
        }
        self.assertTrue(validate_binary_json(props))

    def test_valid_schema_with_spots_in_primary(self):
        props = {
            "system": {
                "eccentricity": 0.0,
                "argument_of_periastron": 90,
                "gamma": 0.0,
                "period": 100.0,
                "inclination": 90
            },
            "primary": {
                "mass": 1.0,
                "t_eff": 5774.0,
                "surface_potential": 104.3,
                "synchronicity": 1.0,
                "gravity_darkening": 0.32,
                "discretization_factor": 3,
                "albedo": 0.6,
                "metallicity": 0.0,
                "spots": [
                    {
                        "longitude": 10,
                        "latitude": 5,
                        "angular_radius": 10,
                        "temperature_factor": 0.9,
                        "discretization_factor": 5
                    }
                ]
            },
            "secondary": {
                "mass": 0.5,
                "t_eff": 4000.0,
                "surface_potential": 130.0,
                "synchronicity": 1.0,
                "gravity_darkening": 0.32,
                "discretization_factor": 5,
                "albedo": 0.6,
                "metallicity": 0.0
            }
        }
        self.assertTrue(validate_binary_json(props))

    def test_valid_schema_all(self):
        props = {
            "system": {
                "eccentricity": 0.0,
                "argument_of_periastron": 90,
                "gamma": 0.0,
                "period": 100.0,
                "inclination": 90
            },
            "primary": {
                "mass": 1.0,
                "t_eff": 5774.0,
                "surface_potential": 104.3,
                "synchronicity": 1.0,
                "gravity_darkening": 0.32,
                "discretization_factor": 3,
                "albedo": 0.6,
                "metallicity": 0.0,
                "spots": [
                    {
                        "longitude": 10,
                        "latitude": 5,
                        "angular_radius": 10,
                        "temperature_factor": 0.9,
                        "discretization_factor": 5
                    }
                ],
                "pulsations": [
                    {
                        'l': 1,
                        'm': 1,
                        'amplitude': 1,
                        'frequency': 1,
                        'start_phase': 1,
                        'mode_axis_phi': 1,
                        'mode_axis_theta': 1
                    },
                    {
                        'l': 1,
                        'm': 1,
                        'amplitude': 1,
                        'frequency': 1,
                        'start_phase': 1,
                        'mode_axis_phi': 1,
                        'mode_axis_theta': 1
                    }
                ]
            },
            "secondary": {
                "mass": 0.5,
                "t_eff": 4000.0,
                "surface_potential": 130.0,
                "synchronicity": 1.0,
                "gravity_darkening": 0.32,
                "discretization_factor": 5,
                "albedo": 0.6,
                "metallicity": 0.0,
                "spots": [
                    {
                        "longitude": 10,
                        "latitude": 5,
                        "angular_radius": 10,
                        "temperature_factor": 0.9,
                        "discretization_factor": 5
                    },
                    {
                        "longitude": 10,
                        "latitude": 5,
                        "angular_radius": 10,
                        "temperature_factor": 0.9
                    }
                ],
                "pulsations": [
                    {
                        'l': 1,
                        'm': 1,
                        'amplitude': 1,
                        'frequency': 1,
                        'start_phase': 1,
                        'mode_axis_phi': 1,
                        'mode_axis_theta': 1
                    }
                ]
            }
        }
        self.assertTrue(validate_binary_json(props))


class CommunityBinarySystemmSchemaRegistryTestCase(utils.ElisaTestCase):
    def test_valid_schema(self):
        props = {
            "system": {
                "eccentricity": 0.0,
                "argument_of_periastron": 90,
                "gamma": 0.0,
                "period": 100.0,
                "inclination": 90,
                "mass_ratio": 0.5,
                "semi_major_axis": 10
            },
            "primary": {
                "t_eff": 5774.0,
                "surface_potential": 104.3,
                "synchronicity": 1.0,
                "gravity_darkening": 0.32,
                "discretization_factor": 3,
                "albedo": 0.6,
                "metallicity": 0.0
            },
            "secondary": {
                "t_eff": 4000.0,
                "surface_potential": 130.0,
                "synchronicity": 1.0,
                "gravity_darkening": 0.32,
                "discretization_factor": 5,
                "albedo": 0.6,
                "metallicity": 0.0
            }
        }
        self.assertTrue(validate_binary_json(props))

    def test_valid_schema_all(self):
        props = {
            "system": {
                "eccentricity": 0.0,
                "argument_of_periastron": 90,
                "gamma": 0.0,
                "period": 100.0,
                "inclination": 90,
                "mass_ratio": 0.5,
                "semi_major_axis": 10
            },
            "primary": {
                "t_eff": 5774.0,
                "surface_potential": 104.3,
                "synchronicity": 1.0,
                "gravity_darkening": 0.32,
                "discretization_factor": 3,
                "albedo": 0.6,
                "metallicity": 0.0,
                "spots": [
                    {
                        "longitude": 10,
                        "latitude": 5,
                        "angular_radius": 10,
                        "temperature_factor": 0.9,
                        "discretization_factor": 5
                    }
                ],
                "pulsations": [
                    {
                        'l': 1,
                        'm': 1,
                        'amplitude': 1,
                        'frequency': 1,
                        'start_phase': 1,
                        'mode_axis_phi': 1,
                        'mode_axis_theta': 1
                    },
                    {
                        'l': 1,
                        'm': 1,
                        'amplitude': 1,
                        'frequency': 1,
                        'start_phase': 1,
                        'mode_axis_phi': 1,
                        'mode_axis_theta': 1
                    }
                ]
            },
            "secondary": {
                "t_eff": 4000.0,
                "surface_potential": 130.0,
                "synchronicity": 1.0,
                "gravity_darkening": 0.32,
                "discretization_factor": 5,
                "albedo": 0.6,
                "metallicity": 0.0,
                "spots": [
                    {
                        "longitude": 10,
                        "latitude": 5,
                        "angular_radius": 10,
                        "temperature_factor": 0.9,
                        "discretization_factor": 5
                    },
                    {
                        "longitude": 10,
                        "latitude": 5,
                        "angular_radius": 10,
                        "temperature_factor": 0.9
                    }
                ],
                "pulsations": [
                    {
                        'l': 1,
                        'm': 1,
                        'amplitude': 1,
                        'frequency': 1,
                        'start_phase': 1,
                        'mode_axis_phi': 1,
                        'mode_axis_theta': 1
                    }
                ]
            }
        }
        self.assertTrue(validate_binary_json(props))


class StdSingleSystemmSchemaRegistryTestCase(utils.ElisaTestCase):
    def test_valid_schema(self):
        props = {
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
        self.assertTrue(validate_single_json(props))

    def test_valid_schema2(self):
        props = {
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
                "equivalent_radius": 6.96340e8
            }
        }
        self.assertTrue(validate_single_json(props))

    def test_invalid_schema3(self):
        props = {
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
                "equivalent_radius": 6.96340e8,
                "polar_log_g": 2.43775
            }
        }
        with self.assertRaises(Exception) as context:
            self.assertTrue(validate_single_json(props))
        self.assertTrue('`polar_log_g` or `polar_radius`' in str(context.exception))

    def test_valid_schema_with_spots_in_secondary(self):
        props = {
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
                "polar_log_g": 2.43775,
                "spots": [
                    {
                        "longitude": 10,
                        "latitude": 5,
                        "angular_radius": 10,
                        "temperature_factor": 0.9,
                        "discretization_factor": 5
                    },
                    {
                        "longitude": 10,
                        "latitude": 5,
                        "angular_radius": 10,
                        "temperature_factor": 0.9
                    }
                ]
            }
        }
        self.assertTrue(validate_single_json(props))

    def test_valid_schema_all(self):
        props = {
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
                "equivalent_radius": 6.96340e8,
                "spots": [
                    {
                        "longitude": 10,
                        "latitude": 5,
                        "angular_radius": 10,
                        "temperature_factor": 0.9,
                        "discretization_factor": 5
                    },
                    {
                        "longitude": 10,
                        "latitude": 5,
                        "angular_radius": 10,
                        "temperature_factor": 0.9
                    }
                ],
                "pulsations": [
                    {
                        'l': 1,
                        'm': 1,
                        'amplitude': 1,
                        'frequency': 1,
                        'start_phase': 1,
                        'mode_axis_phi': 1,
                        'mode_axis_theta': 1
                    },
                    {
                        'l': 1,
                        'm': 1,
                        'amplitude': 1,
                        'frequency': 1,
                        'start_phase': 1,
                        'mode_axis_phi': 1,
                        'mode_axis_theta': 1
                    }
                ]
            }
        }
        self.assertTrue(validate_single_json(props))
