import unittest
import numpy as np
from engine.binary_system import BinarySystem
from engine.star import Star
import engine.const as c
from astropy import units as u

class TestBinarySystemProperties(unittest.TestCase):

    def setUp(self):
        pass

    def test_periastron_distance(self):
        pass

    def test_critical_potential(self):
        params_combination = [{"primary_mass": 2.0, "secondary_mass": 1.0,
                               "primary_surface_potential": 100.0, "secondary_surface_potential": 100.0,
                               "primary_synchronicity": 1.0, "secondary_synchronicity": 1.0,
                               "argument_of_periastron": np.pi / 2.0, "gamma": 0.0, "period": 0.9,
                               "eccentricity": 0.0, "inclination": c.HALF_PI, "primary_minimum_time": 0.0,
                               "phase_shift": 0.0},

                              {"primary_mass": 2.0, "secondary_mass": 1.0,
                               "primary_surface_potential": 100.0, "secondary_surface_potential": 100.0,
                               "primary_synchronicity": 1.7, "secondary_synchronicity": 1.3,
                               "argument_of_periastron": c.PI, "gamma": 0.0, "period": 1.0,
                               "eccentricity": 0.1, "inclination": 85.0 * u.deg, "primary_minimum_time": 0.0,
                               "phase_shift": 0.0}]

        phases_to_use = [0.25, 0.0]
        expected_potentials = [[2.8758446321, 2.8758446321],
                               [3.3060281748, 2.9883984921]]
        obtained_potentials = []

        for i, combo in enumerate(params_combination):
            primary = Star(mass=combo["primary_mass"], surface_potential=combo["primary_surface_potential"],
                           synchronicity=combo["primary_synchronicity"])

            secondary = Star(mass=combo["secondary_mass"], surface_potential=combo["secondary_surface_potential"],
                             synchronicity=combo["secondary_synchronicity"])

            bs = BinarySystem(primary=primary,
                              secondary=secondary,
                              argument_of_periastron=combo["argument_of_periastron"],
                              gamma=combo["gamma"],
                              period=combo["period"],
                              eccentricity=combo["eccentricity"],
                              inclination=combo["inclination"],
                              primary_minimum_time=combo["primary_minimum_time"],
                              phase_shift=combo["phase_shift"])

            primary_cp = bs.critical_potential(component="primary", phase=phases_to_use[i])
            secondary_cp = bs.critical_potential(component="secondary", phase=phases_to_use[i])

            obtained_potentials.append([round(primary_cp, 10), round(secondary_cp, 10)])

        self.assertEquals(expected_potentials, obtained_potentials)



