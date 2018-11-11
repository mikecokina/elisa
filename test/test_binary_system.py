import unittest
import numpy as np
from engine.binary_system import BinarySystem
from engine.star import Star
import engine.const as c
from astropy import units as u
from numpy.testing import assert_array_almost_equal
from engine import utils


# class TestBinarySystemProperties(unittest.TestCase):
#
#     def test_orbit_change(self):
#         bs = BinarySystem(primary=None, secondary=None)
#
#         bs.period = 10.0
#         bs.eccentricity = 0.6
#         bs.inclination = 0.25
#         bs.periastron = 0.2345
#
#         bs.init()
#
#         self.assertEqual(bs.orbit.eccentricity, 0.6)
#         self.assertEqual(bs.orbit.inclination, 0.25)
#         self.assertEqual(bs.orbit.periastron, 0.2345)
#         self.assertEqual(bs.orbit.period, 10)
#
#     def test_binary_system_unit(self):
#         bs = BinarySystem(primary=None, secondary=None)
#
#         bs.period = 432000 * u.s
#         self.assertEqual(bs.period, 5)
#         with self.assertRaises(TypeError):
#             bs.period = '0000'
#
#         bs.inclination = 90 * u.deg
#         self.assertEqual(bs.inclination, 0.5 * np.pi)
#         with self.assertRaises(TypeError):
#             bs.inclination = '0000'
#
#         bs.periastron = 45 * u.deg
#         self.assertEqual(bs.periastron, np.pi * 0.25)
#         with self.assertRaises(TypeError):
#             bs.inclination = '0000'
#
#
#         # add all
#
#     def test_binary_change(self):
#         bs = BinarySystem(primary=None, secondary=None)
#
#         bs.period = 12.0
#         self.assertEqual(bs.period, 12)
#
#         bs.gamma = 2.556
#         self.assertEqual(bs.gamma, 2.556)

class TestBinarySystem(unittest.TestCase):

    def setUp(self):
        self.params_combination = [
            {"primary_mass": 2.0, "secondary_mass": 1.0,
             "primary_surface_potential": 100.0, "secondary_surface_potential": 100.0,
             "primary_synchronicity": 1.0, "secondary_synchronicity": 1.0,
             "argument_of_periastron": c.HALF_PI * u.rad, "gamma": 0.0, "period": 1.0,
             "eccentricity": 0.0, "inclination": c.HALF_PI * u.deg, "primary_minimum_time": 0.0,
             "phase_shift": 0.0,
             "primary_t_eff": 5000, "secondary_t_eff": 5000,
             "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0},
            # compact spherical components on circular orbit

            {"primary_mass": 2.0, "secondary_mass": 1.0,
             "primary_surface_potential": 100.0, "secondary_surface_potential": 80.0,
             "primary_synchronicity": 400, "secondary_synchronicity": 550,
             "argument_of_periastron": c.HALF_PI*u.rad, "gamma": 0.0, "period": 1.0,
             "eccentricity": 0.0, "inclination": 90.0 * u.deg, "primary_minimum_time": 0.0,
             "phase_shift": 0.0,
             "primary_t_eff": 5000, "secondary_t_eff": 5000,
             "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0
             },  # rotationally squashed compact spherical components

            {"primary_mass": 2.0, "secondary_mass": 1.0,
             "primary_surface_potential": 3.5, "secondary_surface_potential": 3.0,
             "primary_synchronicity": 1.5, "secondary_synchronicity": 1.2,
             "argument_of_periastron": c.HALF_PI, "gamma": 0.0, "period": 1.0,
             "eccentricity": 0.0, "inclination": 90.0 * u.deg, "primary_minimum_time": 0.0,
             "phase_shift": 0.0,
             "primary_t_eff": 5000, "secondary_t_eff": 5000,
             "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0
             },  # close tidally deformed components with asynchronous rotation
                                   # on circular orbit

            {"primary_mass": 2.0, "secondary_mass": 1.0,
             "primary_surface_potential": 4.8, "secondary_surface_potential": 4.0,
             "primary_synchronicity": 1.5, "secondary_synchronicity": 1.2,
             "argument_of_periastron": c.HALF_PI*u.rad, "gamma": 0.0, "period": 1.0,
             "eccentricity": 0.3, "inclination": 90.0 * u.deg, "primary_minimum_time": 0.0,
             "phase_shift": 0.0,
             "primary_t_eff": 5000, "secondary_t_eff": 5000,
             "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0
             },  # close tidally deformed components with asynchronous rotation
                                   # on eccentric orbit

            {"primary_mass": 2.0, "secondary_mass": 1.0,
             "primary_surface_potential": 2.875844632141054,
             "secondary_surface_potential": 2.875844632141054,
             "primary_synchronicity": 1.0, "secondary_synchronicity": 1.0,
             "argument_of_periastron": c.HALF_PI*u.rad, "gamma": 0.0, "period": 1.0,
             "eccentricity": 0.0, "inclination": 90.0 * u.deg, "primary_minimum_time": 0.0,
             "phase_shift": 0.0,
             "primary_t_eff": 5000, "secondary_t_eff": 5000,
             "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0
             },  # synchronous contact system

            {"primary_mass": 2.0, "secondary_mass": 1.0,
             "primary_surface_potential": 3.159639848886489,
             "secondary_surface_potential": 3.229240544834036,
             "primary_synchronicity": 1.5, "secondary_synchronicity": 2.0,
             "argument_of_periastron": c.HALF_PI*u.rad, "gamma": 0.0, "period": 1.0,
             "eccentricity": 0.0, "inclination": 90.0 * u.deg, "primary_minimum_time": 0.0,
             "phase_shift": 0.0,
             "primary_t_eff": 5000, "secondary_t_eff": 5000,
             "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0
             },  # asynchronous contact system (improbable but whatever...)

            {"primary_mass": 2.0, "secondary_mass": 1.0,
             "primary_surface_potential": 2.7,
             "secondary_surface_potential": 2.7,
             "primary_synchronicity": 1.0, "secondary_synchronicity": 1.0,
             "argument_of_periastron": c.HALF_PI*u.rad, "gamma": 0.0, "period": 1.0,
             "eccentricity": 0.0, "inclination": 90.0 * u.deg, "primary_minimum_time": 0.0,
             "phase_shift": 0.0,
             "primary_t_eff": 5000, "secondary_t_eff": 5000,
             "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0
             }  # contact system
        ]

    def test_critical_potential(self):
        phases_to_use = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        expected_potentials = [
            [2.875844632141054, 2.875844632141054],
            [93.717106763853593, 73.862399105365014],
            [3.159639848886489, 2.935086409515319],
            [4.027577786299736, 3.898140726941630],
            [2.875844632141054, 2.875844632141054],
            [3.159639848886489, 3.229240544834036],
            [2.875844632141054, 2.875844632141054]
        ]
        obtained_potentials = []

        for i, combo in enumerate(self.params_combination):
            primary = Star(mass=combo["primary_mass"], surface_potential=combo["primary_surface_potential"],
                           synchronicity=combo["primary_synchronicity"],
                           t_eff=combo["primary_t_eff"], gravity_darkening=combo["primary_gravity_darkening"])

            secondary = Star(mass=combo["secondary_mass"], surface_potential=combo["secondary_surface_potential"],
                             synchronicity=combo["secondary_synchronicity"],
                             t_eff=combo["secondary_t_eff"], gravity_darkening=combo["secondary_gravity_darkening"])

            bs = BinarySystem(primary=primary,
                              secondary=secondary,
                              argument_of_periastron=combo["argument_of_periastron"],
                              gamma=combo["gamma"],
                              period=combo["period"],
                              eccentricity=combo["eccentricity"],
                              inclination=combo["inclination"],
                              primary_minimum_time=combo["primary_minimum_time"],
                              phase_shift=combo["phase_shift"])

            components_distance = bs.orbit.orbital_motion(phase=phases_to_use[i])[0][0]
            primary_cp = bs.critical_potential(component="primary", components_distance=components_distance)
            secondary_cp = bs.critical_potential(component="secondary", components_distance=components_distance)
            obtained_potentials.append([round(primary_cp, 10), round(secondary_cp, 10)])
        assert_array_almost_equal(expected_potentials, obtained_potentials)

    def test_lagrangian_points(self):

        expected_points = [
            [-0.80302796065835425, 0.57075157151852673, 1.5823807222136623],
            [-0.80302796065835425, 0.57075157151852673, 1.5823807222136623],
            [-0.80302796065835425, 0.57075157151852673, 1.5823807222136623],
            [-0.7308068505479407, 0.41566688133312363, 1.4990376377419574],
            [-0.80302796065835425, 0.57075157151852673, 1.5823807222136623],
            [-0.80302796065835425, 0.57075157151852673, 1.5823807222136623],
            [-0.80302796065835425, 0.57075157151852673, 1.5823807222136623]
        ]

        obtained_points = []

        for i, combo in enumerate(self.params_combination):
            primary = Star(mass=combo["primary_mass"], surface_potential=combo["primary_surface_potential"],
                           synchronicity=combo["primary_synchronicity"],
                           t_eff=combo["primary_t_eff"], gravity_darkening=combo["primary_gravity_darkening"])

            secondary = Star(mass=combo["secondary_mass"], surface_potential=combo["secondary_surface_potential"],
                             synchronicity=combo["secondary_synchronicity"],
                             t_eff=combo["secondary_t_eff"], gravity_darkening=combo["secondary_gravity_darkening"])

            bs = BinarySystem(primary=primary,
                              secondary=secondary,
                              argument_of_periastron=combo["argument_of_periastron"],
                              gamma=combo["gamma"],
                              period=combo["period"],
                              eccentricity=combo["eccentricity"],
                              inclination=combo["inclination"],
                              primary_minimum_time=combo["primary_minimum_time"],
                              phase_shift=combo["phase_shift"])

            obtained_points.append(bs.lagrangian_points())

        self.assertAlmostEquals(expected_points, obtained_points)

    def test_mesh_for_duplicate_points(self):
        phases_to_use = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        alpha = 10

        for i, combo in enumerate(self.params_combination):
            primary = Star(mass=combo["primary_mass"], surface_potential=combo["primary_surface_potential"],
                           synchronicity=combo["primary_synchronicity"], discretization_factor=alpha,
                           t_eff=combo["primary_t_eff"], gravity_darkening=combo["primary_gravity_darkening"])

            secondary = Star(mass=combo["secondary_mass"], surface_potential=combo["secondary_surface_potential"],
                             synchronicity=combo["secondary_synchronicity"], discretization_factor=alpha,
                             t_eff=combo["secondary_t_eff"], gravity_darkening=combo["secondary_gravity_darkening"])

            bs = BinarySystem(primary=primary,
                              secondary=secondary,
                              argument_of_periastron=combo["argument_of_periastron"],
                              gamma=combo["gamma"],
                              period=combo["period"],
                              eccentricity=combo["eccentricity"],
                              inclination=combo["inclination"],
                              primary_minimum_time=combo["primary_minimum_time"],
                              phase_shift=combo["phase_shift"])

            if bs.morphology == "over-contact":
                mesh_primary = bs.mesh_over_contact(component='primary')
                mesh_secondary = bs.mesh_over_contact(component='secondary')
            else:
                components_distance = bs.orbit.orbital_motion(phase=phases_to_use[i])[0][0]
                mesh_primary = bs.mesh_detached(component='primary', components_distance=components_distance)
                mesh_secondary = bs.mesh_detached(component='secondary', components_distance=components_distance)

            distance1 = round(utils.find_nearest_dist_3d(list(mesh_primary)), 10)
            distance2 = round(utils.find_nearest_dist_3d(list(mesh_secondary)), 10)
            print(distance1, distance2)
            self.assertFalse(distance1 < 1e-10)
            self.assertFalse(distance2 < 1e-10)

    def test_morphology(self):
        # todo: doplnit test pre rozne kriticke pripady v morfologii
        pass

    def test_spots(self):
        # todo: doplnit testy pre rozne patologicke pripady (aj nepatologicke)
        pass
