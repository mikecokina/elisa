import unittest
import numpy as np
from engine.binary_system import BinarySystem
from engine.star import Star
import engine.const as c
from astropy import units as u
from numpy.testing import assert_array_almost_equal
from engine import utils
from unit_test import test_utils


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
             "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
             "primary_albedo": 0.6, "secondary_albedo": 0.6,
             },
            # compact spherical components on circular orbit

            {"primary_mass": 2.0, "secondary_mass": 1.0,
             "primary_surface_potential": 100.0, "secondary_surface_potential": 80.0,
             "primary_synchronicity": 400, "secondary_synchronicity": 550,
             "argument_of_periastron": c.HALF_PI*u.rad, "gamma": 0.0, "period": 1.0,
             "eccentricity": 0.0, "inclination": 90.0 * u.deg, "primary_minimum_time": 0.0,
             "phase_shift": 0.0,
             "primary_t_eff": 5000, "secondary_t_eff": 5000,
             "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
             "primary_albedo": 0.6, "secondary_albedo": 0.6
             },  # rotationally squashed compact spherical components

            {"primary_mass": 2.0, "secondary_mass": 1.0,
             "primary_surface_potential": 3.5, "secondary_surface_potential": 3.0,
             "primary_synchronicity": 1.5, "secondary_synchronicity": 1.2,
             "argument_of_periastron": c.HALF_PI, "gamma": 0.0, "period": 1.0,
             "eccentricity": 0.0, "inclination": 90.0 * u.deg, "primary_minimum_time": 0.0,
             "phase_shift": 0.0,
             "primary_t_eff": 5000, "secondary_t_eff": 5000,
             "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
             "primary_albedo": 0.6, "secondary_albedo": 0.6
             },  # close tidally deformed components with asynchronous rotation
                                   # on circular orbit

            {"primary_mass": 2.0, "secondary_mass": 1.0,
             "primary_surface_potential": 4.8, "secondary_surface_potential": 4.0,
             "primary_synchronicity": 1.5, "secondary_synchronicity": 1.2,
             "argument_of_periastron": c.HALF_PI*u.rad, "gamma": 0.0, "period": 1.0,
             "eccentricity": 0.3, "inclination": 90.0 * u.deg, "primary_minimum_time": 0.0,
             "phase_shift": 0.0,
             "primary_t_eff": 5000, "secondary_t_eff": 5000,
             "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
             "primary_albedo": 0.6, "secondary_albedo": 0.6
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
             "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
             "primary_albedo": 0.6, "secondary_albedo": 0.6
             },  # synchronous contact system

            {"primary_mass": 2.0, "secondary_mass": 1.0,
             "primary_surface_potential": 3.159639848886489,
             "secondary_surface_potential": 3.229240544834036,
             "primary_synchronicity": 1.5, "secondary_synchronicity": 2.0,
             "argument_of_periastron": c.HALF_PI*u.rad, "gamma": 0.0, "period": 1.0,
             "eccentricity": 0.0, "inclination": 90.0 * u.deg, "primary_minimum_time": 0.0,
             "phase_shift": 0.0,
             "primary_t_eff": 5000, "secondary_t_eff": 5000,
             "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
             "primary_albedo": 0.6, "secondary_albedo": 0.6
             },  # asynchronous contact system (improbable but whatever...)

            {"primary_mass": 2.0, "secondary_mass": 1.0,
             "primary_surface_potential": 2.7,
             "secondary_surface_potential": 2.7,
             "primary_synchronicity": 1.0, "secondary_synchronicity": 1.0,
             "argument_of_periastron": c.HALF_PI*u.rad, "gamma": 0.0, "period": 1.0,
             "eccentricity": 0.0, "inclination": 90.0 * u.deg, "primary_minimum_time": 0.0,
             "phase_shift": 0.0,
             "primary_t_eff": 5000, "secondary_t_eff": 5000,
             "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
             "primary_albedo": 0.6, "secondary_albedo": 0.6
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
                           t_eff=combo["primary_t_eff"], gravity_darkening=combo["primary_gravity_darkening"],
                           albedo=combo['primary_albedo'])

            secondary = Star(mass=combo["secondary_mass"], surface_potential=combo["secondary_surface_potential"],
                             synchronicity=combo["secondary_synchronicity"],
                             t_eff=combo["secondary_t_eff"], gravity_darkening=combo["secondary_gravity_darkening"],
                             albedo=combo['secondary_albedo'])

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
                           t_eff=combo["primary_t_eff"], gravity_darkening=combo["primary_gravity_darkening"],
                           albedo=combo['primary_albedo'])

            secondary = Star(mass=combo["secondary_mass"], surface_potential=combo["secondary_surface_potential"],
                             synchronicity=combo["secondary_synchronicity"],
                             t_eff=combo["secondary_t_eff"], gravity_darkening=combo["secondary_gravity_darkening"],
                             albedo=combo['secondary_albedo'])

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
                           t_eff=combo["primary_t_eff"], gravity_darkening=combo["primary_gravity_darkening"],
                           albedo=combo['primary_albedo'])

            secondary = Star(mass=combo["secondary_mass"], surface_potential=combo["secondary_surface_potential"],
                             synchronicity=combo["secondary_synchronicity"], discretization_factor=alpha,
                             t_eff=combo["secondary_t_eff"], gravity_darkening=combo["secondary_gravity_darkening"],
                             albedo=combo['secondary_albedo'])

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

            bs.build_mesh(components_distance=components_distance)

            distance1 = round(utils.find_nearest_dist_3d(list(bs.primary.points)), 10)
            distance2 = round(utils.find_nearest_dist_3d(list(bs.secondary.points)), 10)
            # print(distance1, distance2)
            if distance1 < 1e-10:
                bad_points = []
                for ii, point in enumerate(bs.primary.points):
                    for jj, xx in enumerate(bs.primary.points[ii+1:]):
                        dist = np.linalg.norm(point-xx)
                        if dist <= 1e-10:
                            print('Match: {0}, {1}, {2}'.format(point, ii, jj))
                            bad_points.append(point)

            self.assertFalse(distance1 < 1e-10)
            self.assertFalse(distance2 < 1e-10)

    def test_morphology(self):
        # todo: doplnit test pre rozne kriticke pripady v morfologii
        pass

    def test_spots(self):
        phases_to_use = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        alpha = 3
        spots_metadata = {
            "primary":
                [
                    {"longitude": 90,
                     "latitude": 58,
                     # "angular_density": 1,
                     "angular_diameter": 17,
                     "temperature_factor": 0.9},
                    {"longitude": 90,
                     "latitude": 57,
                     # "angular_density": 2,
                     "angular_diameter": 30,
                     "temperature_factor": 1.05},
                    {"longitude": 45,
                     "latitude": 90,
                     # "angular_density": 2,
                     "angular_diameter": 30,
                     "temperature_factor": 0.95},
                ],

            "secondary":
                [
                    {"longitude": 10,
                     "latitude": 45,
                     # "angular_density": 3,
                     "angular_diameter": 28,
                     "temperature_factor": 0.7},
                    {"longitude": 30,
                     "latitude": 65,
                     # "angular_density": 3,
                     "angular_diameter": 45,
                     "temperature_factor": 0.75},
                    {"longitude": 45,
                     "latitude": 40,
                     # "angular_density": 3,
                     "angular_diameter": 40,
                     "temperature_factor": 0.80},
                    {"longitude": 50,
                     "latitude": 55,
                     # "angular_density": 3,
                     "angular_diameter": 28,
                     "temperature_factor": 0.85},
                    {"longitude": 25,
                     "latitude": 55,
                     # "angular_density": 3,
                     "angular_diameter": 15,
                     "temperature_factor": 0.9},
                    {"longitude": 0,
                     "latitude": 70,
                     # "angular_density": 3,
                     "angular_diameter": 45,
                     "temperature_factor": 0.95}
                ]
        }

        for i, combo in enumerate(self.params_combination):
            primary = Star(mass=combo["primary_mass"], surface_potential=combo["primary_surface_potential"],
                           synchronicity=combo["primary_synchronicity"], discretization_factor=alpha,
                           t_eff=combo["primary_t_eff"], gravity_darkening=combo["primary_gravity_darkening"],
                           spots=spots_metadata['primary'],
                           albedo=combo['primary_albedo'])

            secondary = Star(mass=combo["secondary_mass"], surface_potential=combo["secondary_surface_potential"],
                             synchronicity=combo["secondary_synchronicity"], discretization_factor=alpha,
                             t_eff=combo["secondary_t_eff"], gravity_darkening=combo["secondary_gravity_darkening"],
                             spots=spots_metadata['secondary'],
                             albedo=combo['secondary_albedo'])

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
            points, faces = bs.build_surface(components_distance=components_distance, return_surface=True)

            duplicity_check1 = test_utils.check_face_duplicity(faces=faces['primary'], points=points['primary'])
            duplicity_check2 = test_utils.check_face_duplicity(faces=faces['secondary'])
            self.assertTrue(duplicity_check1)
            self.assertTrue(duplicity_check2)

            # distance1 = round(utils.find_nearest_dist_3d(list(bs.primary.points)), 10)
            # distance2 = round(utils.find_nearest_dist_3d(list(bs.secondary.points)), 10)
            # # print(distance1, distance2)
            # if distance1 < 1e-10:
            #     bad_points = []
            #     for ii, point in enumerate(bs.primary.points):
            #         for jj, xx in enumerate(bs.primary.points[ii + 1:]):
            #             dist = np.linalg.norm(point - xx)
            #             if dist <= 1e-10:
            #                 print('Match: {0}, {1}, {2}'.format(point, ii, jj))
            #                 bad_points.append(point)
            #
            # self.assertFalse(distance1 < 1e-10)
            # self.assertFalse(distance2 < 1e-10)