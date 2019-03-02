import unittest
import numpy as np
from elisa.engine.single_system.system import SingleSystem
from elisa.engine.base.star import Star
from astropy import units as u
from elisa.engine import utils
from numpy import testing


class TestSingleSystem(unittest.TestCase):

    def setUp(self):
        self.params_combination = [{'mass': 1.0*u.solMass,
                                    't_eff': 5700*u.K,
                                    'gravity_darkening': 0.32,
                                    'discretization_factor': 3,
                                    'spots': None,
                                    'gamma': 0*u.km/u.s,
                                    'inclination': 90*u.deg,
                                    'rotation_period': 30*u.d,
                                    'polar_log_g': 4.1 * u.dex(u.cm / u.s ** 2)},  # sun-like basic star

                                   {'mass': 1.0 * u.solMass,
                                    't_eff': 5700 * u.K,
                                    'gravity_darkening': 0.32,
                                    'discretization_factor': 3,
                                    'spots': None,
                                    'gamma': 0 * u.km / u.s,
                                    'inclination': 90 * u.deg,
                                    'rotation_period': 0.382 * u.d,
                                    'polar_log_g': 4.1 * u.dex(u.cm / u.s ** 2)},  # sun-like very fast rotating star
                                   ]

    def test_for_duplicate_points(self):
        for i, combo in enumerate(self.params_combination):
            s = Star(mass=combo['mass'],
                     t_eff=combo['t_eff'],
                     gravity_darkening=combo['gravity_darkening'],
                     discretization_factor=combo['discretization_factor'],
                     spots=combo['spots']
                     )

            single = SingleSystem(star=s,
                                  gamma=combo['gamma'],
                                  inclination=combo['inclination'],
                                  rotation_period=combo['rotation_period'],
                                  polar_log_g=combo['polar_log_g'])

            single.build_mesh()
            distance1 = round(utils.find_nearest_dist_3d(list(s.points)), 10)
            self.assertFalse(distance1 < 1e-10)

    def test_faces_validity(self):
        """
        test whether surface contain weird too big triangles
        :return:
        """
        for i, combo in enumerate(self.params_combination):
            s = Star(mass=combo['mass'],
                     t_eff=combo['t_eff'],
                     gravity_darkening=combo['gravity_darkening'],
                     discretization_factor=combo['discretization_factor'],
                     spots=combo['spots']
                     )

            single = SingleSystem(star=s,
                                  gamma=combo['gamma'],
                                  inclination=combo['inclination'],
                                  rotation_period=combo['rotation_period'],
                                  polar_log_g=combo['polar_log_g'])

            points, faces = single.build_surface(return_surface=True)
            points_faces = points[faces]
            average_radius = np.mean(np.linalg.norm(points_faces, axis=2))
            limit_distance = 3*average_radius*s.discretization_factor
            distances = np.linalg.norm(points_faces - points_faces[:, [1, 2, 0], :], axis=2)
            testing.assert_array_less(distances, limit_distance)
