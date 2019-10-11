import os.path as op
import numpy as np

from unittest import skip
from copy import copy
from astropy import units as u
from mpl_toolkits.mplot3d import Axes3D
from numpy.testing import assert_array_equal

from elisa import const as c
from elisa.base.star import Star
from elisa.single_system.system import SingleSystem
from elisa.conf import config
from elisa.utils import is_empty, find_nearest_dist_3d
from unittests.utils import ElisaTestCase
from unittests.utils import plot_points, plot_faces, polar_gravity_acceleration, prepare_binary_system

ax3 = Axes3D


class TestSingleSystemSetters(ElisaTestCase):
    MANDATORY_KWARGS = ['gamma', 'inclination', 'rotation_period']

    def setUp(self):
        combo = {
            "mass": 1.0, 't_eff': 5774, 'gravity_darkening': 1.0, 'polar_log_g': 4.1,
            "gamma": 0.0, 'inclination': 90*u.deg, 'rotation_period': 28
        }

        star = Star(mass=combo['mass'], t_eff=combo['t_eff'],
                    gravity_darkening=combo['gravity_darkening'],
                    polar_log_g=combo['polar_log_g'])

        self._single = SingleSystem(star=star,
                                    gamma=combo["gamma"],
                                    inclination=combo["inclination"],
                                    rotation_period=combo['rotation_period'])

    def _test_generic_setter(self, input_vals, expected, val_str):
        obtained = list()
        for val in input_vals:
            setattr(self._single, val_str, val)
            obtained.append(round(getattr(self._single, val_str), 3))
        assert_array_equal(obtained, expected)

    def test_period(self):
        periods = [0.25 * u.d, 0.65, 86400 * u.s]
        expected = [0.25, 0.65, 1.0]
        self._test_generic_setter(periods, expected, 'rotation_period')

    def test_inclination(self):
        inclinations = [135 * u.deg, 90.0, 1.56 * u.rad]
        expected = [2.356, 1.571, 1.56]
        self._test_generic_setter(inclinations, expected, 'inclination')