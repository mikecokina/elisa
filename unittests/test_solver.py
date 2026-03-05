# keep it first
# due to stupid astropy units/constants implementation
from unittests import set_astropy_units

import numpy as np

from elisa.opt.fsolver import fsolve, fsolver
from unittests import utils

set_astropy_units()


class FSolveTestCase(utils.ElisaTestCase):
    def test_fsolve_returns_solution_only(self):
        def fn(x):
            return x ** 2 - 4.0

        solution = fsolve(fn, np.array([1.5]))
        self.assertAlmostEqual(solution[0], 2.0, places=8)

    def test_fsolve_returns_full_output(self):
        def fn(x):
            return x - 3.0

        solution, info, ier, msg = fsolve(fn, np.array([0.5]), full_output=True)

        self.assertAlmostEqual(solution[0], 3.0, places=8)
        self.assertEqual(ier, 1)
        self.assertIsInstance(info, dict)
        self.assertTrue(isinstance(msg, str))


class FSolverTestCase(utils.ElisaTestCase):
    def test_fsolver_returns_solution_when_condition_passes(self):
        def fn(x, target):
            return x - target

        def condition(solution_, target):
            return abs(solution_ - target) < 1e-8

        solution, use = fsolver(fn, condition, (2.5,))

        self.assertAlmostEqual(solution, 2.5, places=8)
        self.assertTrue(use)

    def test_fsolver_returns_nan_when_condition_fails(self):
        def fn(x, target):
            return x - target

        # noinspection PyUnusedLocal
        def condition(solution_, target):
            return solution_ < 0.0

        solution, use = fsolver(fn, condition, (2.5,))

        self.assertTrue(np.isnan(solution))
        self.assertFalse(use)
