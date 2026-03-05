# keep it first
# due to stupid astropy units/constants implementation
from elisa.opt.newton import newton
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


class NewtonTestCase(utils.ElisaTestCase):
    def test_newton_solves_scalar_root(self):
        def func(x, _packed_args):
            return x ** 2 - 4.0

        def fprime(x):
            return 2.0 * x

        solution = newton(func, 3.0, fprime, args=((),), rtol=1e-10)

        self.assertAlmostEqual(solution, 2.0, places=8)

    def test_newton_solves_with_args(self):
        def func(x, packed_args):
            target, = packed_args
            return x - target

        def fprime(x, target):
            return 1.0

        solution = newton(func, 0.5, fprime, args=((3.0,),), rtol=1e-12)

        self.assertAlmostEqual(solution, 3.0, places=8)

    def test_newton_solves_array_root(self):
        def func(x, _packed_args):
            return x ** 2 - 4.0

        def fprime(x):
            return 2.0 * x

        x0 = np.array([3.0, 4.0])
        solution = newton(func, x0, fprime, args=((),), rtol=1e-10)

        np.testing.assert_allclose(solution, np.array([2.0, 2.0]), rtol=1e-8, atol=1e-8)
