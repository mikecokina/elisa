import numpy as np

from elisa.opt.newton import newton
from unittests.utils import ElisaTestCase


class NewtonSolverTestCase(ElisaTestCase):

    @staticmethod
    def x_square(x, a):
        return a * np.power(x + 1, 3)

    @staticmethod
    def d_x_square(x, a):
        return 3.0 * a * np.power(x + 1, 2)

    def test_solver(self):
        x0 = 1.001
        args = (1.0, )
        expected = -1.0
        obtained = round(newton(self.x_square, x0, fprime=self.d_x_square, args=args, maxiter=100, rtol=1e-10), 7)
        self.assertEqual(expected, obtained)

    def test_return_matrix(self):
        x0 = np.array([1.001, 1.002])
        args = (np.array([1.0, 3.0]), )
        expected = np.array([-1., -1.])
        obtained = np.round(newton(self.x_square, x0, fprime=self.d_x_square, args=args, maxiter=100, rtol=1e-10), 7)
        self.assertTrue(np.all(expected == obtained))
