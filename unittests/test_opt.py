from unittests.utils import ElisaTestCase


class NewtonSolverTestCase(ElisaTestCase):

    @staticmethod
    def x_square(x, a):
        return a * x ** 3 + 1

    @staticmethod
    def d_x_square(x, a):
        return 3.0 * a * x ** 2

    def test_solver(self):
        pass

    def test_return_scalar(self):
        pass

    def test_return_matrix(self):
        pass

    def test_reaction_on_changed_parameters(self):
        pass
