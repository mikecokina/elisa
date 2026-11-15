from __future__ import annotations

from typing import TYPE_CHECKING

# The tests package selects Astropy constants before test-module imports.
import numpy as np
import pytest

from elisa.base import error
from elisa.opt.fsolver import fsolve, fsolver
from elisa.opt.newton import newton
from tests import set_astropy_units, utils
from tests.utils import ElisaTestCase

if TYPE_CHECKING:
    from numpy.typing import NDArray


set_astropy_units()


class FSolveTestCase(utils.ElisaTestCase):
    def test_fsolve_returns_solution_only(self) -> None:
        def fn(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return x**2 - 4.0

        solution = fsolve(fn, np.array([1.5]))
        self.assertAlmostEqual(solution[0], 2.0, places=8)

    def test_fsolve_returns_full_output(self) -> None:
        def fn(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return x - 3.0

        solution, info, ier, msg = fsolve(fn, np.array([0.5]), full_output=True)

        self.assertAlmostEqual(solution[0], 3.0, places=8)
        assert ier == 1
        assert isinstance(info, dict)
        assert isinstance(msg, str)


class FSolverTestCase(utils.ElisaTestCase):
    def test_fsolver_returns_solution_when_condition_passes(self) -> None:
        def fn(x: NDArray[np.float64], target: tuple[float]) -> NDArray[np.float64]:
            return x - target

        def condition(solution_: np.float64, target: tuple[float]) -> NDArray[np.bool]:
            return abs(solution_ - target) < 1e-8

        solution, use = fsolver(fn, condition, (2.5,))

        self.assertAlmostEqual(solution, 2.5, places=8)
        assert use

    def test_fsolver_returns_nan_when_condition_fails(self) -> None:
        def fn(x: NDArray[np.float64], target: tuple[float]) -> NDArray[np.float64]:
            return x - target

        # noinspection PyUnusedLocal
        def condition(solution_: np.float64, _target: tuple[float]) -> np.bool:
            return solution_ < 0.0

        solution, use = fsolver(fn, condition, (2.5,))

        assert np.isnan(solution)
        assert not use


class NewtonTestCase(utils.ElisaTestCase):
    def test_newton_solves_scalar_root(self) -> None:
        def func(x: float, _packed_args: tuple[()]) -> float:
            return x**2 - 4.0

        def fprime(x: float) -> float:
            return 2.0 * x

        solution = newton(func, 3.0, fprime, args=((),), rtol=1e-10)

        self.assertAlmostEqual(solution, 2.0, places=8)

    def test_newton_solves_with_args(self) -> None:
        def func(x: float, packed_args: tuple[float]) -> float:
            (target,) = packed_args
            return x - target

        def fprime(_x: float, _target: float) -> float:
            return 1.0

        solution = newton(func, 0.5, fprime, args=((3.0,),), rtol=1e-12)

        self.assertAlmostEqual(solution, 3.0, places=8)

    def test_newton_solves_array_root(self) -> None:
        def func(x: NDArray[np.float64], _packed_args: tuple[()]) -> NDArray[np.float64]:
            return x**2 - 4.0

        def fprime(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return 2.0 * x

        x0 = np.array([3.0, 4.0])
        solution = newton(func, x0, fprime, args=((),), rtol=1e-10)

        np.testing.assert_allclose(solution, np.array([2.0, 2.0]), rtol=1e-8, atol=1e-8)


class NewtonSolverTestCase(ElisaTestCase):
    @staticmethod
    def x_square(
        x: float | np.float64 | NDArray[np.float64],
        *args,
    ) -> np.float64 | NDArray[np.float64]:
        (a,) = args[0]
        return a * np.power(x + 1, 3) - args[1]

    @staticmethod
    def d_x_square(
        x: float | np.float64 | NDArray[np.float64],
        *args,
    ) -> np.float64 | NDArray[np.float64]:
        (a,) = args
        return 3.0 * a * np.power(x + 1, 2)

    def test_solver(self) -> None:
        x0 = 1.001
        args = ((1.0,), 0.0)
        expected = -1.0
        obtained = round(newton(self.x_square, x0, fprime=self.d_x_square, args=args, maxiter=100, rtol=1e-10), 7)
        assert expected == obtained

    def test_return_matrix(self) -> None:
        x0 = np.array([1.001, 1.002])
        args = ((np.array([1.0, 3.0]),), 0.0)
        expected = np.array([-1.0, -1.0])
        obtained = np.round(newton(self.x_square, x0, fprime=self.d_x_square, args=args, maxiter=100, rtol=1e-10), 7)
        assert np.all(expected == obtained)


def test_point_solver_keeps_nonconverged_and_unusable_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    solver = importlib.import_module("elisa.opt.fsolver")
    for solution, status in [(np.nan, 1), (0.5, 4), (-1.0, 1)]:
        monkeypatch.setattr(
            solver,
            "fsolve",
            lambda *_a, solution=solution, status=status, **_k: (np.array([solution]), {}, status, "nonconverged"),
        )
        radius, use = fsolver(lambda x: x, lambda _x: True)
        assert not use
        if status != 1 or np.isnan(solution):
            assert np.isnan(radius)


def test_unrecoverable_newton_failure_and_unexpected_function_error() -> None:
    with pytest.raises(error.ElisaMaxIterationError):
        newton(lambda x, *_a: x * x + 1, 1.0, lambda x: 2 * x, args=((),), maxiter=1)

    def broken_function(*_args) -> None:
        raise TypeError("broken numerical implementation")

    with pytest.raises(TypeError, match="broken numerical implementation"):
        newton(broken_function, 1.0, lambda _x: 1.0, args=((),))


@pytest.mark.parametrize("failure_type", [ValueError, TypeError, FloatingPointError, MemoryError])
def test_implicit_solver_wraps_failures_with_original_cause(
    failure_type: type[FloatingPointError | MemoryError | TypeError | ValueError],
) -> None:
    cause = failure_type("failed solver callback")

    def broken(*_args) -> None:
        raise cause

    with pytest.raises(error.ElisaNumericSolverError) as caught:
        fsolver(broken, lambda *_args: True)
    assert caught.value.__cause__ is cause


def test_implicit_solver_preserves_keyboard_interrupt() -> None:
    cause = KeyboardInterrupt()

    def interrupted(*_args) -> None:
        raise cause

    with pytest.raises(KeyboardInterrupt) as caught:
        fsolver(interrupted, lambda *_args: True)
    assert caught.value is cause
