"""Verify that temporary synthesis error policies never leak to callers."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_allclose

from elisa import utils
from elisa.base.error import ElisaFloatingPointError
from elisa.binary_system import system as binary_system

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray

ErrorPolicy = dict[str, str]


@pytest.fixture(
    params=[
        pytest.param(
            {"divide": "warn", "invalid": "ignore", "over": "raise", "under": "ignore"},
            id="warn-divide-ignore-invalid",
        ),
        pytest.param(
            {"divide": "ignore", "invalid": "raise", "over": "warn", "under": "raise"},
            id="ignore-divide-raise-invalid",
        ),
    ],
)
def caller_policy(request: pytest.FixtureRequest) -> Iterator[ErrorPolicy]:
    """Exercise distinct caller policies and restore the test runner's settings."""
    with np.errstate(**request.param):
        # noinspection PyTypeChecker
        yield np.geterr()


def test_cartesian_to_spherical_preserves_policy_on_success(caller_policy: ErrorPolicy) -> None:
    """Origin and polar-axis coordinates retain their intentional NaN repairs."""
    points = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    expected = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, np.pi / 2, np.pi / 2]]

    result = utils.cartesian_to_spherical(np.array(points))

    assert np.geterr() == caller_policy
    assert_allclose(result, expected)


def test_cartesian_to_spherical_preserves_policy_on_invalid_shape(caller_policy: ErrorPolicy) -> None:
    """Missing z coordinates fail inside the temporary policy without leaking it."""
    with pytest.raises(IndexError, match="out of bounds"):
        utils.cartesian_to_spherical(np.array([[1.0, 2.0]]))

    assert np.geterr() == caller_policy


def test_cartesian_to_spherical_preserves_policy_on_arithmetic_failure(caller_policy: ErrorPolicy) -> None:
    """An exception from an angular operation propagates with policy restored."""
    message = "Angular calculation failed"
    failure = FloatingPointError(message)
    with (
        patch.object(utils.up, "arcsin", side_effect=failure),
        pytest.raises(FloatingPointError, match=message) as caught,
    ):
        utils.cartesian_to_spherical(np.array([[1.0, 2.0, 3.0]]))

    assert caught.value is failure
    assert np.geterr() == caller_policy


def test_lagrangian_points_preserves_policy_on_success(caller_policy: ErrorPolicy) -> None:
    """Real root solving preserves both caller settings and reference positions."""
    expected = [-0.80302796065835425, 0.57075157151852673, 1.5823807222136623]

    result = binary_system.BinarySystem.lagrangian_points_static(1.0, 0.5)

    assert np.geterr() == caller_policy
    assert_allclose(result, expected, rtol=0.0, atol=1e-10)


def test_lagrangian_points_restores_policy_before_logging_and_continue(caller_policy: ErrorPolicy) -> None:
    """Singular candidates restore policy before logging and subsequent solves."""
    logged_policies: list[ErrorPolicy] = []

    def record_log(_message: str, *_args: object) -> None:
        # noinspection PyTypeChecker
        logged_policies.append(np.geterr())

    # x=0 and x=distance fail the real potential evaluation. Only the two
    # nonsingular candidates should reach the solver, under the caller's policy.
    candidates = np.array([0.0, -1.0, 1.0, 2.0])
    expected_solver_inputs = [-1.0, 2.0]
    expected_policies = [caller_policy, caller_policy]
    solver_policies: list[ErrorPolicy] = []

    def record_solver_policy(
        *_args: object,
        **_kwargs: object,
    ) -> tuple[NDArray[np.float64], dict[str, object], int, str]:
        # noinspection PyTypeChecker
        solver_policies.append(np.geterr())
        # A failed convergence status lets the candidate loop run to completion.
        return np.array([0.0]), {}, 0, "No convergence"

    with (
        patch.object(binary_system.np, "linspace", return_value=candidates),
        patch.object(binary_system, "fsolve", side_effect=record_solver_policy) as solver,
        patch.object(binary_system.logger, "debug", side_effect=record_log) as logger,
    ):
        with pytest.raises(ElisaFloatingPointError, match="Lagrange solver"):
            binary_system.BinarySystem.lagrangian_points_static(1.0, 0.5)
    assert np.geterr() == caller_policy
    assert [call.args[1] for call in solver.call_args_list] == expected_solver_inputs
    assert solver_policies == expected_policies
    assert logged_policies == expected_policies
    assert all(call.args[0] == "invalid value passed to potential, exception: %s" for call in logger.call_args_list)
