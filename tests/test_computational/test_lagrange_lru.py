"""Lagrange caching preserves independent results and numerical failure behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest

from elisa.base.error import ElisaMaxIterationError, ElisaNumericError
from elisa.binary_system import system

if TYPE_CHECKING:
    import functools
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def isolated_cache() -> Iterator[functools._lru_cache_wrapper]:
    cache = system.BinarySystem._lagrangian_points_static
    cache.cache_clear()
    yield cache
    cache.cache_clear()


@pytest.mark.usefixtures("isolated_cache")
def test_repeated_points_reuse_solver_and_return_independent_lists() -> None:
    with patch.object(system, "fsolve", wraps=system.fsolve) as solver:
        first = system.BinarySystem.lagrangian_points_static(1.0, 0.7)
        attempts = solver.call_count
        second = system.BinarySystem.lagrangian_points_static(1.0, 0.7)
        assert attempts > 0
        assert solver.call_count == attempts
    np.testing.assert_array_equal(first, second)
    assert isinstance(first, list)
    assert first is not second
    first[0] = 123.0
    second.clear()
    assert system.BinarySystem.lagrangian_points_static(1.0, 0.7)[0] != 123.0
    assert len(system.BinarySystem.lagrangian_points_static(1.0, 0.7)) == 3


@pytest.mark.parametrize(
    ("distance", "ratio"),
    [(1.0, 0.8), (0.9, 0.7), (1.0, float(np.nextafter(0.7, 1.0))), (1.0, 1.4)],
)
def test_distinct_exact_inputs_have_independent_results(
    isolated_cache: functools._lru_cache_wrapper,
    distance: float,
    ratio: float,
) -> None:
    system.BinarySystem.lagrangian_points_static(1.0, 0.7)
    with patch.object(system, "fsolve", wraps=system.fsolve) as solver:
        result = system.BinarySystem.lagrangian_points_static(distance, ratio)
        assert solver.call_count > 0
    expected = isolated_cache.__wrapped__(distance, ratio)
    np.testing.assert_array_equal(result, expected)
    assert isolated_cache.cache_info().currsize == 2


def test_exhausted_solver_does_not_poison_cache(isolated_cache: functools._lru_cache_wrapper) -> None:
    with patch.object(system, "fsolve", return_value=(np.array([np.nan]), {}, 4, "no root")) as solver:
        for _ in range(2):
            with pytest.raises(ElisaMaxIterationError, match="Lagrange solver"):
                system.BinarySystem.lagrangian_points_static(1.0, 0.7)
        assert solver.call_count > 100
    assert isolated_cache.cache_info().currsize == 0
    assert len(system.BinarySystem.lagrangian_points_static(1.0, 0.7)) == 3
    assert isolated_cache.cache_info().currsize == 1


def test_failure_injection_after_a_cached_calculation(isolated_cache: functools._lru_cache_wrapper) -> None:
    system.BinarySystem.lagrangian_points_static(1.0, 0.7)
    isolated_cache.cache_clear()
    failure = RuntimeError("failed guess")
    with patch.object(system, "fsolve", side_effect=failure) as solver:
        with pytest.raises(RuntimeError) as caught:
            system.BinarySystem.lagrangian_points_static(1.0, 0.7)
        assert caught.value is failure
        assert solver.call_count == 1
    assert isolated_cache.cache_info().currsize == 0


def test_changed_error_policy_executes_solver_with_policy_restored(
    isolated_cache: functools._lru_cache_wrapper,
) -> None:
    system.BinarySystem.lagrangian_points_static(1.0, 0.7)
    with np.errstate(all="raise"):
        policy = np.geterr()
        with patch.object(system, "fsolve", side_effect=FloatingPointError("failed guess")) as solver:
            with pytest.raises(ElisaMaxIterationError, match="Lagrange solver"):
                system.BinarySystem.lagrangian_points_static(1.0, 0.7)
            assert solver.call_count > 1
        assert np.geterr() == policy
    assert isolated_cache.cache_info().currsize == 1


@pytest.mark.parametrize(("distance", "ratio"), [(0.0, 0.7), (-1.0, 0.7), (1.0, -0.7), (np.inf, 0.7), (1.0, np.nan)])
@pytest.mark.parametrize("policy", ["ignore", "raise"])
def test_invalid_inputs_retain_uncached_behavior(
    isolated_cache: functools._lru_cache_wrapper,
    distance: float,
    ratio: float,
    policy: str,
) -> None:
    with np.errstate(all=policy):
        try:
            expected = isolated_cache.__wrapped__(distance, ratio)
        except (FloatingPointError, ElisaMaxIterationError, ValueError) as exc:
            with pytest.raises(type(exc), match=str(exc)):
                system.BinarySystem.lagrangian_points_static(distance, ratio)
        else:
            np.testing.assert_array_equal(system.BinarySystem.lagrangian_points_static(distance, ratio), expected)
    assert isolated_cache.cache_info().currsize == 0


def test_scalar_precision_is_preserved_in_cache_keys(isolated_cache: functools._lru_cache_wrapper) -> None:
    for scalar_type in (float, np.float32, np.float64):
        distance, ratio = scalar_type(1.0), scalar_type(0.5)
        expected = isolated_cache.__wrapped__(distance, ratio)
        result = system.BinarySystem.lagrangian_points_static(distance, ratio)
        np.testing.assert_array_equal(result, expected)
    assert isolated_cache.cache_info().currsize == 3


@pytest.mark.parametrize(("distance", "ratio"), [(10**100, 0.7), (1.0, 10**100)])
@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
def test_large_integer_inputs_retain_solver_failure(
    isolated_cache: functools._lru_cache_wrapper,
    distance: float,
    ratio: float,
) -> None:
    # Native float arithmetic avoids object dtype, but converged attempts cannot
    # resolve the three distinct validated roots for these extreme candidates.
    with pytest.raises(ElisaNumericError, match="distinct validated roots"):
        system.BinarySystem.lagrangian_points_static(distance, ratio)
    assert isolated_cache.cache_info().currsize == 0


def test_unrepresentable_integer_is_numeric_failure(isolated_cache: functools._lru_cache_wrapper) -> None:
    with pytest.raises(ElisaNumericError, match="floating-point range") as caught:
        system.BinarySystem.lagrangian_points_static(10**400, 0.7)
    assert isinstance(caught.value.__cause__, OverflowError)
    assert isolated_cache.cache_info().currsize == 0


@pytest.mark.usefixtures("isolated_cache")
def test_integer_scalars_use_native_solver_dtype() -> None:
    expected = system.BinarySystem.lagrangian_points_static(1.0, 1.0)
    np.testing.assert_array_equal(system.BinarySystem.lagrangian_points_static(1, 1), expected)
