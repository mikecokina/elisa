"""Numerical and failure contracts of the vectorized secant calculation."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from typing import NoReturn

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from numpy.typing import DTypeLike, NDArray

from elisa.base.error import ElisaFloatingPointError, ElisaMaxIterationError
from elisa.binary_system.orbit.orbit import Orbit
from elisa.types import Float
from tests.kepler_helpers import scalar_orbital_motion


@pytest.fixture
def orbit() -> Orbit:
    return Orbit(
        period=3.0,
        inclination=np.pi / 2,
        eccentricity=0.3,
        argument_of_periastron=0.7,
    )


@pytest.fixture(autouse=True)
def default_error_policy() -> Iterator[None]:
    """Start numerical cases with NumPy's standard warning policy."""
    with np.errstate(divide="warn", invalid="warn", over="warn", under="ignore"):
        yield


def solve_scalar_anomalies(orbit: Orbit, mean_anomalies: NDArray) -> NDArray:
    """Solve each anomaly in its original order using the scalar reference."""
    return np.array([orbit.mean_anomaly_to_eccentric_anomaly(mean_anomaly) for mean_anomaly in mean_anomalies])


def assert_same_numerical_failure(scalar_error: BaseException, batch_error: BaseException) -> None:
    """Preserve the scalar exception type, message and absence of chaining."""
    assert type(batch_error) is type(scalar_error)
    assert str(batch_error) == str(scalar_error)
    assert scalar_error.__cause__ is None
    assert batch_error.__cause__ is None
    assert scalar_error.__context__ is None
    assert batch_error.__context__ is None


def overflowing_residual(eccentric_anomaly: Float | NDArray, _mean_anomaly: Float | NDArray) -> Float | NDArray:
    """Drive finite secant iterates to infinity with a residual that has no root."""
    return np.abs(eccentric_anomaly) ** (-1e-8)


@pytest.mark.parametrize("eccentricity", [0.0, 0.3, 0.95, 0.999999])
def test_batch_freezes_independently_converged_roots_in_input_order(orbit: Orbit, eccentricity: float) -> None:
    orbit.eccentricity = eccentricity
    # Boundary and interior roots require different numbers of secant updates.
    mean_anomalies = np.array([np.pi, 0.0, 1e-10, 2 * np.pi, 0.1, np.pi, -4 * np.pi - 0.1])
    original_anomalies = mean_anomalies.copy()
    scalar_anomalies = solve_scalar_anomalies(orbit, mean_anomalies)

    batch_anomalies = orbit._mean_anomalies_to_eccentric_anomalies(mean_anomalies)

    # Keep the scalar stopping iterate instead of polishing already finished roots.
    assert_array_equal(batch_anomalies, scalar_anomalies)
    assert_array_equal(mean_anomalies, original_anomalies)
    assert batch_anomalies[0] == batch_anomalies[5]


@pytest.mark.parametrize("failure_kind", ["zero-slope", "exhausted"])
@pytest.mark.parametrize(
    ("mean_anomalies", "failing_anomalies"),
    [
        pytest.param([0.2, 0.7, 0.2], [0.7], id="one-failure-after-success"),
        pytest.param([0.2, 0.7, 0.5, 0.2], [0.7, 0.5], id="first-of-two-failures"),
        pytest.param([0.2, 0.5, 0.7, 0.2], [0.5, 0.7], id="reordered-failures"),
        pytest.param([0.7, 0.5], [0.7, 0.5], id="all-elements-fail"),
    ],
)
def test_failed_batch_replays_original_first_error(
    orbit: Orbit,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
    mean_anomalies: list[float],
    failing_anomalies: list[float],
) -> None:
    input_anomalies = np.array(mean_anomalies)
    original_residual = orbit.mean_anomaly_fn

    def partially_failing_residual(
        eccentric_anomaly: Float | NDArray,
        mean_anomaly: Float | NDArray,
    ) -> NDArray:
        if failure_kind == "zero-slope":
            failing_residual = np.ones_like(eccentric_anomaly)
        else:
            failing_residual = np.exp(-eccentric_anomaly)
        return np.where(
            np.isin(mean_anomaly, failing_anomalies),
            failing_residual,
            original_residual(eccentric_anomaly, mean_anomaly),
        )

    monkeypatch.setattr(orbit, "mean_anomaly_fn", partially_failing_residual)

    with warnings.catch_warnings(record=True) as scalar_warnings:
        warnings.simplefilter("always")
        with pytest.raises(ElisaMaxIterationError) as scalar_error:
            solve_scalar_anomalies(orbit, input_anomalies)
    with warnings.catch_warnings(record=True) as batch_warnings:
        warnings.simplefilter("always")
        with pytest.raises(ElisaMaxIterationError) as batch_error:
            orbit._mean_anomalies_to_eccentric_anomalies(input_anomalies)

    assert_same_numerical_failure(scalar_error.value, batch_error.value)
    assert f"mean_anomaly={failing_anomalies[0]}." in str(batch_error.value)
    scalar_messages = [str(warning.message) for warning in scalar_warnings]
    batch_messages = [str(warning.message) for warning in batch_warnings]
    assert batch_messages == scalar_messages


@pytest.mark.parametrize("strict_errors", [False, True], ids=["warn", "raise"])
def test_nonfinite_root_preserves_scalar_exception_and_warnings(
    orbit: Orbit,
    monkeypatch: pytest.MonkeyPatch,
    strict_errors: bool,
) -> None:
    # Under the standard policy SciPy reports a converged infinite root, which
    # ELISA rejects. A strict caller instead receives the arithmetic exception.
    monkeypatch.setattr(orbit, "mean_anomaly_fn", overflowing_residual)
    mean_anomalies = np.array([0.2, 0.7])
    error_policy = np.errstate(all="raise") if strict_errors else np.errstate()
    expected_error_type = FloatingPointError if strict_errors else ElisaFloatingPointError

    with error_policy:
        original_policy = np.geterr()
        with warnings.catch_warnings(record=True) as scalar_warnings:
            warnings.simplefilter("always")
            with pytest.raises(expected_error_type) as scalar_error:
                solve_scalar_anomalies(orbit, mean_anomalies)
        with warnings.catch_warnings(record=True) as batch_warnings:
            warnings.simplefilter("always")
            with pytest.raises(expected_error_type) as batch_error:
                orbit._mean_anomalies_to_eccentric_anomalies(mean_anomalies)

        assert_same_numerical_failure(scalar_error.value, batch_error.value)
        scalar_messages = [str(warning.message) for warning in scalar_warnings]
        batch_messages = [str(warning.message) for warning in batch_warnings]
        assert batch_messages == scalar_messages
        assert np.geterr() == original_policy


@pytest.mark.parametrize("error_type", [TypeError, RuntimeError, FloatingPointError])
def test_batch_programming_errors_keep_identity_and_cause(
    orbit: Orbit,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
) -> None:
    injected_error = error_type("broken residual")
    original_cause = ValueError("invalid operand")
    injected_error.__cause__ = original_cause

    def broken_residual(_eccentric_anomaly: Float | NDArray, _mean_anomaly: Float | NDArray) -> NoReturn:
        raise injected_error

    monkeypatch.setattr(orbit, "mean_anomaly_fn", broken_residual)
    original_policy = np.geterr()

    with pytest.raises(error_type) as raised_error:
        orbit._mean_anomalies_to_eccentric_anomalies(np.array([0.1, 0.5]))

    assert raised_error.value is injected_error
    assert raised_error.value.__cause__ is original_cause
    assert np.geterr() == original_policy


@pytest.mark.parametrize("strict_errors", [False, True], ids=["warn", "raise"])
def test_successful_batch_preserves_numpy_policy(orbit: Orbit, strict_errors: bool) -> None:
    mean_anomalies = np.array([0.1, 0.5, 0.9])
    error_policy = np.errstate(all="raise") if strict_errors else np.errstate()

    with error_policy:
        original_policy = np.geterr()
        scalar_anomalies = solve_scalar_anomalies(orbit, mean_anomalies)

        batch_anomalies = orbit._mean_anomalies_to_eccentric_anomalies(mean_anomalies)

        assert_array_equal(batch_anomalies, scalar_anomalies)
        assert np.geterr() == original_policy


@pytest.mark.parametrize("eccentricity", [1.0 - 1e-12, np.nextafter(1.0, 0.0)])
def test_near_parabolic_batch_preserves_real_scalar_nonconvergence(orbit: Orbit, eccentricity: float) -> None:
    orbit.eccentricity = eccentricity
    # The scalar solver converges at pi but exhausts its budget at zero.
    mean_anomalies = np.array([np.pi, 0.0, 0.1, np.pi])

    with pytest.raises(ElisaMaxIterationError) as scalar_error:
        solve_scalar_anomalies(orbit, mean_anomalies)
    with pytest.raises(ElisaMaxIterationError) as batch_error:
        orbit._mean_anomalies_to_eccentric_anomalies(mean_anomalies)

    assert_same_numerical_failure(scalar_error.value, batch_error.value)


def test_batch_warning_as_error_preserves_scalar_diagnostic(orbit: Orbit, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orbit, "mean_anomaly_fn", overflowing_residual)
    mean_anomalies = np.array([0.2, 0.7])
    original_policy = np.geterr()

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(RuntimeWarning) as scalar_error:
            solve_scalar_anomalies(orbit, mean_anomalies)
        with pytest.raises(RuntimeWarning) as batch_error:
            orbit._mean_anomalies_to_eccentric_anomalies(mean_anomalies)

    assert_same_numerical_failure(scalar_error.value, batch_error.value)
    assert np.geterr() == original_policy


@pytest.mark.parametrize("dtype", [object, np.longdouble])
def test_public_orbit_preserves_non_float64_phase_inputs(orbit: Orbit, dtype: DTypeLike) -> None:
    phases = np.array([0.3, -0.1, 0.3, 1.1], dtype=dtype)
    scalar_positions = scalar_orbital_motion(orbit, phases)

    positions = orbit.orbital_motion(phases)

    assert positions.dtype == scalar_positions.dtype
    assert_array_equal(positions, scalar_positions)
