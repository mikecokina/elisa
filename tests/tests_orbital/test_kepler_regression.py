"""Public orbital outputs and input behavior for both Kepler solvers."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from numpy.typing import NDArray

from elisa import settings
from elisa.base.error import ElisaMaxIterationError
from elisa.binary_system.orbit.orbit import Orbit
from elisa.conf.enums import MeanAnomalySolver
from tests.kepler_helpers import scalar_orbital_motion


@pytest.fixture(autouse=True, params=list(MeanAnomalySolver))
def solver_setting(request: pytest.FixtureRequest) -> Iterator[None]:
    with settings.override(MEAN_ANOMALY_SOLVER=request.param):
        yield


@pytest.mark.parametrize(
    ("eccentricity", "periastron_degrees"),
    [
        pytest.param(0.0, 270.0, id="circular"),
        pytest.param(0.3, 25.0, id="moderately-eccentric"),
        pytest.param(0.95, 75.0, id="highly-eccentric"),
        pytest.param(0.999999, 90.0, id="near-parabolic"),
    ],
)
def test_orbital_outputs_match_scalar_reference(eccentricity: float, periastron_degrees: float) -> None:
    orbit = Orbit(
        period=3.0,
        inclination=np.pi / 2,
        eccentricity=eccentricity,
        argument_of_periastron=np.radians(periastron_degrees),
    )
    phases = np.array([0.5, -0.1, 0.0, 0.125, 0.5, 1.0, 1.1, -1.0, 2.0])
    original_phases = phases.copy()
    scalar_positions = scalar_orbital_motion(orbit, phases)

    positions = orbit.orbital_motion(phases)

    assert positions.shape == (len(phases), 4)
    # Compare raw angles: cosine or modulo comparisons would hide a 0/2pi change.
    assert_allclose(positions[:, :3], scalar_positions[:, :3], rtol=0.0, atol=1e-10)
    assert_array_equal(positions[:, 3], original_phases)
    assert_array_equal(phases, original_phases)
    assert_array_equal(positions[0], positions[4])

    for phase_index in (0, 2, 5):
        single_phase = phases[phase_index : phase_index + 1]
        array_position = orbit.orbital_motion(single_phase)
        scalar_position = orbit.orbital_motion(float(phases[phase_index]))
        expected_position = scalar_positions[phase_index : phase_index + 1]

        assert array_position.shape == scalar_position.shape == (1, 4)
        assert_allclose(array_position, expected_position, rtol=0.0, atol=1e-10)
        assert_array_equal(scalar_position, array_position)


def test_negative_eccentric_anomaly_is_corrected_once() -> None:
    orbit = Orbit(period=3.0, inclination=np.pi / 2, eccentricity=0.0, argument_of_periastron=0.0)
    mean_anomaly = -4 * np.pi - 0.1

    eccentric_anomaly = orbit.mean_anomaly_to_eccentric_anomaly(mean_anomaly)

    assert_allclose(eccentric_anomaly, -2 * np.pi - 0.1, rtol=0.0, atol=1e-10)
    assert eccentric_anomaly < 0  # Arbitrarily wrapped anomalies are not folded modulo 2pi.


@pytest.mark.parametrize(
    ("phases", "error_type"),
    [
        pytest.param(np.array(0.2), TypeError, id="zero-dimensional-array"),
        pytest.param(np.ones((2, 2)), ValueError, id="two-dimensional-array"),
    ],
)
def test_unsupported_orbital_shapes_preserve_exception_type(phases: NDArray, error_type: type[Exception]) -> None:
    orbit = Orbit(period=3.0, inclination=np.pi / 2, eccentricity=0.3, argument_of_periastron=0.7)

    with pytest.raises(error_type):
        orbit.orbital_motion(phases)


def test_empty_orbital_input_retains_four_columns() -> None:
    orbit = Orbit(period=3.0, inclination=np.pi / 2, eccentricity=0.3, argument_of_periastron=0.7)

    positions = orbit.orbital_motion(np.array([]))

    assert positions.shape == (0, 4)


@pytest.mark.parametrize("nonfinite_phase", [np.nan, np.inf, -np.inf], ids=["nan", "positive-inf", "negative-inf"])
@pytest.mark.parametrize("strict_errors", [False, True], ids=["warn", "raise"])
@pytest.mark.filterwarnings("ignore:Tolerance of .* reached:RuntimeWarning")
def test_nonfinite_phase_fails_without_leaking_error_policy(nonfinite_phase: float, strict_errors: bool) -> None:
    orbit = Orbit(period=3.0, inclination=np.pi / 2, eccentricity=0.3, argument_of_periastron=0.7)
    phases = np.array([0.1, nonfinite_phase, 0.2])
    if strict_errors:
        error_policy = np.errstate(all="raise")
    else:
        error_policy = np.errstate(divide="warn", invalid="warn", over="warn", under="ignore")

    with error_policy:
        original_policy = np.geterr()
        with pytest.raises(ElisaMaxIterationError, match="Kepler solver failed to converge") as raised_error:
            orbit.orbital_motion(phases)

        assert type(raised_error.value) is ElisaMaxIterationError
        assert raised_error.value.__cause__ is None
        assert np.geterr() == original_policy


def test_successful_orbital_solve_preserves_caller_error_policy() -> None:
    orbit = Orbit(period=3.0, inclination=np.pi / 2, eccentricity=0.3, argument_of_periastron=0.7)
    phases = np.array([0.3, 0.1, 0.3, -0.2])

    with np.errstate(divide="ignore", invalid="raise", over="raise", under="warn"):
        original_policy = np.geterr()
        scalar_positions = scalar_orbital_motion(orbit, phases)

        positions = orbit.orbital_motion(phases)

        assert_allclose(positions, scalar_positions, rtol=0.0, atol=1e-10)
        assert np.geterr() == original_policy
