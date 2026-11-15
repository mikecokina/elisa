"""Verify scaled Cartesian harmonics against independent numerical references."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import sph_harm_y

from elisa.pulse.mode import PulsationMode
from elisa.pulse.pulsations import cartesian_harmonics

if TYPE_CHECKING:
    from numpy.typing import NDArray


class _StoredOracleCase(TypedDict):
    l: int  # noqa: E741 — degree key in the frozen oracle's JSON schema.
    m: int
    n: list[float]
    label: str
    value: list[float]
    gradient: list[list[float]]


@dataclass(frozen=True)
class _OracleCase:
    degree: int
    order: int
    direction: NDArray[np.float64]
    label: str
    value: complex
    gradient: NDArray[np.complex128]


def _load_oracle_cases() -> tuple[_OracleCase, ...]:
    # The audit's explicit Legendre sum was evaluated at l+120 and l+160
    # decimal digits. Frozen results keep mpmath out of test dependencies.
    path = Path(__file__).parent / "data" / "cartesian_harmonic_oracle.json"
    stored_cases: list[_StoredOracleCase] = json.loads(path.read_text(encoding="utf-8"))
    return tuple(
        _OracleCase(
            degree=case["l"],
            order=case["m"],
            direction=np.asarray(case["n"], dtype=np.float64),
            label=case["label"],
            value=complex(*case["value"]),
            gradient=np.asarray([complex(*component) for component in case["gradient"]], dtype=np.complex128),
        )
        for case in stored_cases
    )


def _absolute_error_bounds(case: _OracleCase) -> tuple[float, float]:
    value_bound = SUBNORMAL_ERROR_BOUND
    gradient_bound = 2e-11 * np.max(np.abs(case.gradient)) + SUBNORMAL_ERROR_BOUND
    # Projection subtracts a radial derivative of scale l*|Y|, which sets
    # an absolute rounding floor even when the tangent gradient is tiny.
    gradient_bound = max(gradient_bound, 8 * np.finfo(float).eps * case.degree * abs(case.value))
    if case.label in ("stationary", "zero"):
        # Binary64 root locations need absolute tests using addition-theorem
        # scales. These bounds apply only to root and stationary-point cases.
        value_bound = max(value_bound, 1e-13 * np.sqrt(2 * case.degree + 1))
        gradient_scale = np.sqrt(case.degree * (case.degree + 1) * (2 * case.degree + 1))
        gradient_bound = max(gradient_bound, 1e-13 * gradient_scale)
    return float(value_bound), float(gradient_bound)


def _scipy_harmonic(
    degree: int,
    order: int,
    theta: NDArray[np.float64],
    phi: NDArray[np.float64],
) -> NDArray[np.complex128]:
    if abs(order) > degree:
        return np.zeros(theta.shape, dtype=np.complex128)
    return np.sqrt(4 * np.pi) * sph_harm_y(degree, order, theta, phi)


ORACLE_CASES = _load_oracle_cases()
ORACLE_IDS = [f"{case.label}-l{case.degree}-m{case.order}-{index}" for index, case in enumerate(ORACLE_CASES)]
# Four least-subnormal units cover reconstruction and phase rounding without
# concealing loss of larger representable outputs.
SUBNORMAL_ERROR_BOUND = 4 * np.nextafter(0.0, 1.0)
COMPONENT_CASES = tuple(
    case
    for case in ORACLE_CASES
    if case.label == "seed"
    and (case.direction[1] == 0 or np.any((np.abs(case.direction) > 0) & (np.abs(case.direction) < 2.0**-900)))
)
COMPONENT_IDS = [f"l{case.degree}-m{case.order}-{index}" for index, case in enumerate(COMPONENT_CASES)]


@pytest.mark.parametrize("case", ORACLE_CASES, ids=ORACLE_IDS)
@pytest.mark.parametrize("sign", [1, -1], ids=["positive-order", "negative-order"])
def test_scaled_recurrence_independent_oracle(case: _OracleCase, sign: int) -> None:
    mode = PulsationMode(l=case.degree, m=sign * case.order, amplitude=1.0, frequency=1.0)
    reference_value = case.value
    reference_gradient = case.gradient
    if sign < 0:
        reference_value = (-1) ** case.order * reference_value.conjugate()
        reference_gradient = (-1) ** case.order * reference_gradient.conjugate()
    phase = np.exp(-0.73j)
    values, gradients = cartesian_harmonics(mode, case.direction[None, :], phase)

    assert values.dtype == np.dtype("complex128")
    assert gradients.dtype == np.dtype("complex128")
    assert np.isfinite(values).all()
    assert np.isfinite(gradients).all()
    value_bound, gradient_bound = _absolute_error_bounds(case)
    assert_allclose(values[0], phase * reference_value, rtol=2e-10, atol=value_bound)
    assert_allclose(gradients[0], phase * reference_gradient, rtol=2e-10, atol=gradient_bound)
    if abs(reference_value) > value_bound:
        assert values[0] != 0
    significant_components = np.abs(phase * reference_gradient) > gradient_bound
    assert np.all(gradients[0, significant_components] != 0)


@pytest.mark.parametrize("degree", [1, 5, 10, 30, 100])
def test_ordinary_modes_against_scipy_ladder(degree: int) -> None:
    rng = np.random.default_rng(20261112)
    directions = rng.normal(size=(32, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    theta = np.arctan2(np.hypot(directions[:, 0], directions[:, 1]), directions[:, 2])
    phi = np.arctan2(directions[:, 1], directions[:, 0])
    for order in range(-degree, degree + 1):
        expected_values = _scipy_harmonic(degree, order, theta, phi)
        raising = np.sqrt((degree - order) * (degree + order + 1)) * _scipy_harmonic(degree, order + 1, theta, phi)
        lowering = np.sqrt((degree + order) * (degree - order + 1)) * _scipy_harmonic(degree, order - 1, theta, phi)
        angular_x = (raising + lowering) / 2
        angular_y = (raising - lowering) / (2j)
        angular_z = order * expected_values
        angular_momentum = np.column_stack((angular_x, angular_y, angular_z))
        expected_gradients = -1j * np.cross(directions, angular_momentum)
        mode = PulsationMode(l=degree, m=order, amplitude=1.0, frequency=1.0)

        values, gradients = cartesian_harmonics(mode, directions, 1)

        assert_allclose(values, expected_values, rtol=2e-11, atol=2e-12)
        assert_allclose(gradients, expected_gradients, rtol=2e-11, atol=2e-10)


def test_mixed_batch_keeps_independent_binary_scales() -> None:
    selected = [case for case in ORACLE_CASES if (case.degree, case.order) == (3000, 500)]
    directions = np.vstack([*(case.direction for case in selected), [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    mode = PulsationMode(l=3000, m=-500, amplitude=1.0, frequency=1.0)
    phase = np.exp(0.29j)

    values, gradients = cartesian_harmonics(mode, directions, phase)

    for index, direction in enumerate(directions):
        individual_values, individual_gradients = cartesian_harmonics(mode, direction[None, :], phase)
        assert_allclose(values[index], individual_values[0], rtol=1e-13, atol=0)
        assert_allclose(gradients[index], individual_gradients[0], rtol=1e-13, atol=0)
    assert_allclose(gradients[-2:], 0, atol=0)
    assert_allclose(values[-2:], 0, atol=0)


@pytest.mark.parametrize("case", COMPONENT_CASES, ids=COMPONENT_IDS)
def test_seed_reconstruction_preserves_tiny_real_and_imaginary_parts(case: _OracleCase) -> None:
    mode = PulsationMode(l=case.degree, m=case.order, amplitude=1.0, frequency=1.0)
    values, gradients = cartesian_harmonics(mode, case.direction[None, :], 1)
    actual_components = np.concatenate((values, gradients[0]))
    expected_components = np.concatenate(([case.value], case.gradient))
    actual_parts = np.column_stack((actual_components.real, actual_components.imag)).ravel()
    expected_parts = np.column_stack((expected_components.real, expected_components.imag)).ravel()

    assert_allclose(actual_parts, expected_parts, rtol=2e-10, atol=SUBNORMAL_ERROR_BOUND)
    significant_parts = np.abs(expected_parts) > SUBNORMAL_ERROR_BOUND
    assert np.all(actual_parts[significant_parts] != 0)
