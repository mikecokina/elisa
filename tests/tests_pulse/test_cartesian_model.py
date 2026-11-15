"""Independent geometry and frozen-equilibrium perturbation kinematics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import assert_allclose

from elisa import settings, utils
from elisa import units as u
from elisa.base.container import StarContainer
from elisa.base.error import ElisaUnexpectedError, PulsationModeError
from elisa.pulse import container_ops, pulsations
from elisa.pulse import utils as pu
from elisa.pulse.mode import PulsationMode
from elisa.pulse.surface import kinematics
from tests.tests_pulse import test_synthesis

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray


def _mode(
    degree: int = 1,
    order: int = 1,
    radial_amplitude: float = 1.0,
    horizontal_amplitude: float = 1.0,
) -> PulsationMode:
    """Create a real mode with controlled physical displacement coefficients."""
    mode = PulsationMode(l=degree, m=order, amplitude=1.0, frequency=1.0)
    mode.radial_amplitude = radial_amplitude
    mode.horizontal_amplitude = horizontal_amplitude
    return mode


def _evaluate_harmonics(degree: int, order: int, directions: NDArray, phase: complex = 1.0) -> tuple[NDArray, NDArray]:
    return pulsations.cartesian_harmonics(_mode(degree, order), directions, phase)


def _polynomial_harmonics(mode: PulsationMode, directions: NDArray, exponential: complex) -> tuple[NDArray, NDArray]:
    """Independent low-degree polynomial values and tangent gradients."""
    if mode.l == 0:
        return np.full(len(directions), exponential), np.zeros_like(directions, dtype=complex)
    if mode.l == 1:
        coefficients = -np.sqrt(1.5) * np.array([1, 1j, 0])
        values = directions @ coefficients
        derivative = np.broadcast_to(coefficients, directions.shape)
    else:
        assert (mode.l, mode.m) == (2, -1)
        coefficient = np.sqrt(7.5)
        transverse = directions[:, 0] - 1j * directions[:, 1]
        values = coefficient * directions[:, 2] * transverse
        derivative = coefficient * np.column_stack((directions[:, 2], -1j * directions[:, 2], transverse))
    return exponential * values, exponential * (derivative - mode.l * values[:, None] * directions)


def _dipole_displacement(
    mode: PulsationMode,
    positions: NDArray,
    rotation: NDArray,
    time: float,
) -> NDArray:
    """Compute a radius-weighted dipole displacement directly in physical units."""
    assert mode.radial_amplitude is not None
    assert mode.horizontal_amplitude is not None
    radii = np.linalg.norm(positions, axis=1)
    directions = positions / radii[:, None]
    coefficients = -np.sqrt(1.5) * np.array([1, 1j, 0])
    tilted_directions = directions @ rotation
    exponential = np.exp(-1j * (mode.angular_frequency * time + mode.start_phase))
    values = (tilted_directions @ coefficients) * exponential
    gradient = (coefficients[None, :] * exponential - values[:, None] * tilted_directions) @ rotation.T
    radial = mode.radial_amplitude * values[:, None] * directions
    horizontal = (
        mode.horizontal_amplitude
        * radii[:, None]
        * gradient
        / np.sqrt(np.mean(radii**2 * np.sum(abs(gradient) ** 2, axis=1)))
    )
    return radial + horizontal


@pytest.mark.parametrize("degree", range(9))
def test_gradient_against_independent_angular_derivatives(degree: int) -> None:
    # Away from coordinate singularities, compare to SciPy values and finite
    # differences along two great-circle tangent directions, not the recurrence.
    rng = np.random.default_rng(104)
    directions = rng.normal(size=(31, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    points = utils.cartesian_to_spherical(directions)
    step = 2e-6
    for order in range(-degree, degree + 1):
        mode = _mode(degree, order)
        values, gradient = _evaluate_harmonics(degree, order, directions, np.exp(-0.37j))
        assert_allclose(values, pulsations.spherical_harmonics(mode, points, np.exp(-0.37j)), atol=2e-13)
        assert_allclose(np.sum(gradient * directions, axis=1), 0, atol=2e-12)
        for direction in [np.array([1.0, 0, 0]), np.array([0.0, 1, 0])]:
            tangent = direction - (directions @ direction)[:, None] * directions
            tangent /= np.linalg.norm(tangent, axis=1)[:, None]
            plus = utils.cartesian_to_spherical(np.cos(step) * directions + np.sin(step) * tangent)
            minus = utils.cartesian_to_spherical(np.cos(step) * directions - np.sin(step) * tangent)
            finite_difference = (
                pulsations.spherical_harmonics(mode, plus, np.exp(-0.37j))
                - pulsations.spherical_harmonics(mode, minus, np.exp(-0.37j))
            ) / (2 * step)
            assert_allclose(np.sum(gradient * tangent, axis=1), finite_difference, atol=2e-8, rtol=2e-8)


@pytest.mark.parametrize("degree", [1, 2, 5, 12, 30])
@pytest.mark.parametrize("south", [False, True])
def test_exact_poles_and_multimeridian_continuity(degree: int, south: bool) -> None:
    sign = -1.0 if south else 1.0
    pole = np.array([[0.0, 0.0, sign]])
    pole_coefficient = -np.sqrt((2 * degree + 1) * degree * (degree + 1)) / 2
    if south:
        pole_coefficient *= (-1) ** (degree + 1)
    for order in [-degree, -1, 0, 1, degree]:
        _, gradient = _evaluate_harmonics(degree, order, pole)
        expected = pole_coefficient * np.array([[1, 1j, 0]]) if abs(order) == 1 else np.zeros((1, 3))
        if order == -1:
            expected = -expected.conj()
        assert_allclose(gradient, expected, atol=1e-12)
        for distance in [1e-4, 1e-8, 1e-12]:
            phi = np.arange(8) * np.pi / 4
            directions = np.column_stack(
                (distance * np.cos(phi), distance * np.sin(phi), np.full(8, sign * np.sqrt(1 - distance**2))),
            )
            _, near_pole_gradient = _evaluate_harmonics(degree, order, directions)
            assert_allclose(
                near_pole_gradient,
                np.repeat(expected, 8, axis=0),
                atol=distance * 2 * (degree + 1) ** 2 * np.sqrt(2 * degree + 1),
                rtol=0,
            )
    # No azimuth is an input to the evaluator at a Cartesian pole.


@pytest.mark.parametrize("m", [-1, 0, 1])
def test_analytic_dipole_and_radial(m: int) -> None:
    directions = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.3, 0.4, np.sqrt(0.75)]])
    coefficients = np.array([0.0, 0.0, np.sqrt(3)]) if m == 0 else -np.sqrt(1.5) * np.array([1, 1j, 0])
    if m == -1:
        coefficients = -coefficients.conj()
    values, gradient = _evaluate_harmonics(1, m, directions)
    assert_allclose(values, directions @ coefficients, atol=1e-15)
    assert_allclose(gradient, coefficients - (directions @ coefficients)[:, None] * directions, atol=1e-15)
    values, gradient = _evaluate_harmonics(0, 0, directions, np.exp(-0.7j))
    assert_allclose(values, np.exp(-0.7j))
    assert_allclose(gradient, 0, atol=0)


@pytest.mark.parametrize("q", [-2.0, 0.0, 2.0])
@pytest.mark.parametrize("scale", [1.0, 7e9])
def test_radius_weighted_complex_rms_and_signed_amplitudes(q: float, scale: float) -> None:
    directions = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    radius = np.array([1.0, 1.3, 1.7, 2.1])
    values, gradient = _evaluate_harmonics(1, 1, directions, np.exp(-0.4j))
    mode = _mode(radial_amplitude=3.0, horizontal_amplitude=q * 3.0)
    displacement = (
        kinematics.calculate_cartesian_displacement(mode, directions, values, gradient, radius / scale, scale) * scale
    )
    radial = np.sum(displacement * directions, axis=1)
    horizontal = displacement - radial[:, None] * directions
    assert_allclose(radial, 3 * values, atol=2e-15)
    assert_allclose(np.sqrt(np.mean(np.sum(abs(horizontal) ** 2, axis=1))), abs(q * 3), atol=2e-15)
    expected = q * 3 * radius[:, None] * gradient / np.sqrt(np.mean(radius**2 * np.sum(abs(gradient) ** 2, axis=1)))
    assert_allclose(horizontal, expected, atol=3e-15)


@pytest.fixture
def system_factory() -> Iterator[test_synthesis.TestPulsationModule]:
    case = test_synthesis.TestPulsationModule()
    case.setUp()
    try:
        yield case
    finally:
        case.tearDown()


def _frozen_surface(
    system_factory: test_synthesis.TestPulsationModule,
    kind: str,
    modes: list[dict[str, object]],
) -> tuple[StarContainer, float, float]:
    """Prepare equilibrium geometry with explicit single/binary coordinate scales."""
    if kind == "binary":
        binary = system_factory.prepare_binary(modes)
        binary_container = binary.build_container(phase=0.17, build_pulsations=False)
        star = binary_container.secondary
        scale = float(binary.semi_major_axis)
        com_x = float(binary_container.position.distance)
    else:
        single = system_factory.prepare_system(modes)
        if kind == "spotted":
            single.star.spots = [
                {
                    "longitude": 40 * u.deg,
                    "latitude": 60 * u.deg,
                    "angular_radius": 20 * u.deg,
                    "temperature_factor": 0.9,
                },
            ]
        star = single.build_container(phase=0.17, build_pulsations=False).star
        scale = 1.0
        com_x = 0.0
    return star, scale, com_x


@pytest.mark.parametrize("kind", ["single", "binary", "tilted", "multiple", "spotted"])
def test_frozen_pipeline_time_derivatives(system_factory: test_synthesis.TestPulsationModule, kind: str) -> None:
    meta: dict[str, object] = {
        "l": 1,
        "m": 1,
        "amplitude": 100 * u.m / u.s,
        "frequency": 15 / u.d,
        "start_phase": 0.4,
        "horizontal_to_radial_amplitude_ratio": -1.0,
        "temperature_amplitude_factor": 0.01,
    }
    if kind in ["tilted", "multiple"]:
        meta.update(mode_axis_phi=21 * u.deg, mode_axis_theta=37 * u.deg)
    modes = [meta]
    if kind == "multiple":
        modes.append(dict(meta, l=2, m=-1, frequency=19 / u.d, start_phase=0.8))
    star, scale, com_x = _frozen_surface(system_factory, kind, modes)
    star.flat_it()
    assert star.points is not None
    equilibrium = star.points.copy()
    time = 123.4

    def evaluate(evaluation_time: float) -> NDArray:
        star.points = equilibrium.copy()
        # Fixed phase holds the sampled equilibrium and axis orientation fixed.
        container_ops.generate_harmonics(star, com_x, 0.17, evaluation_time)
        container_ops.complex_displacement(star, scale)
        container_ops.position_perturbation(star, com_x)
        assert star.points is not None
        return star.points.copy() * scale

    step = 2.0
    minus = evaluate(time - step)
    plus = evaluate(time + step)
    center = evaluate(time)
    velocity = container_ops.velocity_perturbation(star, scale, return_perturbation=True, point_perturbations=True)
    acceleration = container_ops.gravity_acc_perturbation(
        star,
        scale,
        return_perturbation=True,
        point_perturbations=True,
    )
    assert velocity is not None
    assert acceleration is not None
    assert_allclose((plus - minus) / (2 * step), velocity, atol=2e-4, rtol=2e-5)
    assert_allclose((plus - 2 * center + minus) / step**2, acceleration, atol=2e-6, rtol=2e-5)
    # Direct polynomial dipole oracle in the tilted frame, including distorted
    # radius weighting and the actual exp(-i(omega*t+phase)) convention.
    rotation = pu.mode_rotation(star.pulsations[0].tilt_phi, star.pulsations[0].tilt_theta)
    local_positions = (equilibrium - np.array([com_x, 0.0, 0.0])) * scale
    mode = star.pulsations[0]
    expected = _dipole_displacement(mode, local_positions, rotation, time)
    assert_allclose(mode.complex_displacement * scale, expected, atol=1e-9)
    sum_displacement = sum(
        mode_contribution.complex_displacement.real for mode_contribution in star.pulsations.values()
    )
    assert_allclose(star.points - equilibrium, sum_displacement, atol=3e-7 / scale + 2e-16)
    assert star.velocities is not None
    assert star.faces is not None
    initial_velocity = star.velocities.copy()
    container_ops.velocity_perturbation(star, scale, update_container=True)
    assert_allclose(star.velocities - initial_velocity, velocity[star.faces].mean(axis=1), atol=1e-11)
    # Temperature uses the radial coefficient, independent of the x column.
    expected_temperature = (
        sum(
            mode_contribution.temperature_amplitude_factor
            * np.real(
                pu.generate_phase_shift(mode_contribution.temperature_perturbation_phase_shift)
                * mode_contribution.point_harmonics,
            )
            for mode_contribution in star.pulsations.values()
        )
        * star.t_eff
    )
    temperature_perturbation = container_ops.temp_perturbation(
        star,
        scale,
        return_perturbation=True,
        point_perturbations=True,
    )
    assert temperature_perturbation is not None
    assert_allclose(temperature_perturbation, expected_temperature, atol=1e-12)


def test_moving_frame_contract() -> None:
    """Transport changes orientation, not the intrinsic harmonic derivative.

    Q(t)*v_intrinsic differs from d(Q(t)*d)/dt by Q_dot*d. Test that
    deliberately excluded term explicitly instead of claiming inertial closure.
    """
    directions = np.array([[0.3, 0.4, np.sqrt(0.75)], [0.0, 0.0, 1.0]])
    values, gradient = _evaluate_harmonics(1, 1, directions)
    mode = _mode(radial_amplitude=2.0, horizontal_amplitude=3.0)
    initial_displacement = kinematics.calculate_cartesian_displacement(mode, directions, values, gradient, np.ones(2))
    angular_frequency = 1.7
    rotation_rate = 0.13
    time = 0.41
    step = 1e-4

    def rotated_displacement(time: float) -> NDArray:
        return (initial_displacement * np.exp(-1j * angular_frequency * time)).real @ pu.mode_rotation(
            rotation_rate * time,
            0.2,
        ).T

    rotation = pu.mode_rotation(rotation_rate * time, 0.2)
    complex_displacement = initial_displacement * np.exp(-1j * angular_frequency * time)
    velocity = kinematics.calculate_mode_derivatives(complex_displacement, angular_frequency) @ rotation.T
    position = rotated_displacement(time)
    transport = rotation_rate * np.column_stack((-position[:, 1], position[:, 0], np.zeros(2)))
    assert_allclose(
        (rotated_displacement(time + step) - rotated_displacement(time - step)) / (2 * step),
        velocity + transport,
        atol=3e-8,
    )
    assert np.linalg.norm(transport) > 0.01


@pytest.mark.requires_ld(passbands=("Generic.Bessell.V", "bolometric"))
@pytest.mark.requires_atmosphere
@pytest.mark.parametrize("kind", ["radial", "nonradial", "tilted", "multiple", "spotted", "binary"])
def test_atmosphere_observables_against_polynomial_oracle(
    system_factory: test_synthesis.TestPulsationModule,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    """Full LC/RV synthesis with independent low-degree harmonic polynomials."""
    from elisa import Observer

    meta: dict[str, object] = {
        "l": 1,
        "m": 1,
        "amplitude": 100 * u.m / u.s,
        "frequency": 15 / u.d,
        "start_phase": 0.4,
        "horizontal_to_radial_amplitude_ratio": 1.0,
        "temperature_amplitude_factor": 0.01,
    }
    if kind == "radial":
        meta.update(l=0, m=0, horizontal_to_radial_amplitude_ratio=0.0)
    if kind in ["tilted", "multiple"]:
        meta.update(mode_axis_phi=21 * u.deg, mode_axis_theta=37 * u.deg)
    modes = [meta]
    if kind == "multiple":
        modes.append(dict(meta, l=2, m=-1, frequency=19 / u.d))
    single = system_factory.prepare_system(modes) if kind != "binary" else None
    if single is not None and kind == "spotted":
        single.star.spots = [
            {"longitude": 40 * u.deg, "latitude": 60 * u.deg, "angular_radius": 20 * u.deg, "temperature_factor": 0.9},
        ]
    system = single if single is not None else system_factory.prepare_binary(modes)
    phases = np.linspace(0.025, 0.025 + 1 / (15 * system.period), 5)
    observer = Observer(passband=["Generic.Bessell.V"], system=system)
    flux = observer.lc(phases=phases)[1]
    rv = observer.rv(phases=phases, method="radiometric")[1]

    monkeypatch.setattr(pulsations, "cartesian_harmonics", _polynomial_harmonics)
    independent_flux = observer.lc(phases=phases)[1]
    independent_rv = observer.rv(phases=phases, method="radiometric")[1]
    for band in flux:
        assert_allclose(flux[band], independent_flux[band], rtol=2e-13, atol=0)
    for component in rv:
        assert_allclose(rv[component], independent_rv[component], rtol=2e-12, atol=1e-8)


@pytest.mark.parametrize(
    "operation",
    [
        "complex_displacement",
        "position_perturbation",
        "velocity_perturbation",
        "gravity_acc_perturbation",
    ],
)
def test_unprepared_pulsation_reference_is_rejected(operation: str) -> None:
    star = StarContainer()
    assert star.pulsation_rotation is None
    assert star.pulsation_directions is None
    assert star.pulsation_equilibrium_points is None
    star.flat_it()
    with pytest.raises(ElisaUnexpectedError, match="generate_harmonics"):
        getattr(container_ops, operation)(star, 1.0)


def test_cartesian_displacement_rejects_unsupported_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "PULSATION_MODEL", "unsupported")
    mode = _mode()
    with pytest.raises(NotImplementedError, match="unsupported"):
        kinematics.calculate_cartesian_displacement(mode, np.eye(3), np.ones(3), np.zeros((3, 3)), np.ones(3))


def test_cartesian_displacement_rejects_zero_horizontal_rms() -> None:
    mode = _mode()
    with pytest.raises(PulsationModeError, match="zero discrete RMS"):
        kinematics.calculate_cartesian_displacement(mode, np.eye(3), np.ones(3), np.zeros((3, 3)), np.ones(3))


def test_rotation_and_spherical_vector_diagnostics() -> None:
    # Orthonormal components are physical vector projections, never coordinate
    # increments. Include both poles and verify the meridian-independent norm.
    points = np.array([[2.0, 0.7, 0.0], [3.0, 2.1, np.pi], [4.0, 0.4, 1.2]])
    vectors = np.array([[1.0, 2.0, 3.0], [-2.0, 3.0, 4.0], [5.0, -1.0, 2.0]])
    components = pu.spherical_vector_components(vectors, points)
    directions = utils.spherical_to_cartesian(points) / points[:, :1]
    radial = np.sum(vectors * directions, axis=1)
    assert_allclose(components[:, 0], radial, atol=2e-15)
    assert_allclose(np.sum(components**2, axis=1), np.sum(vectors**2, axis=1), atol=1e-14)
    assert_allclose(
        np.linalg.norm(components[:, 1:], axis=1),
        np.linalg.norm(vectors - radial[:, None] * directions, axis=1),
        atol=2e-15,
    )
    rotation = pu.mode_rotation(0.7, 1.2)
    assert_allclose(rotation.T @ rotation, np.eye(3), atol=2e-16)
    # General point-coordinate utilities are retained; their rotation must agree
    # with the Cartesian frame convention independently of coordinate charts.
    tilted = pu.tilt_mode_coordinates(points, 0.7, 1.2)
    assert_allclose(utils.spherical_to_cartesian(tilted), utils.spherical_to_cartesian(points) @ rotation, atol=3e-15)
