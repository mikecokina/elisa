"""Physical-vector API contracts and explicit plotting projections."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from elisa import units, utils
from elisa.base.container import StarContainer
from elisa.base.graphics import plot
from elisa.pulse import container_ops
from elisa.pulse.mode import PulsationMode


@pytest.fixture
def vector_star() -> StarContainer:
    """Use nonradial vectors and unequal radii, including both coordinate poles."""
    equilibrium = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, -1.0], [2.0, 1.0, 0.0]])
    star = StarContainer(points=equilibrium + np.array([5.0, 0.0, 0.0]), faces=np.array([[0, 1, 2]]))
    star.pulsation_equilibrium_points = equilibrium
    star.pulsation_directions = equilibrium / np.linalg.norm(equilibrium, axis=1)[:, None]
    star.pulsation_rotation = np.eye(3)
    star.points_spherical = utils.cartesian_to_spherical(equilibrium)
    mode = PulsationMode(l=1, m=1, amplitude=1.0, frequency=3.0 / (2 * np.pi) / units.s)
    mode.complex_displacement = np.array(
        [
            [0.2 + 0.5j, 0.4 + 0.1j, 0.1 - 0.2j],
            [0.3 - 0.3j, -0.2 + 0.2j, 0.2 + 0.1j],
            [-0.1 + 0.1j, 0.3 - 0.4j, 0.4 + 0.2j],
        ],
    )
    star.pulsations = {0: mode}
    return star


@pytest.mark.parametrize("scale", [1.0, 7e9])
def test_perturbations_return_cartesian_vectors(vector_star: StarContainer, scale: float) -> None:
    """Component order is always x,y,z; averaging never changes representation."""
    star = vector_star
    assert star.points is not None
    reference_points = star.points.copy()
    mode: PulsationMode = star.pulsations[0]
    assert mode.complex_displacement is not None
    assert star.faces is not None
    assert star.pulsation_equilibrium_points is not None
    expected_displacement = np.real(mode.complex_displacement)
    expected_velocity = scale * (mode.angular_frequency * np.imag(mode.complex_displacement))
    expected_acceleration = scale * (-(mode.angular_frequency**2) * np.real(mode.complex_displacement))

    displacement = container_ops.position_perturbation(star, update_container=False, return_perturbation=True)
    assert displacement is not None
    assert_array_equal(displacement, expected_displacement)
    assert_array_equal(star.points, reference_points)
    for point_values in [True, False]:
        velocity = container_ops.velocity_perturbation(
            star,
            scale,
            return_perturbation=True,
            point_perturbations=point_values,
        )
        acceleration = container_ops.gravity_acc_perturbation(
            star,
            scale,
            return_perturbation=True,
            point_perturbations=point_values,
        )
        assert velocity is not None
        assert acceleration is not None
        if point_values:
            assert_allclose(velocity, expected_velocity, rtol=2e-16, atol=0)
            assert_allclose(acceleration, expected_acceleration, rtol=2e-16, atol=0)
        else:
            assert_allclose(velocity, expected_velocity[star.faces].mean(axis=1), rtol=2e-16, atol=0)
            assert_allclose(acceleration, expected_acceleration[star.faces].mean(axis=1), rtol=2e-16, atol=0)

    container_ops.position_perturbation(star, com_x=5.0)
    assert_array_equal(
        star.points,
        star.pulsation_equilibrium_points + expected_displacement + np.array([5.0, 0.0, 0.0]),
    )


@pytest.mark.parametrize("scale", [1.0, 7e9])
def test_kinematic_colormaps_project_cartesian_face_vectors(vector_star: StarContainer, scale: float) -> None:
    """Verify projections by radial dot products, without the projection helper."""
    star = vector_star
    mode: PulsationMode = star.pulsations[0]
    assert mode.complex_displacement is not None
    assert star.pulsation_equilibrium_points is not None
    assert star.faces is not None
    centers = star.pulsation_equilibrium_points[star.faces].mean(axis=1)
    radial_direction = centers / np.linalg.norm(centers, axis=1)[:, None]
    velocity = (mode.angular_frequency * np.imag(mode.complex_displacement) * scale)[star.faces].mean(axis=1)
    acceleration = (-(mode.angular_frequency**2) * np.real(mode.complex_displacement) * scale)[star.faces].mean(axis=1)
    radial_velocity = np.sum(velocity * radial_direction, axis=1)
    radial_acceleration = np.sum(acceleration * radial_direction, axis=1)
    horizontal_velocity = np.linalg.norm(velocity - radial_velocity[:, None] * radial_direction, axis=1)
    horizontal_acceleration = np.linalg.norm(acceleration - radial_acceleration[:, None] * radial_direction, axis=1)

    arguments = ("linear", "default", True, scale)
    assert_allclose(plot.v_rad_pert_cmap(star, *arguments), radial_velocity, rtol=1e-14)
    assert_allclose(plot.v_horizontal_pert_cmap(star, *arguments), horizontal_velocity, rtol=1e-14)
    assert_allclose(plot.horizontal_g_pert_cmap(star, *arguments), horizontal_acceleration, rtol=1e-14)
    assert_allclose(
        plot.g_cmap(star, "linear", units.ACCELERATION_UNIT, True, scale),
        radial_acceleration,
        rtol=1e-14,
    )


@pytest.mark.parametrize("scale", [1.0, 7e9])
def test_displacement_colormaps_project_vectors_without_mesh_mutation(vector_star: StarContainer, scale: float) -> None:
    star = vector_star
    assert star.points is not None
    reference_points = star.points.copy()
    mode: PulsationMode = star.pulsations[0]
    assert mode.complex_displacement is not None
    assert star.pulsation_equilibrium_points is not None
    assert star.faces is not None
    displacement = np.real(mode.complex_displacement)
    directions = star.pulsation_directions
    assert directions is not None
    radial = np.sum(displacement * directions, axis=1)
    horizontal = np.linalg.norm(displacement - radial[:, None] * directions, axis=1)

    assert_allclose(plot.r_cmap(star, "linear", "default", True, scale), radial[star.faces].mean(axis=1) * scale)
    assert_allclose(
        plot.horizonatal_displacement_cmap(star, "linear", "default", True, scale),
        horizontal[star.faces].mean(axis=1) * scale,
    )
    assert_array_equal(star.points, reference_points)

    # Finite differences of spherical radius include horizontal motion. They
    # are deliberately distinct from the radial projection plotted above.
    radius_difference = np.linalg.norm(star.pulsation_equilibrium_points + displacement, axis=1)
    radius_difference -= np.linalg.norm(star.pulsation_equilibrium_points, axis=1)
    assert not np.allclose(radius_difference, radial)
