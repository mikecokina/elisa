"""Rotation checks against trajectories, independent of the velocity cross product."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import assert_allclose

from elisa import const, settings, utils
from elisa.base.container import StarContainer
from elisa.base.curves.rv_point import _calculate_rv_point
from elisa.base.spot import Spot
from elisa.binary_system.container import OrbitalPositionContainer
from elisa.binary_system.surface.faces import build_velocities as build_binary_velocities
from elisa.single_system.container import SinglePositionContainer
from elisa.single_system.orbit.orbit import Orbit, angular_velocity
from elisa.single_system.utils import move_sys_onpos
from tests import utils as testutils

if TYPE_CHECKING:
    from numpy.typing import NDArray


# Central differences in surface coordinates in metres need an absolute tolerance
# for components near zero, where relative error is not meaningful.
VELOCITY_RTOL = 2e-9
VELOCITY_ATOL = 2e-6  # m/s


def _position_at_phase(phase: float) -> const.Position:
    """Use the production phase-to-azimuth convention without building a mesh."""
    azimuth, _, phase = Orbit.rotational_motion(phase)[0]
    return const.Position(0, np.nan, azimuth, np.nan, phase)


def _surface_patch(points: NDArray[np.float64]) -> StarContainer:
    """Create complete triangular faces with deterministic surface properties."""
    points = np.asarray(points, dtype=np.float64)
    faces = np.arange(len(points)).reshape(-1, 3)
    centres = points[faces].mean(axis=1)
    count = len(faces)
    return StarContainer(
        points=points,
        faces=faces,
        face_centres=centres,
        normals=centres / np.linalg.norm(centres, axis=1)[:, None],
        temperatures=np.ones(count),
        log_g=np.ones(count),
        areas=np.ones(count),
    )


def _single_system(inclination: float, period: float = 2.3) -> SinglePositionContainer:
    # Deliberately asymmetric triangles at several latitudes and longitudes.
    star = _surface_patch(
        np.array(
            [
                [1, 0.2, 0.4],
                [0.8, 0.5, 0.3],
                [0.9, 0.3, 0.7],
                [-0.6, -0.7, 0.2],
                [-0.8, -0.4, 0.5],
                [-0.7, -0.6, 0.4],
            ],
        )
        * 1e8,
    )
    spot_points = np.array([[0.2, -0.9, 0.3], [0.4, -0.8, 0.5], [0.3, -0.7, 0.6]]) * 1e8
    spot_faces = np.array([[0, 1, 2]], dtype=np.int64)
    spot_centres = np.mean(spot_points[spot_faces], axis=1)
    spot = Spot(longitude=290.0, latitude=60.0, angular_radius=10.0, temperature_factor=0.8)
    spot.points = spot_points
    spot.faces = spot_faces
    spot.normals = spot_centres / np.linalg.norm(spot_centres, axis=1)[:, None]
    spot.face_centres = spot_centres
    spot.temperatures = np.ones(1)
    spot.log_g = np.ones(1)
    spot.areas = np.ones(1)
    star.spots[0] = spot
    return SinglePositionContainer(
        star,
        _position_at_phase(0),
        rotation_period=period,
        angular_velocity=angular_velocity(period),
        inclination=inclination,
        gamma=0,
    )


def _geometry_velocity(
    model: SinglePositionContainer,
    phase: float,
    phase_step: float = 1e-6,
) -> NDArray[np.float64]:
    """Differentiate observer-frame face positions using actual elapsed seconds."""

    def sample(sample_phase: float) -> tuple[NDArray[np.float64], float]:
        snapshot = model.copy()
        snapshot.set_on_position_params(_position_at_phase(sample_phase))
        time = float(snapshot.set_time())
        snapshot.flat_it()
        snapshot.apply_rotation()
        assert snapshot.star.points is not None
        assert snapshot.star.faces is not None
        centres = np.mean(snapshot.star.points[snapshot.star.faces], axis=1)
        return centres, time

    before, time_before = sample(phase - phase_step)
    after, time_after = sample(phase + phase_step)
    return (after - before) / (time_after - time_before)


@pytest.mark.parametrize("inclination", [0.0, 0.4, np.pi / 2, 2.2])
@pytest.mark.parametrize("phase", [-0.23, 0.0, 0.17, 0.63])
def test_star_and_off_axis_spot_follow_geometry(inclination: float, phase: float) -> None:
    model = _single_system(inclination)
    model.build_velocities()
    numerical = _geometry_velocity(model, phase)
    observed = move_sys_onpos(model, _position_at_phase(phase))
    # Flattening includes the spot as the last face before applying the same rotation.
    assert observed.star.velocities is not None
    assert len(observed.star.velocities) == 3
    assert_allclose(observed.star.velocities, numerical, rtol=VELOCITY_RTOL, atol=VELOCITY_ATOL)
    for surface, expected in ((model.star, numerical[:2]), (model.star.spots[0], numerical[2:])):
        assert surface.velocities is not None
        assert_allclose(
            utils.rotate_item(surface.velocities, _position_at_phase(phase), inclination),
            expected,
            rtol=VELOCITY_RTOL,
            atol=VELOCITY_ATOL,
        )


def test_analytic_local_rotation_and_observer_basis() -> None:
    model = _single_system(np.pi / 2)
    radius = 1e8
    # A nondegenerate triangle with centroid (R,0,0).
    model.star = _surface_patch(np.array([[radius, -1e6, -1e6], [radius, 1e6, -1e6], [radius, 0, 2e6]]))
    model.build_velocities()
    expected = np.array([[0.0, angular_velocity(model.rotation_period) * radius, 0.0]])
    assert model.star.velocities is not None
    assert_allclose(model.star.velocities, expected, atol=1e-10)
    # Spherical coordinates establish the usual right-handed local x,y,z axes.
    basis = utils.spherical_to_cartesian(np.array([[1, 0, np.pi / 2], [1, np.pi / 2, np.pi / 2], [1, 0, 0]]))
    assert_allclose(basis, np.eye(3), atol=1e-15)
    # At phase zero and i=90deg, local +x -> observer +y, local +y -> observer -x.
    assert_allclose(
        utils.rotate_item(basis, _position_at_phase(0), np.pi / 2),
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
        atol=1e-15,
    )


@pytest.mark.parametrize("inclination", [0.4, np.pi / 2, 2.2])
@pytest.mark.parametrize("phase", [-0.23, 0.17, 0.63])
def test_rv_is_flux_weighted_geometry_derivative(
    inclination: float,
    phase: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "LIMB_DARKENING_LAW", "linear")
    model = _single_system(inclination)
    model.build_velocities()
    numerical = _geometry_velocity(model, phase)
    observed = move_sys_onpos(model, _position_at_phase(phase))
    star = observed.star
    # Asymmetric radiance ensures rotational RV cannot cancel by symmetry.
    radiance = np.array([1.0, 3.0, 7.0])
    monkeypatch.setattr(star, "normal_radiance", {"rv_band": radiance})
    monkeypatch.setattr(star, "ld_cfs", {"rv_band": np.zeros((3, 1))})
    star.coverage = np.ones(3)
    assert star.indices is not None
    assert star.los_cosines is not None
    assert star.velocities is not None
    indices = star.indices
    assert len(indices) > 0
    weights = radiance[indices] * star.los_cosines[indices]
    expected = np.average(numerical[indices, 0], weights=weights)
    assert_allclose(_calculate_rv_point(star), expected, rtol=VELOCITY_RTOL, atol=VELOCITY_ATOL)
    # Observer at -x: negative dx/dt approaches it; positive dx/dt recedes.
    assert_allclose(const.LINE_OF_SIGHT, [-1, 0, 0])
    for vx in (-123.0, 123.0):
        star.velocities[:] = [vx, 0, 0]
        assert_allclose(_calculate_rv_point(star), vx)
    observed.gamma = 41.0
    before = star.velocities.copy()
    observed.add_secular_velocity()
    assert_allclose(star.velocities, before + np.array([41.0, 0.0, 0.0]))


def test_zero_rotation_at_velocity_builder() -> None:
    # Period zero is invalid publicly; omega=0 is supported by the velocity builder.
    model = _single_system(0.4)
    model.angular_velocity = 0.0
    model.build_velocities()
    assert model.star.velocities is not None
    assert_allclose(model.star.velocities, 0)
    assert_allclose(model.star.spots[0].velocities, 0)


@pytest.mark.parametrize(("phase", "inclination"), [(0.17, 0.4), (0.63, np.pi / 2)])
def test_generated_stellar_and_spot_mesh(phase: float, inclination: float) -> None:
    single = testutils.prepare_single_system(
        testutils.SINGLE_SYSTEM_PARAMS["spherical"],
        spots=testutils.SPOTS_META["primary"],
    )
    model = testutils.prepare_single_system_container(single)
    model.inclination = inclination
    model.build_mesh()
    model.build_faces()
    model.build_faces_orientation()
    model.build_surface_areas()
    model.build_velocities()
    for surface in [model.star, *model.star.spots.values()]:
        assert surface.faces is not None
        count = len(surface.faces)
        surface.temperatures = np.ones(count)
        surface.log_g = np.ones(count)
    expected = _geometry_velocity(model, phase)
    observed = move_sys_onpos(model, _position_at_phase(phase))
    assert observed.star.velocities is not None
    assert_allclose(observed.star.velocities, expected, rtol=VELOCITY_RTOL, atol=VELOCITY_ATOL)


def test_binary_rotation_independently_matches_relative_trajectory(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _single_system(0.7)
    # Evaluate the binary builder with actual orbital velocities. Differencing
    # two faces removes the spatially uniform orbital contribution independently.
    model.star.mass = 2e30
    monkeypatch.setattr(model.star, "synchronicity", 1.0)
    secondary = StarContainer()
    secondary.mass = 1e30
    binary = OrbitalPositionContainer(
        primary=model.star,
        secondary=secondary,
        period=model.rotation_period,
        semi_major_axis=1.0,
        mass_ratio=0.5,
        eccentricity=0.0,
        inclination=model.inclination,
        position=const.Position(0, 1.0, 0.3, 0.3, 0.1),
    )
    build_binary_velocities(binary, 1.0, component="primary")
    expected = _geometry_velocity(model, 0.1)
    assert model.star.velocities is not None
    velocity = utils.rotate_item(model.star.velocities, _position_at_phase(0.1), model.inclination)
    assert_allclose(velocity[1] - velocity[0], expected[1] - expected[0], rtol=VELOCITY_RTOL, atol=VELOCITY_ATOL)
    spot_velocity = utils.rotate_item(model.star.spots[0].velocities, _position_at_phase(0.1), model.inclination)
    assert_allclose(spot_velocity[0] - velocity[0], expected[2] - expected[0], rtol=VELOCITY_RTOL, atol=VELOCITY_ATOL)
