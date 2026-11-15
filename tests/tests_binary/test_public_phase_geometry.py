"""Public snapshots must use the same instantaneous Roche geometry."""

from unittest.mock import patch

import numpy as np
import pytest

from elisa.binary_system import dynamic
from elisa.binary_system.container import OrbitalPositionContainer
from tests.tests_binary.test_periastron_volume import binary
from tests.utils import SPOTS_META


def assert_spot_points(actual: object, expected: object) -> None:
    for index, spot in actual.spots.items():
        np.testing.assert_array_equal(spot.points, expected.spots[index].points)


@pytest.mark.parametrize(("eccentricity", "spin", "spotted"), [
    (0.0, 1.0, False), (0.0, 1.3, False),
    (0.3, 1.0, False), (0.3, 1.3, False), (0.3, 1.3, True),
])
@pytest.mark.parametrize("phase", [0.0, 0.3, 0.5])
def test_public_geometry(eccentricity: float, spin: float, spotted: bool, phase: float) -> None:
    system = binary(eccentricity, spin)
    if spotted:
        system.primary.spots = SPOTS_META["primary"]
    inputs = (system.primary.surface_potential, system.secondary.surface_potential)
    position = system.calculate_orbital_motion(phase)[0]
    potentials = system.correct_potentials(phases=[phase])
    reference = OrbitalPositionContainer.from_binary_system(system, position)
    reference.set_on_position_params(position, potentials["primary"][0], potentials["secondary"][0])
    dynamic.assign_spot_longitudes(
        reference, dynamic.calculate_spot_longitudes(system, phase), index=None, component="all",
    )
    reference.build()
    for actual in (system.build_container(phase=phase), system.build_container(phase=phase),
                   system.build_container(time=system.t0 + phase * system.period)):
        assert actual.position.distance == pytest.approx(position.distance, abs=1e-12)
        for name in ("primary", "secondary"):
            star, expected = getattr(actual, name), getattr(reference, name)
            assert star.surface_potential == pytest.approx(expected.surface_potential, rel=1e-12)
            for field in ("points", "areas", "temperatures", "log_g"):
                # Identical deterministic build; allow only floating-point roundoff (including JD conversion).
                np.testing.assert_allclose(getattr(star, field), getattr(expected, field), rtol=1e-10, atol=1e-12)
            np.testing.assert_array_equal(star.faces, expected.faces)
            assert_spot_points(star, expected)

    original_mesh = OrbitalPositionContainer.build_mesh
    captured = []

    def capture(container: OrbitalPositionContainer, **kwargs: object) -> object:
        for name in ("primary", "secondary"):
            assert getattr(container, name).surface_potential == potentials[name][0]
        result = original_mesh(container, **kwargs)
        captured.append(container.copy())
        return result

    with patch.object(OrbitalPositionContainer, "build_mesh", capture), \
         patch.object(OrbitalPositionContainer, "build", side_effect=AssertionError("full build")), \
         patch("elisa.ld.interpolate_on_ld_grid", side_effect=AssertionError("LD tables")), \
         patch("elisa.atm.atm.NaiveInterpolatedAtm.radiance", side_effect=AssertionError("atmosphere tables")), \
         patch("elisa.binary_system.graphic.plot.graphics.binary_mesh"), \
         patch("elisa.binary_system.graphic.plot.graphics.binary_wireframe"):
        system.plot.mesh(phase=phase)
        system.plot.wireframe(phase=phase)
    expected_mesh = OrbitalPositionContainer.from_binary_system(system, position)
    expected_mesh.set_on_position_params(position, potentials["primary"][0], potentials["secondary"][0])
    dynamic.assign_spot_longitudes(
        expected_mesh, dynamic.calculate_spot_longitudes(system, phase), index=None, component="all",
    )
    expected_mesh.build_mesh()
    for snapshot in captured:
        for name in ("primary", "secondary"):
            np.testing.assert_array_equal(getattr(snapshot, name).points, getattr(expected_mesh, name).points)
            assert_spot_points(getattr(snapshot, name), getattr(expected_mesh, name))
    assert (system.primary.surface_potential, system.secondary.surface_potential) == inputs


@pytest.mark.parametrize("phase", [0.0, 0.3, 0.5])
def test_surface_full_build_matches_public_container(phase: float) -> None:
    system = binary(spin=1.3)
    system.primary.spots = SPOTS_META["primary"]
    expected = system.build_container(phase=phase)

    class GeometryCapturedError(Exception):
        pass

    def check_geometry(**kwargs: object) -> None:
        snapshot = kwargs["system"]
        for name in ("primary", "secondary"):
            star, reference = getattr(snapshot, name), getattr(expected, name)
            assert star.surface_potential == reference.surface_potential
            for field in ("points", "areas", "log_g", "temperatures", "velocities"):
                np.testing.assert_array_equal(getattr(star, field), getattr(reference, field))
            # Surface plotting normalizes triangle winding for rendering.
            np.testing.assert_array_equal(np.sort(star.faces, axis=1), np.sort(reference.faces, axis=1))
        raise GeometryCapturedError

    # Inspect the completed physical surface before table-dependent radiance preparation.
    with patch("elisa.binary_system.graphic.plot.crv_utils.prep_surface_params", side_effect=check_geometry), \
         pytest.raises(GeometryCapturedError):
        system.plot.surface(phase=phase)
