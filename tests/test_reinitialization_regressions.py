"""Single-star potentials and numerical pulsation snapshots across init()."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from elisa import SingleSystem, Star, const
from elisa import units as u
from elisa.base.error import ElisaParametersError
from elisa.binary_system.container import OrbitalPositionContainer
from elisa.pulse import container_ops
from elisa.single_system.container import SinglePositionContainer
from tests.test_reinitialization import MODES, SPOTS, make_binary, make_single


@pytest.mark.parametrize("explicit", [False, True])
def test_single_derived_potential_reinitialization(explicit: bool) -> None:
    inputs = {"mass": 1.0 * u.solMass, "t_eff": 6000 * u.K, "polar_log_g": 4.4 * u.dex(u.cm / u.s**2)}
    if explicit:
        inputs["surface_potential"] = 6.0
    system = SingleSystem(star=Star(**inputs), inclination=80 * u.deg, rotation_period=25 * u.d)
    for change in ({}, {"mass": 1.2 * u.solMass}, {"polar_log_g": 4.6 * u.dex(u.cm / u.s**2)}, {}, {}):
        inputs.update(change)
        for key, value in change.items():
            setattr(system.star, key, value)
        system.init()
        fresh = SingleSystem(star=Star(**inputs), inclination=80 * u.deg, rotation_period=25 * u.d)
        for field in ("mass", "polar_log_g", "surface_potential", "polar_radius", "equivalent_radius", "t_eff"):
            assert_allclose(getattr(system.star, field), getattr(fresh.star, field), rtol=1e-13)
        assert system.star.surface_potential < 0
        assert system.star.mass == pytest.approx(inputs["mass"].to_value(u.kg))
        assert system.inclination == pytest.approx(np.deg2rad(80))
        assert system.star.kwargs_serializer()["mass"].unit == u.kg
        assert system.star.kwargs_serializer()["polar_log_g"].unit == u.LOG_ACCELERATION_UNIT


def test_roche_potential_constructor_validation_unchanged() -> None:
    with pytest.raises(ElisaParametersError, match="Invalid surface potential"):
        Star(mass=1, t_eff=6000, polar_log_g=4.4, surface_potential=-1)
    binary = make_binary()
    binary.init()
    assert binary.primary.surface_potential == 6.0


@pytest.mark.parametrize("owner", ["single", "binary"])
def test_position_container_owns_pulsation_snapshot(owner: str) -> None:
    system = make_single() if owner == "single" else make_binary(spots=SPOTS, pulsations=MODES)
    star = system.star if owner == "single" else system.primary
    position = const.Position(0, 1.0, 0.0, 0.0, 0.0)
    container = (
        SinglePositionContainer.from_single_system(system, position)
        if owner == "single"
        else OrbitalPositionContainer.from_binary_system(system, position)
    )
    container.build() if owner == "single" else container.build(components_distance=1.0)
    snapshot = container.star if owner == "single" else container.primary
    mode = snapshot.pulsations[0]
    original_mode = star.pulsations[0]
    assert mode is not original_mode
    assert snapshot.spots[0] is not star.spots[0]
    fields = (
        *mode.ALL_KWARGS,
        "angular_frequency",
        "renorm_const",
        "radial_amplitude",
        "horizontal_amplitude",
        "point_harmonics",
        "point_harmonics_gradient",
        "complex_radial_displacement",
        "complex_displacement",
        "tilt_phi",
        "tilt_theta",
    )
    state = {field: deepcopy(getattr(mode, field)) for field in fields}
    points = snapshot.points.copy()
    mass = snapshot.mass
    scale = 1.0 if owner == "single" else container.semi_major_axis
    displacement = mode.complex_displacement.copy()
    star.mass *= 1.2
    star.spots[0].temperature_factor = 0.8
    system.init()
    assert star.pulsations[0] is original_mode
    assert snapshot.mass == mass
    assert snapshot.spots[0].temperature_factor == 0.9
    assert_array_equal(snapshot.points, points)
    for field, value in state.items():
        actual = getattr(mode, field)
        if isinstance(value, np.ndarray):
            assert_array_equal(actual, value)
        elif isinstance(value, (float, int, bool)) or value is None:
            assert actual == value
    assert mode.radial_amplitude != star.pulsations[0].radial_amplitude
    container_ops.complex_displacement(snapshot, scale=scale)
    assert_allclose(mode.complex_displacement, displacement, rtol=1e-13)


@pytest.mark.parametrize("field", ["point_harmonics_gradient", "complex_displacement", "complex_radial_displacement"])
def test_properties_serializer_copies_existing_mode_arrays(field: str) -> None:
    system = make_single()
    container = system.build_container(phase=0.0)
    # Callers may attach computed Cartesian synthesis arrays to Star modes.
    setattr(system.star.pulsations[0], field, getattr(container.star.pulsations[0], field).copy())
    properties = system.star.properties_serializer()
    mode = properties["pulsations"][0]
    assert mode is not system.star.pulsations[0]
    actual = getattr(mode, field)
    original = getattr(system.star.pulsations[0], field)
    assert not np.shares_memory(actual, original)
    actual[:] = 0
    assert np.any(original != 0)
