"""Scientific state serializers must include defaults omitted from reconstruction."""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from elisa import const
from elisa import units as u
from elisa.binary_system.container import OrbitalPositionContainer
from elisa.single_system.container import SinglePositionContainer
from tests.test_reinitialization import MODES, SPOTS, make_binary, make_single


@pytest.mark.parametrize("owner", ["single", "binary"])
def test_full_state_and_position_container_survive_reinitialization(owner: str) -> None:
    system = make_single() if owner == "single" else make_binary(spots=SPOTS, pulsations=MODES)
    fresh = make_single() if owner == "single" else make_binary(spots=SPOTS, pulsations=MODES)
    expected_keys = set(system.ALL_KWARGS) | (
        {"angular_velocity"} if owner == "single" else {"semi_major_axis", "morphology", "mass_ratio"}
    )
    position = const.Position(0, 1.0, 0.0, 0.0, 0.25)
    factory = (
        SinglePositionContainer.from_single_system if owner == "single"
        else OrbitalPositionContainer.from_binary_system
    )
    reference_container = factory(fresh, position)

    for iteration in range(4):
        if iteration:
            system.init()
        state = system.properties_serializer()
        kwargs = system.kwargs_serializer()
        container = factory(system, position)

        assert set(state) == expected_keys
        assert set(kwargs) == set(system.ALL_KWARGS)
        assert state["distance"] == 10 * const.PC
        assert state["phase_shift"] == 0.0
        assert state["additional_light"] == 0.0
        assert state["reference_time" if owner == "single" else "primary_minimum_time"] == 0.0
        assert kwargs["distance"].unit == u.m
        assert kwargs["inclination"].unit == u.rad
        assert kwargs["gamma"].unit == u.m / u.s
        assert "distance" not in system.reinitialization_kwargs()
        for key, value in state.items():
            expected = fresh.properties_serializer()[key]
            assert type(value) is type(expected)
            if isinstance(value, str):
                assert value == expected
                assert getattr(container, key) == getattr(reference_container, key)
            else:
                assert_allclose(value, expected, rtol=1e-13, equal_nan=True)
                assert_allclose(getattr(container, key), getattr(reference_container, key), rtol=1e-13, equal_nan=True)
        if owner == "single":
            assert np.isnan(state["gamma"])
            assert "gamma" not in system.reinitialization_kwargs()
        assert_allclose(container.time, reference_container.time)


@pytest.mark.parametrize("owner", ["single", "binary"])
def test_nested_state_serialization_keeps_computed_defaults(owner: str) -> None:
    system = make_single() if owner == "single" else make_binary(spots=SPOTS, pulsations=MODES)
    star = system.star if owner == "single" else system.primary

    for iteration in range(4):
        if iteration:
            system.init()
        state = star.kwargs_serializer()
        properties = star.properties_serializer()
        reconstruction = star.reinitialization_kwargs()

        assert set(state) == set(star.ALL_KWARGS)
        assert state["mass"].unit == u.kg
        assert state["t_eff"].unit == u.K
        assert state["polar_log_g"].unit == u.LOG_ACCELERATION_UNIT
        assert state["gravity_darkening"] == star.gravity_darkening
        assert properties["gravity_darkening"] == star.gravity_darkening
        assert properties["surface_potential"] == star.surface_potential
        assert "gravity_darkening" not in reconstruction
        assert set(state["spots"][0]) == set(star.spots[0].ALL_KWARGS)
        assert state["spots"][0]["discretization_factor"].unit == u.rad
        assert "discretization_factor" not in reconstruction["spots"][0]
        assert set(state["pulsations"][0]) == set(star.pulsations[0].ALL_KWARGS)
        for field in ("horizontal_to_radial_amplitude_ratio", "temperature_amplitude_factor"):
            assert state["pulsations"][0][field] == getattr(star.pulsations[0], field)
            assert field not in reconstruction["pulsations"][0]
        assert state["pulsations"][0]["frequency"].unit == u.FREQUENCY_UNIT
