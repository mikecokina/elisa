"""Explicit inputs survive rebuilds while parent-owned defaults are refreshed."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose

from elisa import BinarySystem, SingleSystem, Star, settings
from elisa.base.error import ElisaInvalidKwargsError, ElisaParametersError
from elisa.observer.observer import Observer
from elisa.pulse.mode import PulsationMode
from tests.test_reinitialization import MODES, SPOTS, make_binary, make_single


@pytest.mark.parametrize("owner", ["binary", "single"])
@pytest.mark.parametrize("override", [False, True])
def test_temperature_defaults_follow_current_inputs(owner: str, override: bool) -> None:
    system = make_binary() if owner == "binary" else make_single()
    star = system.primary if owner == "binary" else system.star
    old_beta = star.gravity_darkening
    old_albedo = star.albedo
    if override:
        star.gravity_darkening = old_beta
        if owner == "binary":
            star.albedo = old_albedo
    star.t_eff = 8500.0

    for _ in range(3):
        system.init()

        assert star.t_eff == 8500.0
        assert (star.gravity_darkening == old_beta) == override
        if owner == "binary":
            assert (star.albedo == old_albedo) == override
        assert ("gravity_darkening" in star.reinitialization_kwargs()) == override


def test_external_constructor_replaces_provenance() -> None:
    star = Star(mass=1.0, t_eff=6000.0, albedo=0.7)
    star.init()
    assert star.albedo == 0.7

    star.__init__(mass=1.0, t_eff=7000.0)
    star.init()

    assert np.isnan(star.albedo)
    assert "albedo" not in star.reinitialization_kwargs()


def test_nested_objects_keep_identity_and_parameter_edits() -> None:
    binary = make_binary(spots=SPOTS, pulsations=MODES)
    star = binary.primary
    spot, mode = star.spots[0], star.pulsations[0]
    spot.longitude = 0.7
    mode.frequency *= 2
    star.discretization_factor = np.radians(6.0)

    for _ in range(3):
        binary.init()

        assert star.spots[0] is spot
        assert star.pulsations[0] is mode
        assert list(star.spots) == [0, 1]
        assert list(star.pulsations) == [0, 1]
        assert spot.longitude == pytest.approx(0.7)
        assert spot.discretization_factor == star.discretization_factor
        assert star.spots[1].discretization_factor == pytest.approx(np.radians(2.0))
        assert star.pulsations[1].horizontal_to_radial_amplitude_ratio == 0.5
        assert star.pulsations[1].temperature_amplitude_factor == 0.1


def test_standalone_nested_init_resets_parent_defaults() -> None:
    binary = make_binary(spots=SPOTS, pulsations=MODES)
    spot, mode = binary.primary.spots[0], binary.primary.pulsations[0]
    spot.temperature_factor = 0.8
    mode.mode_axis_phi = 0.3

    spot.init()
    mode.init()

    assert spot.temperature_factor == 0.8
    assert np.isnan(spot.discretization_factor)
    assert mode.mode_axis_phi == pytest.approx(0.3)
    assert mode.horizontal_to_radial_amplitude_ratio is None
    assert mode.temperature_amplitude_factor is None
    binary.init()
    assert np.isfinite(spot.discretization_factor)
    assert mode.horizontal_to_radial_amplitude_ratio is not None


def test_failure_restores_tracking_and_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    binary = make_binary()
    setup = binary.setup_albedos

    def fail() -> None:
        raise RuntimeError("interrupted parent initialization")

    monkeypatch.setattr(binary, "setup_albedos", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        binary.init()
    binary.primary.albedo = 0.7
    binary.primary.t_eff = 8500.0
    monkeypatch.setattr(binary, "setup_albedos", setup)

    binary.init()
    binary.init()

    assert binary.primary.albedo == 0.7
    assert "gravity_darkening" not in binary.primary.reinitialization_kwargs()


@pytest.mark.parametrize("invalid", [None, np.nan])
def test_explicit_invalid_mode_factor_is_not_omitted(invalid: float | None) -> None:
    mode = PulsationMode(**MODES[0])
    mode.horizontal_to_radial_amplitude_ratio = invalid

    with pytest.raises(ElisaParametersError):
        mode.init()

    mode.horizontal_to_radial_amplitude_ratio = 0.5
    mode.init()
    assert mode.horizontal_to_radial_amplitude_ratio == 0.5


def test_component_discretization_matches_fresh_system() -> None:
    binary = make_binary()
    binary.primary.t_eff = 8500.0
    reference = make_binary(t_eff=8500.0)

    for _ in range(3):
        binary.init()

        assert_allclose(
            [binary.primary.discretization_factor, binary.secondary.discretization_factor],
            [reference.primary.discretization_factor, reference.secondary.discretization_factor],
        )


def test_single_mass_and_gravity_match_fresh_constructor() -> None:
    single = make_single()
    single.star.mass *= 1.2
    single.star.polar_log_g = 2.6
    reference = make_single()
    reference.star.__init__(mass=1.2, t_eff=6000.0, polar_log_g=4.6, spots=SPOTS, pulsations=deepcopy(MODES))
    reference.init()

    for _ in range(3):
        single.init()

        assert single.star.surface_potential == pytest.approx(reference.star.surface_potential)
        assert single.star.polar_log_g == pytest.approx(2.6)
        assert "surface_potential" not in single.star.reinitialization_kwargs()


def test_callable_setup_helpers_do_not_promote_defaults() -> None:
    binary = make_binary(spots=SPOTS, pulsations=MODES)

    binary.setup_albedos()
    binary.setup_betas()
    binary.setup_discretisation_factor()
    binary.assign_pulsations_amplitudes(binary.semi_major_axis)

    kwargs = binary.primary.reinitialization_kwargs()
    assert "albedo" not in kwargs
    assert "gravity_darkening" not in kwargs
    assert "discretization_factor" not in kwargs
    assert "discretization_factor" not in kwargs["spots"][0]
    assert "temperature_amplitude_factor" not in kwargs["pulsations"][0]


def test_failed_star_rebuild_keeps_nested_inputs_for_retry() -> None:
    binary = make_binary(spots=SPOTS, pulsations=MODES)
    star = binary.primary
    spots, modes = star.spots, star.pulsations
    star.albedo = np.nan

    with pytest.raises(ElisaParametersError):
        binary.init()
    star.albedo = 0.7
    binary.init()

    assert star.spots is spots
    assert star.pulsations is modes
    assert star.albedo == 0.7


def test_failed_descriptor_assignment_does_not_mark_explicit() -> None:
    star = Star(mass=1.0, t_eff=6000.0)

    with pytest.raises(ElisaInvalidKwargsError):
        star.spots = [{"unexpected": 1}]

    assert "spots" not in star.reinitialization_kwargs()


def test_json_canonical_inputs_and_serializer_contract() -> None:
    binary = make_binary(spots=SPOTS, pulsations=MODES)
    serialized = binary.primary.kwargs_serializer()
    data = binary.to_json()

    rebuilt = type(binary).from_json(data)
    rebuilt.init()

    assert set(serialized) == set(Star.ALL_KWARGS)
    assert "discretization_factor" in serialized["spots"][0]
    assert "horizontal_to_radial_amplitude_ratio" in serialized["pulsations"][0]
    assert rebuilt.primary.mass == pytest.approx(binary.primary.mass)
    assert rebuilt.primary.albedo == binary.primary.albedo
    assert "albedo" in rebuilt.primary.reinitialization_kwargs()


@pytest.mark.parametrize("owner", ["binary", "single"])
@pytest.mark.parametrize("replace_collections", [False, True])
def test_reinitialized_light_curve_matches_fresh_model(owner: str, replace_collections: bool) -> None:
    """Exercise real surface generation with valid descriptor-assigned collections."""
    spots, modes = deepcopy(SPOTS), deepcopy(MODES)
    spots[0]["longitude"] = np.degrees(0.7)
    modes[0]["frequency"] = 24.0

    def construct(spot_inputs: list, mode_inputs: list, temperature: float) -> BinarySystem | SingleSystem:
        params = {
            "spots": spot_inputs, "pulsations": mode_inputs, "t_eff": temperature,
            "atmosphere": "black_body", "limb_darkening_coefficients": {"bolometric": [0.5]},
        }
        if owner == "binary":
            return make_binary(
                **params,
                secondary_kwargs={"atmosphere": "black_body", "limb_darkening_coefficients": {"bolometric": [0.5]}},
            )
        return SingleSystem(
            star=Star(mass=1.0, polar_log_g=4.4, **params), inclination=80.0, rotation_period=25.0,
        )

    with settings.override(LIMB_DARKENING_LAW="linear", NUMBER_OF_PROCESSES=1):
        system = construct(deepcopy(SPOTS), deepcopy(MODES), 6000.0)
        star = system.primary if owner == "binary" else system.star
        star.t_eff = 6500.0
        if replace_collections:
            # Public setters enumerate and validate descriptors, including additions/removals.
            star.spots = spots[:1]
            star.pulsations = modes[:1]
            spots, modes = spots[:1], modes[:1]
        else:
            star.spots[0].longitude = 0.7
            star.pulsations[0].frequency = 24.0 / 86400
        fresh = construct(spots, modes, 6500.0)
        expected_star = fresh.primary if owner == "binary" else fresh.star
        phases = np.array([0.05, 0.23, 0.47])
        expected = Observer(passband=["bolometric"], system=fresh).lc(phases=phases)[1]["bolometric"]

        for _ in range(3):
            system.init()
            actual = Observer(passband=["bolometric"], system=system).lc(phases=phases)[1]["bolometric"]

            assert np.all(np.isfinite(actual))
            assert_allclose(actual, expected, rtol=1e-10, atol=0)
            for field in ("mass", "surface_potential", "gravity_darkening", "discretization_factor"):
                assert_allclose(getattr(star, field), getattr(expected_star, field), rtol=1e-13)
            for index, mode in star.pulsations.items():
                reference = expected_star.pulsations[index]
                for field in ("frequency", "horizontal_to_radial_amplitude_ratio", "temperature_amplitude_factor"):
                    assert_allclose(getattr(mode, field), getattr(reference, field), rtol=1e-13)
            assert list(star.spots) == list(range(len(spots)))
            assert list(star.pulsations) == list(range(len(modes)))
