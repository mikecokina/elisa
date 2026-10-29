"""Current model attributes survive constructor-based reinitialization."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from elisa import BinarySystem, SingleSystem, Star, const, units as u
from elisa import umpy as up
from elisa.pulse.utils import phase_correction
from elisa.single_system.container import SinglePositionContainer


OPTIONAL_SENTINELS = (
    "surface_potential", "synchronicity", "albedo", "polar_log_g", "gravity_darkening",
)
SPOTS = [
    dict(longitude=45.0, latitude=60.0, angular_radius=20.0, temperature_factor=0.9),
    dict(longitude=120.0, latitude=30.0, angular_radius=15.0, temperature_factor=1.1,
         discretization_factor=2.0),
]
MODES = [
    dict(l=1, m=1, amplitude=1.0, frequency=12.0, start_phase=0.75,
         mode_axis_theta=30.0, mode_axis_phi=45.0, temperature_perturbation_phase_shift=60.0),
    dict(l=2, m=-1, amplitude=2.0, frequency=16.0 / u.d, tidally_locked=True,
         horizontal_to_radial_amplitude_ratio=0.5, temperature_amplitude_factor=0.1),
]


def make_binary(*, secondary_kwargs: dict | None = None, **primary_kwargs) -> BinarySystem:
    defaults = dict(mass=1.2, t_eff=6000.0, surface_potential=6.0, synchronicity=1.0)
    primary = Star(**(defaults | primary_kwargs))
    secondary_defaults = dict(mass=0.8, t_eff=5000.0, surface_potential=6.0, synchronicity=1.0)
    secondary = Star(**(secondary_defaults | (secondary_kwargs or {})))
    return BinarySystem(primary=primary, secondary=secondary, inclination=80.0,
                        period=3.0, eccentricity=0.1, argument_of_periastron=45.0)


def make_single(**system_kwargs) -> SingleSystem:
    star = Star(mass=1.0, t_eff=6000.0, polar_log_g=4.4, spots=SPOTS, pulsations=MODES)
    return SingleSystem(star=star, inclination=80.0, rotation_period=25.0, **system_kwargs)


def spot_state(star: Star) -> list[list[float]]:
    return [[getattr(spot, key) for key in spot.ALL_KWARGS] for spot in star.spots.values()]


def mode_state(star: Star) -> list[list[float | None]]:
    return [[getattr(mode, key) for key in mode.ALL_KWARGS] for mode in star.pulsations.values()]


def binary_state(system: BinarySystem) -> list[float]:
    state = [getattr(system, key) for key in system.ALL_KWARGS]
    state += [system.semi_major_axis, system.mass_ratio]
    for star in system.components.values():
        state += [star.mass, star.t_eff, star.surface_potential, star.synchronicity,
                  star.polar_log_g, star.albedo, star.gravity_darkening, star.discretization_factor,
                  star.polar_radius, star.equatorial_radius, star.equivalent_radius]
    return state


def test_star_omitted_optional_sentinels_survive_repeated_init() -> None:
    star = Star(mass=1.0, t_eff=6000.0)
    for _ in range(3):
        serialized = star.kwargs_serializer()
        kwargs = star.reinitialization_kwargs()
        assert set(serialized) == set(star.ALL_KWARGS)
        for key in OPTIONAL_SENTINELS:
            assert key not in kwargs
            assert key not in star.kwargs
            assert getattr(star, key) is up.NaN
        star.init()
        assert star.mass == u.solMass.to(u.kg)
        assert star.t_eff == 6000.0
    # The numerical sentinel still selects the existing pulsation phase rule.
    assert phase_correction(0.25, star.synchronicity) == np.pi / 2


@pytest.mark.parametrize("key", Star.MANDATORY_KWARGS)
def test_missing_mandatory_star_parameters_are_not_filtered(key: str) -> None:
    values = dict(mass=1.0, t_eff=6000.0)
    del values[key]
    star = Star(**values)
    assert key in star.reinitialization_kwargs()
    with pytest.raises(FloatingPointError):
        star.init()


@pytest.mark.parametrize("explicit", [False, True])
def test_repeated_star_init_preserves_current_finite_parameters(explicit: bool) -> None:
    optional = dict(surface_potential=6.0, synchronicity=1.0, albedo=0.5,
                    gravity_darkening=0.32, polar_log_g=4.4)
    star = Star(mass=1.0, t_eff=6000.0, **(optional if explicit else {}))
    # Direct attribute changes use internal units, rather than original kwargs.
    changed = dict(mass=1.1 * u.solMass.to(u.kg), t_eff=6200.0, surface_potential=7.0,
                   synchronicity=1.2, albedo=0.0, gravity_darkening=0.0,
                   polar_log_g=2.6, discretization_factor=np.radians(7.0))
    for key, value in changed.items():
        setattr(star, key, value)
    for _ in range(4):
        star.init()
        for key, expected in changed.items():
            assert_allclose(getattr(star, key), expected, rtol=1e-14, atol=0)


@pytest.mark.parametrize("value", [
    4.4, np.float64(4.4), 4.4 * u.dex(u.cm / u.s**2), 2.4 * u.dex(u.m / u.s**2),
])
def test_polar_log_g_preserves_internal_logarithmic_unit(value) -> None:
    star = Star(mass=1.0, t_eff=6000.0, polar_log_g=value)
    expected = star.polar_log_g
    for _ in range(4):
        serialized = star.reinitialization_kwargs()["polar_log_g"]
        assert isinstance(serialized, u.Dex)
        assert serialized.unit == u.LOG_ACCELERATION_UNIT
        star.init()
        assert isinstance(star.polar_log_g, float)
        assert_allclose(star.polar_log_g, expected, rtol=0, atol=1e-14)


@pytest.mark.parametrize("explicit_resolution", [False, True])
def test_binary_repeated_init_preserves_defaults_and_physical_state(explicit_resolution: bool) -> None:
    resolution = dict(discretization_factor=5.0) if explicit_resolution else {}
    system = make_binary(secondary_kwargs=resolution, **resolution)
    assert np.isnan(system.primary.polar_log_g)
    defaults = [system.primary.albedo, system.primary.gravity_darkening,
                system.secondary.albedo, system.secondary.gravity_darkening]
    assert np.all(np.isfinite(defaults))
    if not explicit_resolution:
        # Construction adjusts the secondary's mesh factor after computing radii.
        # The first init recomputes mesh-dependent radii at that adjusted resolution.
        system.init()
    expected = binary_state(system)
    for _ in range(4):
        system.init()
        assert_allclose(binary_state(system), expected, rtol=1e-13, atol=0, equal_nan=True)
        assert "polar_log_g" not in system.primary.kwargs
        assert "polar_log_g" not in system.secondary.kwargs
        assert_allclose([system.primary.albedo, system.primary.gravity_darkening,
                         system.secondary.albedo, system.secondary.gravity_darkening], defaults)


def test_binary_init_uses_modified_stellar_discretization_and_recomputes_properties() -> None:
    system = make_binary(polar_log_g=4.4)
    old_axis = system.semi_major_axis
    old_log_g = system.primary.polar_log_g
    system.primary.discretization_factor = np.radians(7.0)
    system.primary.mass *= 1.1
    system.period *= 1.2
    system.init()
    assert system.semi_major_axis > old_axis
    assert_allclose(system.mass_ratio, system.secondary.mass / system.primary.mass)
    expected = binary_state(system)
    for _ in range(3):
        system.init()
        assert system.primary.discretization_factor == np.radians(7.0)
        assert_allclose(system.primary.polar_log_g, old_log_g, rtol=0, atol=1e-14)
        assert_allclose(binary_state(system), expected, rtol=1e-13, atol=0, equal_nan=True)


@pytest.mark.parametrize("owner", ["star", "binary", "single"])
def test_spots_survive_reconstruction_without_accumulation_or_angular_drift(owner: str) -> None:
    if owner == "star":
        model = star = Star(mass=1.0, t_eff=6000.0, spots=SPOTS)
    elif owner == "binary":
        model = make_binary(spots=SPOTS)
        star = model.primary
    else:
        model = make_single()
        star = model.star
    star.spots[0].longitude = np.radians(75.0)
    star.spots[1].temperature_factor = 0.95
    expected = spot_state(star)
    for _ in range(3):
        model.init()
        assert len(star.spots) == 2
        assert_allclose(spot_state(star), expected, rtol=1e-14, atol=0)


def test_standalone_pulsations_reconstruct_current_descriptors_with_internal_units() -> None:
    star = Star(mass=1.0, t_eff=6000.0, pulsations=MODES)
    star.pulsations[0].frequency = 20.0 / 86400
    star.pulsations[0].mode_axis_phi = np.radians(75.0)
    star.pulsations[0].amplitude = 3.0
    expected = mode_state(star)
    for _ in range(3):
        kwargs = star.reinitialization_kwargs()
        assert isinstance(kwargs["pulsations"], list)
        assert kwargs["pulsations"][0]["frequency"].unit == u.FREQUENCY_UNIT
        assert "horizontal_to_radial_amplitude_ratio" not in kwargs["pulsations"][0]
        star.init()
        assert len(star.pulsations) == 2
        for actual, original in zip(mode_state(star), expected, strict=True):
            for value, reference in zip(actual, original, strict=True):
                if reference is None:
                    assert value is None
                else:
                    assert_allclose(value, reference, rtol=1e-14, atol=0)
        assert_allclose(star.pulsations[0].angular_frequency, 2 * np.pi * 20 / 86400)


@pytest.mark.parametrize("owner", ["binary", "single"])
def test_system_pulsations_preserve_calculated_amplitudes_and_current_configuration(owner: str) -> None:
    system = make_binary(pulsations=MODES) if owner == "binary" else make_single()
    star = system.primary if owner == "binary" else system.star
    star.pulsations[0].frequency = 20.0 / 86400
    star.pulsations[0].amplitude = 3.0
    expected = mode_state(star)
    for _ in range(3):
        system.init()
        assert len(star.pulsations) == 2
        assert_allclose(mode_state(star), expected, rtol=1e-14, atol=0)
        mode = star.pulsations[0]
        assert_allclose(mode.angular_frequency, 2 * np.pi * mode.frequency)
        assert_allclose(mode.radial_amplitude,
                        mode.amplitude / mode.angular_frequency
                        / np.sqrt(1 + mode.horizontal_to_radial_amplitude_ratio**2))


@pytest.mark.parametrize("owner", ["star", "binary", "single"])
def test_replacing_pulsation_descriptors_remains_effective_after_init(owner: str) -> None:
    if owner == "star":
        model = star = Star(mass=1.0, t_eff=6000.0, pulsations=MODES)
    elif owner == "binary":
        model = make_binary(pulsations=MODES)
        star = model.primary
    else:
        model = make_single()
        star = model.star
    star.pulsations = [dict(l=0, m=0, amplitude=3.0, frequency=20.0,
                           temperature_amplitude_factor=0.1)]
    for _ in range(3):
        model.init()
        assert list(star.pulsations) == [0]
        mode = star.pulsations[0]
        assert mode.l == 0
        assert mode.m == 0
        assert mode.amplitude == 3.0
        assert_allclose(mode.frequency, 20.0 / 86400, rtol=1e-14)
        if owner != "star":
            assert np.isfinite(mode.radial_amplitude)
    star.pulsations = []
    model.init()
    assert not star.has_pulsations()


def test_single_omitted_gamma_and_current_stellar_features_survive_repeated_init() -> None:
    system = make_single()
    star = system.star
    expected = [star.polar_log_g, star.surface_potential, star.polar_radius,
                star.equatorial_radius, star.equivalent_radius, star.gravity_darkening,
                system.angular_velocity]
    modes = mode_state(star)
    for _ in range(3):
        assert "gamma" not in system.reinitialization_kwargs()
        assert np.isnan(system.kwargs_serializer()["gamma"].value)
        system.init()
        assert system.star is star
        assert system.gamma is up.NaN
        assert_allclose([star.polar_log_g, star.surface_potential, star.polar_radius,
                         star.equatorial_radius, star.equivalent_radius, star.gravity_darkening,
                         system.angular_velocity], expected, rtol=1e-13, atol=0)
        assert_allclose(mode_state(star), modes, rtol=1e-14, atol=0)
        props = system.properties_serializer()
        assert props["gamma"] is up.NaN
        container = SinglePositionContainer.from_single_system(
            system, const.Position(0, np.nan, 0.0, np.nan, 0.0),
        )
        assert np.isnan(container.gamma)


@pytest.mark.parametrize("explicit", [False, True])
def test_single_current_finite_gamma_is_preserved(explicit: bool) -> None:
    system = make_single(**({"gamma": -1000.0} if explicit else {}))
    if not explicit:
        system.gamma = 1234.0
    expected = system.gamma
    for _ in range(3):
        system.init()
        assert system.gamma == expected
        assert system.properties_serializer()["gamma"] == expected


@pytest.mark.parametrize("key", OPTIONAL_SENTINELS + Star.MANDATORY_KWARGS)
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_explicit_nonfinite_stellar_inputs_still_fail(key: str, invalid: float) -> None:
    kwargs = dict(mass=1.0, t_eff=6000.0) | {key: invalid}
    with pytest.raises(FloatingPointError):
        Star(**kwargs)


@pytest.mark.parametrize("key", OPTIONAL_SENTINELS)
@pytest.mark.parametrize("invalid", [float("nan"), np.float64(np.nan), np.inf, -np.inf])
def test_nonfinite_current_values_are_not_mistaken_for_omitted_sentinels(key: str, invalid: float) -> None:
    star = Star(mass=1.0, t_eff=6000.0)
    setattr(star, key, invalid)
    assert key in star.reinitialization_kwargs()
    with pytest.raises(FloatingPointError):
        star.init()


@pytest.mark.parametrize("key", OPTIONAL_SENTINELS)
def test_nan_arithmetic_results_are_not_discarded(key: str) -> None:
    star = Star(mass=1.0, t_eff=6000.0)
    with np.errstate(invalid="ignore"):
        setattr(star, key, np.sqrt(-1.0))
    with pytest.raises(FloatingPointError):
        star.init()


@pytest.mark.parametrize("key", OPTIONAL_SENTINELS)
def test_explicit_finite_star_value_later_set_to_nan_is_not_discarded(key: str) -> None:
    valid = dict(surface_potential=6.0, synchronicity=1.0, albedo=0.5,
                 polar_log_g=4.4, gravity_darkening=0.32)
    star = Star(mass=1.0, t_eff=6000.0, **{key: valid[key]})
    setattr(star, key, np.nan)
    with pytest.raises(FloatingPointError):
        star.init()


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_single_explicit_invalid_gamma_is_rejected(invalid: float) -> None:
    with pytest.raises(FloatingPointError):
        make_single(gamma=invalid)


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("invalid", [float("nan"), np.float64(np.nan), np.inf, -np.inf])
def test_single_invalid_current_gamma_reaches_validation(explicit: bool, invalid: float) -> None:
    system = make_single(**({"gamma": 0.0} if explicit else {}))
    system.gamma = invalid
    assert "gamma" in system.reinitialization_kwargs()
    with pytest.raises(FloatingPointError):
        system.properties_serializer()
    with pytest.raises(FloatingPointError):
        system.init()


def test_single_quantity_gamma_is_transformed_before_properties_serialization() -> None:
    system = make_single()
    system.gamma = 2.0 * u.km / u.s
    assert system.properties_serializer()["gamma"] == 2000.0
    system.init()
    assert system.gamma == 2000.0


@pytest.mark.parametrize("key", OPTIONAL_SENTINELS)
def test_new_finite_star_parameters_acquire_provenance_after_reconstruction(key: str) -> None:
    star = Star(mass=1.0, t_eff=6000.0)
    star.surface_potential = 6.0
    star.synchronicity = 1.0
    star.albedo = 0.5
    star.polar_log_g = 2.4
    star.gravity_darkening = 0.32
    star.init()
    assert key in star.kwargs
    setattr(star, key, np.nan)
    with pytest.raises(FloatingPointError):
        star.init()


def test_new_finite_single_gamma_acquires_provenance_after_reconstruction() -> None:
    system = make_single()
    system.gamma = 0.0
    system.init()
    system.gamma = np.nan
    with pytest.raises(FloatingPointError):
        system.init()
