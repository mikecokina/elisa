"""Verify passband preparation against independent hemispheric integrals."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias
from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from numpy.typing import NDArray

from elisa import BinarySystem, Observer, Star, ld, settings
from elisa.binary_system.curves import utils

if TYPE_CHECKING:
    from typing import Any

    from elisa.binary_system.container import OrbitalPositionContainer


FloatArray: TypeAlias = NDArray[np.float64]
PassbandValues: TypeAlias = dict[str, FloatArray]
ComponentValues: TypeAlias = dict[str, PassbandValues]
SurfaceParameters: TypeAlias = tuple[ComponentValues, ComponentValues]
SurfacePreparation: TypeAlias = Callable[..., SurfaceParameters | None]
LimbDarkeningSample: TypeAlias = tuple[str, str, FloatArray]

BANDS = ["Generic.Bessell.V", "Generic.Bessell.B"]


@dataclass(frozen=True)
class CapturedScience:
    """Retain the observed curve and the LD arrays used to synthesize it."""

    phases: FloatArray
    fluxes: PassbandValues
    limb_darkening_samples: list[LimbDarkeningSample]


def _reference_surface_preparation(
    system: OrbitalPositionContainer,
    *,
    return_values: bool = True,
    write_to_containers: bool = False,
    **kwargs: Any,
) -> SurfaceParameters | None:
    """Normalize surface flux using analytical passband hemispheric integrals."""
    ld_coefficients = utils.get_limbdarkening_cfs(system, **kwargs)
    # Reuse atmosphere interpolation; independently check the angular normalization.
    # noinspection PyProtectedMember
    normal_radiance = utils._get_normal_radiance(system, **kwargs)
    for component, passbands in normal_radiance.items():
        for band, flux in passbands.items():
            coefficients = ld_coefficients[component][band]
            hemisphere = 1 - coefficients[:, 0] / 3
            if settings.LIMB_DARKENING_LAW == "logarithmic":
                hemisphere += 2 * coefficients[:, 1] / 9
            elif settings.LIMB_DARKENING_LAW == "square_root":
                hemisphere -= coefficients[:, 1] / 5
            passbands[band] = flux / (np.pi * hemisphere)
    if write_to_containers:
        for component in settings.BINARY_COUNTERPARTS:
            star = getattr(system, component)
            star.normal_radiance = normal_radiance[component]
            star.ld_cfs = ld_coefficients[component]

    if return_values:
        return normal_radiance, ld_coefficients
    return None


def _make_observer(case: str, bands: list[str], law: str, *, custom_coefficients: bool) -> Observer:
    coefficients = [0.5] if law in ("linear", "cosine") else [0.5, 0.2]
    star_properties: dict[str, object] = {
        "surface_potential": 5.0,
        "t_eff": 7000.0,
        "gravity_darkening": 1.0,
        "discretization_factor": 10.0,
        "albedo": 0.6,
        "metallicity": 0.0,
        "atmosphere": "bb",
    }
    if custom_coefficients:
        star_properties["limb_darkening_coefficients"] = dict.fromkeys([*BANDS, "bolometric"], coefficients)

    primary = Star(mass=2.0, synchronicity=0.8 if "async" in case else 1.0, **star_properties)
    secondary = Star(mass=1.0, synchronicity=1.2 if "async" in case else 1.0, **star_properties)
    if "spot" in case:
        primary.spots = [{"longitude": 90.0, "latitude": 58.0, "angular_radius": 25.0, "temperature_factor": 0.95}]
    if case == "pulsating":
        primary.pulsations = [{"l": 2, "m": 1, "amplitude": 50.0, "frequency": 16.0, "start_phase": 0.0}]

    binary = BinarySystem(
        primary=primary,
        secondary=secondary,
        argument_of_periastron=58.0,
        gamma=0.0,
        period=2.5,
        eccentricity=0.2 if "ecc" in case else 0.0,
        inclination=85.0,
        primary_minimum_time=0.0,
        phase_shift=0.0,
    )
    return Observer(passband=bands, system=binary)


def _capture_science(preparation: SurfacePreparation, observer: Observer, *, normalize: bool) -> CapturedScience:
    """Record real LD arrays from the observed-band and bolometric stages."""
    samples: list[LimbDarkeningSample] = []
    calculate_coefficients = utils.get_limbdarkening_cfs

    def capture_coefficients(system: OrbitalPositionContainer, *args: object, **kwargs: Any) -> ComponentValues:
        component_coefficients = calculate_coefficients(system, *args, **kwargs)
        for component, passbands in component_coefficients.items():
            temperatures = getattr(system, component).temperatures
            for passband, coefficients in passbands.items():
                assert coefficients.shape[0] == len(temperatures)
                samples.append((component, passband, coefficients.copy()))
        return component_coefficients

    ld.clear_ld_caches()
    with (
        patch.object(utils, "prep_surface_params", preparation),
        patch.object(utils, "get_limbdarkening_cfs", capture_coefficients),
    ):
        phases, fluxes = observer.observe.lc(phases=np.array([0.0, 0.13, 0.27, 0.49, 0.71]), normalize=normalize)

    return CapturedScience(phases, fluxes, samples)


def _assert_matching_science(expected: CapturedScience, actual: CapturedScience, bands: list[str]) -> None:
    assert_array_equal(expected.phases, actual.phases, strict=True)
    assert list(expected.fluxes) == bands
    assert list(actual.fluxes) == bands
    for band in bands:
        # Independent analytical arithmetic may differ by floating-point roundoff.
        assert_allclose(expected.fluxes[band], actual.fluxes[band], rtol=1e-14, atol=0)

    assert expected.limb_darkening_samples
    assert len(expected.limb_darkening_samples) == len(actual.limb_darkening_samples)
    for expected_sample, actual_sample in zip(
        expected.limb_darkening_samples,
        actual.limb_darkening_samples,
        strict=True,
    ):
        expected_component, expected_band, expected_coefficients = expected_sample
        actual_component, actual_band, actual_coefficients = actual_sample
        assert (expected_component, expected_band) == (actual_component, actual_band)
        assert_array_equal(expected_coefficients, actual_coefficients, strict=True)


@pytest.mark.parametrize("law", ["linear", "cosine", "logarithmic", "square_root"])
@pytest.mark.parametrize(
    "custom",
    [
        pytest.param(
            False,
            marks=pytest.mark.requires_ld(passbands=(*BANDS, "bolometric"), source="van_hamme"),
            id="tables",
        ),
        pytest.param(True, id="custom"),
    ],
)
@pytest.mark.parametrize(
    ("case", "bands"),
    [
        ("circular", BANDS),
        ("circular", ["bolometric", *BANDS]),
        ("circular_async_spot", BANDS),
        ("eccentric", BANDS),
        ("eccentric_async_spot", BANDS),
        ("pulsating", BANDS),
    ],
)
def test_passband_scientific_normalization(*, law: str, custom: bool, case: str, bands: list[str]) -> None:
    with settings.override(
        LIMB_DARKENING_LAW=law,
        NUMBER_OF_PROCESSES=1,
        USE_SINGLE_LD_COEFFICIENTS=False,
    ):
        try:
            for normalize in (False, True):
                reference_observer = _make_observer(case, bands, law, custom_coefficients=custom)
                expected = _capture_science(_reference_surface_preparation, reference_observer, normalize=normalize)
                current_observer = _make_observer(case, bands, law, custom_coefficients=custom)
                actual = _capture_science(utils.prep_surface_params, current_observer, normalize=normalize)

                _assert_matching_science(expected, actual, bands)
        finally:
            ld.clear_ld_caches()


@pytest.mark.parametrize("bands", [BANDS, ["bolometric", *BANDS]])
def test_surface_preparation_does_not_construct_passbands(bands: list[str]) -> None:
    with settings.override(LIMB_DARKENING_LAW="cosine", NUMBER_OF_PROCESSES=1):
        observer = _make_observer("circular", bands, "cosine", custom_coefficients=True)
        # Observer initialization must construct the actual passband containers.
        with patch(
            "elisa.observer.passband.PassbandContainer.__init__",
            side_effect=AssertionError("Unexpected passband construction during surface preparation"),
        ):
            _, fluxes = observer.observe.lc(phases=np.array([0.0, 0.25]), normalize=False)

        assert list(fluxes) == bands
        assert all(np.isfinite(values).all() for values in fluxes.values())


@pytest.mark.parametrize("law", ["linear", "cosine", "logarithmic", "square_root"])
def test_reflection_uses_bolometric_coefficients(law: str) -> None:
    """Reflection retains bolometric LD when photometric coefficients differ."""
    with settings.override(LIMB_DARKENING_LAW=law, NUMBER_OF_PROCESSES=1, REFLECTION_EFFECT=True):
        observer = _make_observer("circular", BANDS, law, custom_coefficients=True)
        bolometric = [0.8] if law in ("linear", "cosine") else [0.8, 0.1]
        for component in (observer._system.primary, observer._system.secondary):
            component.limb_darkening_coefficients["bolometric"] = bolometric

        original = ld.get_bolometric_ld_coefficients
        integrated = ld.calculate_integrated_limb_darkening_factor
        reflection_arrays: list[FloatArray] = []
        reflection_integrals: list[FloatArray] = []

        def capture_bolometric(*args: Any, **kwargs: Any) -> FloatArray:
            coefficients = original(*args, **kwargs)
            assert_array_equal(coefficients, np.tile(bolometric, (coefficients.shape[1], 1)).T)
            reflection_arrays.append(coefficients)
            return coefficients

        def capture_integral(*args: Any, **kwargs: Any) -> FloatArray:
            coefficients = args[1] if len(args) > 1 else kwargs["coefficients"]
            result = integrated(*args, **kwargs)
            if any(coefficients is sample for sample in reflection_arrays):
                reflection_integrals.append(result)
            return result

        with (
            patch.object(ld, "get_bolometric_ld_coefficients", capture_bolometric),
            patch.object(ld, "calculate_integrated_limb_darkening_factor", capture_integral),
        ):
            actual = _capture_science(utils.prep_surface_params, observer, normalize=False)
        expected = _capture_science(_reference_surface_preparation, observer, normalize=False)
        _assert_matching_science(expected, actual, BANDS)
        assert reflection_arrays
        assert len(reflection_integrals) == len(reflection_arrays)
        hemisphere = 1 - bolometric[0] / 3
        if law == "logarithmic":
            hemisphere += 2 * bolometric[1] / 9
        elif law == "square_root":
            hemisphere -= bolometric[1] / 5
        for integral in reflection_integrals:
            assert_allclose(integral, np.pi * hemisphere, rtol=1e-14)
