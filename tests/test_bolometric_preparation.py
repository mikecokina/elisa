"""Preserve scientific parity with the former bolometric passband preparation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias
from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from numpy.typing import NDArray

from elisa import BinarySystem, Observer, Star, ld, settings
from elisa.atm import atm
from elisa.binary_system.curves import utils
from elisa.observer.passband import init_bolometric_passband

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


def _legacy_surface_preparation(
    system: OrbitalPositionContainer,
    *,
    return_values: bool = True,
    write_to_containers: bool = False,
    **kwargs: Any,
) -> SurfaceParameters | None:
    """Reproduce the previous implementation, including its unused arguments."""
    ld_coefficients = utils.get_limbdarkening_cfs(system, **kwargs)
    # The oracle deliberately follows the old implementation's internal call.
    # noinspection PyProtectedMember
    normal_radiance = utils._get_normal_radiance(system, **kwargs)  # noqa: SLF001
    if "bolometric" in ld_coefficients["primary"]:
        bolometric_coefficients = {
            component: {"bolometric": ld_coefficients[component]["bolometric"]}
            for component in settings.BINARY_COUNTERPARTS
        }
    else:
        passband, left_bandwidth, right_bandwidth = init_bolometric_passband()
        legacy_arguments = {
            "passband": {"bolometric": passband},
            "left_bandwidth": left_bandwidth,
            "right_bandwith": right_bandwidth,
            "atlas": "whatever",
        }
        bolometric_coefficients = utils.get_limbdarkening_cfs(system, **legacy_arguments)

    normal_radiance = atm.correct_normal_radiance_to_optical_depth(normal_radiance, bolometric_coefficients)
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
        assert_array_equal(expected.fluxes[band], actual.fluxes[band], strict=True)

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
def test_bolometric_scientific_parity(*, law: str, custom: bool, case: str, bands: list[str]) -> None:
    with settings.override(
        LIMB_DARKENING_LAW=law,
        NUMBER_OF_PROCESSES=1,
        USE_SINGLE_LD_COEFFICIENTS=False,
    ):
        try:
            for normalize in (False, True):
                legacy_observer = _make_observer(case, bands, law, custom_coefficients=custom)
                expected = _capture_science(_legacy_surface_preparation, legacy_observer, normalize=normalize)
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
