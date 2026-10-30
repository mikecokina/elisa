"""Preserve Akima flux precision through atmosphere bandwidth preparation."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from scipy.interpolate import Akima1DInterpolator

from elisa import settings
from elisa.atm import atm
from elisa.observer.passband import PassbandContainer


def rounded_boundary_extension(container, left, right):
    """Retain the previous boundary implementation solely for before/after comparisons."""
    model = container.model
    interpolator = Akima1DInterpolator(model.wavelength, model.flux)
    flux = interpolator([left, right])
    if np.isnan(flux).any():
        raise atm.ElisaAtmosphereError("Interpolation on bandwidth boundaries resulted in NaN value.")
    model.wavelength[np.array([0, -1])] = [left, right]
    model.flux[np.array([0, -1])] = flux
    model.flux = np.round(model.flux, 10)
    container.model = model
    return container


def make_spectrum(scale=1.0):
    wave = np.arange(11, dtype=float)
    flux = scale * (1.123456789123456 + np.sin(wave / 3.0))
    return atm.AtmDataContainer(atm.AtmModel(flux=flux, wavelength=wave), 5000.0, 2.0, 0.0)


@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e8])
def test_extension_keeps_interior_precision_and_exact_akima_boundaries(scale):
    original = make_spectrum(scale)
    old = rounded_boundary_extension(deepcopy(original), 0.25, 9.75)
    expected = original.model.flux.copy()
    expected[[0, -1]] = Akima1DInterpolator(original.model.wavelength, original.model.flux)([0.25, 9.75])
    result = atm.extend_atm_container_on_bandwidth_boundary(original, 0.25, 9.75)
    assert result is original
    np.testing.assert_array_equal(result.model.flux, expected)
    np.testing.assert_array_equal(old.model.flux, np.round(expected, 10))
    np.testing.assert_array_equal(result.model.wavelength, [0.25, *range(1, 10), 9.75])
    assert np.isfinite(result.model.flux).all()
    assert np.any(result.model.flux != old.model.flux)


@pytest.mark.parametrize("inplace", [False, True])
def test_trimming_preserves_ownership_and_unrounded_interior(inplace):
    original = make_spectrum()
    initial = deepcopy(original)
    result = atm.strip_to_bandwidth(original, 1.25, 8.75, inplace=inplace)
    np.testing.assert_array_equal(result.model.flux[1:-1], initial.model.flux[2:9])
    np.testing.assert_array_equal(
        result.model.flux[[0, -1]],
        Akima1DInterpolator(initial.model.wavelength[1:10], initial.model.flux[1:10])([1.25, 8.75]),
    )
    assert (result is original) == inplace
    if not inplace:
        np.testing.assert_array_equal(original.model.flux, initial.model.flux)
        np.testing.assert_array_equal(original.model.wavelength, initial.model.wavelength)
        assert not np.shares_memory(result.model.flux, original.model.flux)
        assert not np.shares_memory(result.model.wavelength, original.model.wavelength)


@pytest.mark.parametrize("left,right", [(-0.25, 9.75), (0.25, 10.25)])
def test_nan_boundary_errors_leave_input_untouched(left, right):
    original = make_spectrum()
    before = deepcopy(original)
    for extend in (rounded_boundary_extension, atm.extend_atm_container_on_bandwidth_boundary):
        with pytest.raises(atm.ElisaAtmosphereError, match="Interpolation on bandwidth boundaries resulted in NaN value."):
            extend(original, left, right)
        np.testing.assert_array_equal(original.model.flux, before.model.flux)
        np.testing.assert_array_equal(original.model.wavelength, before.model.wavelength)


def test_tiny_linear_spectrum_stays_continuous_across_grid_boundaries():
    wave = np.arange(9, dtype=float)
    flux = (wave + 1) * 2.0**-40
    original = atm.AtmDataContainer(atm.AtmModel(flux=flux, wavelength=wave), 5000.0, 2.0, 0.0)
    left = np.array([np.nextafter(2.0, 0.0), 2.0, np.nextafter(2.0, np.inf)])
    results = [atm.strip_to_bandwidth(original, value, 6.5) for value in left]
    # The exact linear solution exposes quantization without an arbitrary tolerance.
    for value, result in zip(left, results):
        np.testing.assert_array_equal(result.model.flux, (result.model.wavelength + 1) * 2.0**-40)
        assert result.model.wavelength[0] == value
        assert np.all(np.diff(result.model.wavelength) > 0)
        assert np.isfinite(result.model.flux).all()
    old = rounded_boundary_extension(deepcopy(original), 0.0, 8.0)
    np.testing.assert_array_equal(old.model.flux, np.zeros_like(flux))


def test_repeated_trimming_does_not_quantize_or_drift():
    wave = np.arange(9, dtype=float)
    original = atm.AtmDataContainer(
        atm.AtmModel(flux=(wave + 1) * 2.0**-40, wavelength=wave), 5000.0, 2.0, 0.0,
    )
    first = atm.strip_to_bandwidth(original, 1.25, 6.75)
    result = first
    for _ in range(50):
        result = atm.strip_to_bandwidth(result, 1.25, 6.75)
    np.testing.assert_array_equal(result.model.wavelength, first.model.wavelength)
    np.testing.assert_array_equal(result.model.flux, first.model.flux)
    np.testing.assert_array_equal(result.model.flux, (result.model.wavelength + 1) * 2.0**-40)


def test_absolute_radiance_retains_small_physical_flux():
    wave = np.arange(9, dtype=float)
    original = atm.AtmDataContainer(
        atm.AtmModel(flux=(wave + 1) * 2.0**-40, wavelength=wave), 5000.0, 2.0, 0.0,
    )
    with patch.object(atm, "extend_atm_container_on_bandwidth_boundary", rounded_boundary_extension):
        old = atm.strip_to_bandwidth(original, 1.0, 7.0)
    new = atm.strip_to_bandwidth(original, 1.0, 7.0)
    kwargs = dict(flux_mult=new.flux_to_si_mult, wave_mult=new.wave_to_si_mult)
    old_radiance = atm.compute_normal_intensity(old.model.flux[np.newaxis], old.model.wavelength, **kwargs)
    new_radiance = atm.compute_normal_intensity(new.model.flux[np.newaxis], new.model.wavelength, **kwargs)
    # Integral of (wavelength + 1) from 1 to 7, with the container's SI factors.
    expected = new.flux_to_si_mult * new.wave_to_si_mult * 30 * 2.0**-40
    np.testing.assert_array_equal(new_radiance, [expected])
    np.testing.assert_array_equal(old_radiance, [0.0])


def test_alignment_uses_wavelengths_and_retains_unrounded_akima_values():
    first = make_spectrum(1e-12)
    second = make_spectrum(1e-12)
    second.model.wavelength += 0.125
    models = [atm.strip_to_bandwidth(c, 1.25, 8.75) for c in (first, second)]
    union = np.unique(np.concatenate([c.model.wavelength for c in models]))
    expected = [Akima1DInterpolator(c.model.wavelength, c.model.flux)(union) for c in models]
    aligned = atm.arange_atm_to_same_wavelength(models)
    for model, flux in zip(aligned, expected):
        np.testing.assert_array_equal(model.model.wavelength, union)
        np.testing.assert_array_equal(model.model.flux, flux)
        assert np.isfinite(model.model.flux).all()
    assert atm.arange_atm_to_same_wavelength(aligned) is aligned


def test_ck04_spectra_and_absolute_radiance_match_unrounded_pipeline(monkeypatch):
    directory = Path(__file__).parents[1] / "data" / "ck04"
    monkeypatch.setattr(settings, "CK04_ATM_TABLES", directory)
    monkeypatch.setattr(atm.buffer, "ATMOSPHERE_TABLES", {})
    band = PassbandContainer.from_name("Generic.Bessell.U")
    args = np.array([10000.0, 10001.0, 10250.0]), np.repeat(2.0, 3), 0.0, "ck04"
    kwargs = dict(passband={"Generic.Bessell.U": band}, left_bandwidth=band.left_bandwidth,
                  right_bandwidth=band.right_bandwidth)
    with patch.object(atm, "extend_atm_container_on_bandwidth_boundary", rounded_boundary_extension):
        old, old_flux_mult, old_wave_mult = atm.NaiveInterpolatedAtm.get_atm_profiles(*args, **kwargs)
        old_radiance = atm.NaiveInterpolatedAtm.atlas_radiance(*args, **kwargs)
    new, flux_mult, wave_mult = atm.NaiveInterpolatedAtm.get_atm_profiles(*args, **kwargs)
    new_radiance = atm.NaiveInterpolatedAtm.atlas_radiance(*args, **kwargs)
    assert (flux_mult, wave_mult) == (old_flux_mult, old_wave_mult)
    for name, data in new.items():
        wave = data[settings.ATM_MODEL_DATAFRAME_WAVE]
        flux = data[settings.ATM_MODEL_DATAFRAME_FLUX]
        np.testing.assert_array_equal(wave, old[name][settings.ATM_MODEL_DATAFRAME_WAVE])
        # This U-band fixture's differences vanish at the existing Tensor conversion.
        np.testing.assert_array_equal(flux, old[name][settings.ATM_MODEL_DATAFRAME_FLUX])
        np.testing.assert_array_equal(new_radiance[name], old_radiance[name])
        assert np.all(np.diff(wave) > 0)
        assert np.isfinite(flux).all()
        assert np.isfinite(old_radiance[name]).all()
        np.testing.assert_array_equal(
            new_radiance[name], atm.compute_normal_intensity(flux, wave, flux_mult=flux_mult, wave_mult=wave_mult),
        )


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.requires_ld(passbands=("bolometric", "Generic.Bessell.V"))
@pytest.mark.requires_atmosphere(atlas="ck04")
def test_ck04_raw_and_normalized_eccentric_curves(normalize, monkeypatch):
    """Compare both versions through an eccentric, eclipsing CK04 binary."""
    from elisa import Observer
    from tests.utils import BINARY_SYSTEM_PARAMS, prepare_binary_system

    monkeypatch.setattr(atm.buffer, "ATMOSPHERE_TABLES", {})
    binary = prepare_binary_system(BINARY_SYSTEM_PARAMS["detached.ecc"])
    binary.primary.atmosphere = "ck04"
    binary.secondary.atmosphere = "ck04"
    observer = Observer(passband=["Generic.Bessell.V"], system=binary)
    phases = np.linspace(-0.5, 0.5, 11)
    with patch.object(atm, "extend_atm_container_on_bandwidth_boundary", rounded_boundary_extension):
        old_phases, old_fluxes = observer.observe.lc(phases=phases, normalize=normalize)
    new_phases, new_fluxes = observer.observe.lc(phases=phases, normalize=normalize)
    np.testing.assert_array_equal(new_phases, old_phases)
    for name, flux in new_fluxes.items():
        assert np.isfinite(flux).all()
        np.testing.assert_array_equal(flux, old_fluxes[name])
        # Equality also guards against new phase-to-phase boundary artifacts.
        np.testing.assert_array_equal(np.diff(flux), np.diff(old_fluxes[name]))
