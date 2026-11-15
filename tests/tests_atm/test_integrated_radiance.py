"""Scalar interpolation preserves prepared-spectrum quadrature and face order."""
from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose

from elisa.atm import atm


@pytest.mark.parametrize("scale", [0.0, 1e-30, 1.0, 1e20])
@pytest.mark.parametrize("irregular", [False, True])
def test_integrated_radiance_reference(
    monkeypatch: pytest.MonkeyPatch, scale: float, irregular: bool,
) -> None:
    """Compare with independent float64 quadrature and the legacy Tensor path."""
    wave = np.array([1000., 1100., 1350., 1800., 2200., 3000.])
    low = atm.AtmDataContainer(atm.AtmModel(scale * np.arange(1., 7.), wave), 5000., 2., 0., "low")
    high_wave = wave if not irregular else np.array([1000., 1200., 1700., 2100., 3000.])
    high = atm.AtmDataContainer(
        atm.AtmModel(scale * np.linspace(2., 8., len(high_wave)), high_wave), 6000., 2., 0., "high",
    )
    containers = atm.arange_atm_to_same_wavelength(deepcopy([low, high]))
    mapping = {"low": [0, 1, 3], "high": [2, 4, 5]}
    temperatures = np.array([5000., 5432.123456789, 6000.])
    prepared = {"test": containers}
    monkeypatch.setattr(
        atm.NaiveInterpolatedAtm, "_prepare_atlas_spectra", lambda *_a, **_k: (prepared, mapping, 1e7, 1e-10),
    )
    actual = atm.NaiveInterpolatedAtm.atlas_radiance(temperatures, np.ones(3), 0., "ck04")["test"]
    matrix = atm.remap_passbanded_unique_atms_to_matrix(prepared, mapping)["test"]
    ordered = atm.remap_unique_atm_container_to_origin(containers, mapping)
    weights = atm.NaiveInterpolatedAtm.compute_interpolation_weights(temperatures, ordered[3:], ordered[:3])
    assert_allclose(weights[[0, 2]], 1.)  # equal top/bottom: NaN becomes one
    flux = matrix[:3] + weights[:, None] * (matrix[3:] - matrix[:3])
    expected = atm.compute_normal_intensity(flux, containers[0].model.wavelength, flux_mult=1e7, wave_mult=1e-10)
    assert_allclose(actual, expected, rtol=2e-15, atol=0.)
    legacy = atm.NaiveInterpolatedAtm.interpolate_spectra({"test": ordered}, {"test": matrix}, temperatures)
    reference = atm.compute_normal_radiances(legacy, flux_mult=1e7, wave_mult=1e-10)["test"]
    assert_allclose(actual, reference, rtol=1e-7, atol=0.)


@pytest.mark.parametrize("power", [0, 1, 2])
@pytest.mark.parametrize("irregular", [False, True])
@pytest.mark.parametrize("scale", [0., 1e-30, 1.])
def test_analytical_pipeline(
    monkeypatch: pytest.MonkeyPatch, power: int, irregular: bool, scale: float,
) -> None:
    """Known polynomial integrals through stripping, passband, mapping and T^4 interpolation."""
    from types import SimpleNamespace

    wave = np.array([1., 1.3, 2., 2.8, 4.]) if irregular else np.linspace(1., 4., 7)
    # Flux is linear; response makes the integrand constant, linear or quadratic.
    flux_power = min(power, 1)
    response_power = power - flux_power
    models = [
        atm.AtmDataContainer(atm.AtmModel(scale * factor * wave**flux_power, wave), temp, 4., 0., path)
        for factor, temp, path in [(2., 5000., "low"), (5., 6000., "high")]
    ]
    mapping = {"low": [0, 1, 2, 3], "high": [4, 5, 6, 7]}
    monkeypatch.setattr(atm.NaiveInterpolatedAtm, "atm_files", lambda *_a: [])
    monkeypatch.setattr(atm, "read_unique_atm_tables", lambda *_a: (deepcopy(models), mapping))
    weights = np.array([0., .2, .7, 1.])
    temperatures = ((1 - weights) * 5000.**4 + weights * 6000.**4)**.25
    band = SimpleNamespace(left_bandwidth=1., right_bandwidth=4., akima=lambda x: x**response_power)
    actual = atm.NaiveInterpolatedAtm.atlas_radiance(
        temperatures, np.ones(4), 0., "ck", passband={"test": band}, left_bandwidth=1., right_bandwidth=4.,
    )["test"]
    # FLAM integrated in Angstrom -> W/m²; no extra pi and no bandwidth normalization.
    expected = scale * (2 + 3 * weights) * (4.**(power + 1) - 1) / (power + 1) * 1e-3
    assert_allclose(actual, expected, rtol=3e-15, atol=0.)


def test_flux_to_normal_intensity_pi() -> None:
    """Isotropic hemispheric flux is pi times normal intensity."""
    from elisa import ld

    factor = ld.calculate_integrated_limb_darkening_factor("linear", np.zeros((1, 1)))
    assert_allclose(factor, np.pi, rtol=1e-15)


@pytest.mark.requires_atmosphere(atlas="ck04")
@pytest.mark.requires_atmosphere(atlas="k93")
@pytest.mark.parametrize("atlas", ["ck", "k"])
@pytest.mark.parametrize("metallicity", [-.5, 0., .5])
def test_real_atmosphere_float64_reference(atlas: str, metallicity: float) -> None:
    """Real-table radiances agree with independent float64 spectral quadrature."""
    from scipy.integrate import simpson

    from elisa import Observer

    observer = Observer(passband=["Generic.Bessell." + band for band in "BVRI"] + ["bolometric"])
    temperature = np.array([5000., 5123.456789, 6500., 6500. + 1e-7, 7000.])
    gravity = np.array([2., 2.5, 2., 2., 3.])
    kwargs = {
        "passband": observer.passband,
        "left_bandwidth": observer.left_bandwidth,
        "right_bandwidth": observer.right_bandwidth,
    }
    prepared, mapping, fm, wm = atm.NaiveInterpolatedAtm._prepare_atlas_spectra(
        temperature, gravity, metallicity, atlas, **kwargs,
    )
    actual = atm.NaiveInterpolatedAtm.atlas_radiance(temperature, gravity, metallicity, atlas, **kwargs)
    for band, containers in prepared.items():
        ordered = atm.remap_unique_atm_container_to_origin(containers, mapping)
        n = len(temperature)
        low_t = np.array([c.temperature for c in ordered[:n]])
        high_t = np.array([c.temperature for c in ordered[n:]])
        weights = np.ones(n)
        different = high_t != low_t
        weights[different] = (temperature[different]**4 - low_t[different]**4) / (
            high_t[different]**4 - low_t[different]**4
        )
        low = np.array([c.model.flux for c in ordered[:n]])
        high = np.array([c.model.flux for c in ordered[n:]])
        reference = fm * wm * simpson(low + weights[:, None] * (high - low), x=containers[0].model.wavelength)
        assert np.isfinite(actual[band]).all()
        assert (actual[band] > 0).all()
        assert_allclose(actual[band], reference, rtol=3e-15, atol=0.)
        assert_allclose(actual[band][2], actual[band][3], rtol=1e-9, atol=0.)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_spectrum_is_not_silently_repaired(monkeypatch: pytest.MonkeyPatch, invalid: float) -> None:
    """Malformed prepared spectra propagate nonfinite values rather than becoming finite flux."""
    wave = np.linspace(1000., 2000., 5)
    flux = np.ones(5)
    flux[2] = invalid
    container = atm.AtmDataContainer(atm.AtmModel(flux, wave), 5000., 4., 0., "same")
    monkeypatch.setattr(
        atm.NaiveInterpolatedAtm, "_prepare_atlas_spectra",
        lambda *_a, **_k: ({"test": [container]}, {"same": [0, 1]}, 1e7, 1e-10),
    )
    with np.errstate(invalid="ignore"):
        result = atm.NaiveInterpolatedAtm.atlas_radiance(np.array([5000.]), np.array([2.]), 0., "ck")
    assert np.isnan(result["test"][0])
