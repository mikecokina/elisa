"""CPU blackbody batching preserves the native passband integral for each face."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from elisa import const
from elisa.atm import atm
from elisa.observer.passband import PassbandContainer


def test_shared_blackbody_spectrum(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise overlapping grids, repeated temperatures, multiple batches and mutable passbands."""
    standard = np.array([2100., 2600., 3000., 3800., 4100., 4500., 5000., 5300.])
    monkeypatch.setattr(atm, "get_standard_wavelengths", lambda: standard)
    blue = pd.DataFrame({"wavelength": standard[:5], "throughput": [0., .25, 1., .25, 0.]})
    red = pd.DataFrame({"wavelength": standard[2:], "throughput": [0., .25, 1., .6, .3, 0.]})
    bands = {"blue": PassbandContainer(blue, "blue"), "red": PassbandContainer(red, "red")}
    temperatures = np.array([6100.25, 5000., 6100.25, 6100.26, 7800.])
    model = atm.NaiveInterpolatedAtm
    localized = model.arange_black_body_localized_atms(temperatures, bands)
    reference = atm.compute_normal_radiances(localized, flux_mult=const.PI, wave_mult=1e-10)
    all_waves = np.unique(np.concatenate([band["wave"] * 1e-10 for band in localized.values()]))
    monkeypatch.setattr(atm, "_BLACKBODY_BATCH_ELEMENTS", all_waves.size * 2)
    planck = Mock(wraps=atm.planck_function)
    monkeypatch.setattr(atm, "planck_function", planck)

    actual = model.black_body_radiance(temperatures, passband=bands)
    assert list(actual) == list(bands)
    assert planck.call_count == 2
    for call in planck.call_args_list:
        wavelengths, rows = call.args
        assert_array_equal(wavelengths, all_waves)
        assert rows.shape == (2, 1)
    for name, result in actual.items():
        assert_allclose(result, reference[name], rtol=2e-15, atol=0)
        assert result[0] == result[2]
        assert result[0] != result[3]

    bands["blue"].table = bands["blue"].table.assign(throughput=lambda df: .5 * df.throughput)
    updated = model.black_body_radiance(temperatures, passband=bands)
    assert_allclose(updated["blue"], .5 * actual["blue"], rtol=2e-15)
    assert_array_equal(updated["red"], actual["red"])


def test_blackbody_two_wavelengths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep two-sample quadrature and the original treatment of negative and NaN throughput."""
    wave = np.array([3000., 4500.])
    monkeypatch.setattr(atm, "get_standard_wavelengths", lambda: wave)
    band = PassbandContainer(pd.DataFrame({"wavelength": wave, "throughput": [1., 1.]}), "bolometric")
    temperature = np.array([5000., 7000.])
    for endpoint in (np.nan, -0.25):
        band.akima = Mock(return_value=np.array([endpoint, 1.]))
        actual = atm.NaiveInterpolatedAtm.black_body_radiance(temperature, passband={"test": band})
        density = atm.planck_function(wave[1] * band.wave_to_si_mult, temperature)
        assert_allclose(actual["test"], 0.5 * (wave[1] - wave[0]) * const.PI * 1e-10 * density, rtol=2e-15)
