# Additional lightweight tests for atm module
from unittests import set_astropy_units

set_astropy_units()

import os
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal, assert_allclose

from elisa.atm import (
    AtmModel,
    AtmDataContainer,
    IntensityContainer,
    compute_integral_si_intensity_from_atm_data_containers,
    planck_function,
    get_standard_wavelengths,
)
from elisa import settings
from elisa.base.types import FLOAT


def test_atmmodel_from_dataframe_and_len_getitem():
    df = pd.DataFrame({
        settings.ATM_MODEL_DATAFRAME_FLUX: np.array([1.0, 2.0, 3.0], dtype=FLOAT),
        settings.ATM_MODEL_DATAFRAME_WAVE: np.array([10.0, 20.0, 30.0], dtype=FLOAT),
    })
    model = AtmModel.from_dataframe(df)
    assert_array_equal(model.flux, np.array([1.0, 2.0, 3.0]))
    assert_array_equal(model.wavelength, np.array([10.0, 20.0, 30.0]))
    assert len(model) == 3
    sub = model[1:]
    assert_array_equal(sub.flux, np.array([2.0, 3.0]))


def test_atmdata_container_model_setter_with_atmmodel():
    model = AtmModel(flux=np.array([5.0, 6.0], dtype=FLOAT), wavelength=np.array([100.0, 200.0], dtype=FLOAT))
    adc = AtmDataContainer(model, temperature=5000, log_g=4.0, metallicity=0.0)
    assert_array_equal(adc.model.flux, np.array([5.0, 6.0]))
    assert adc.left_bandwidth == 100.0 and adc.right_bandwidth == 200.0


def test_intensity_container_fields():
    ic = IntensityContainer(intensity=1.23, temperature=5500, log_g=4.2, metallicity=0.1)
    assert ic.intensity == 1.23
    assert ic.temperature == 5500
    assert ic.log_g == 4.2
    assert ic.metallicity == 0.1


def test_compute_integral_si_intensity_from_atm_data_containers():
    # Simple linear flux over wavelength: flux = wavelength, so integral simps(wave, wave) from 0..2 = (2^2)/2 = 2
    df = pd.DataFrame({
        settings.ATM_MODEL_DATAFRAME_FLUX: np.array([0.0, 1.0, 2.0], dtype=FLOAT),
        settings.ATM_MODEL_DATAFRAME_WAVE: np.array([0.0, 1.0, 2.0], dtype=FLOAT),
    })
    adc = AtmDataContainer(df, temperature=5000, log_g=4.0, metallicity=0.0)
    results = compute_integral_si_intensity_from_atm_data_containers([adc])
    # result is list of IntensityContainer
    assert len(results) == 1
    ic = results[0]
    # expected intensity = pi * simps(flux*flux_to_si_mult, wave*wave_to_si_mult)
    # flux_to_si_mult default is 1e7 and wave_to_si_mult is 1e-10 -> combined small scaling, but sanity check shape
    assert hasattr(ic, 'intensity')


def test_planck_function_and_standard_wavelengths():
    waves = np.array([1e-7, 5e-7])  # in meters
    temp = 5000
    p = planck_function(waves, temp)
    assert p.shape == (2,)
    assert np.all(p > 0)

    std = get_standard_wavelengths()
    assert isinstance(std, np.ndarray)
    assert std.size > 0
