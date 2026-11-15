"""Independent hemispheric conservation through both surface preparation paths."""

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad

from elisa import ld, settings
from elisa.binary_system.curves import utils as binary
from elisa.single_system.curves import utils as single


@pytest.mark.parametrize("law", ["linear", "cosine", "logarithmic", "square_root"])
@pytest.mark.parametrize("system_kind", ["binary", "single"])
@pytest.mark.parametrize("equal", [False, True])
def test_passband_conservation(
    monkeypatch: pytest.MonkeyPatch, law: str, system_kind: str, equal: bool,
) -> None:
    """Distinct faces, components and bands recover their supplied surface flux."""
    module = binary if system_kind == "binary" else single
    components = ["primary", "secondary"] if system_kind == "binary" else ["star"]
    bands = ["B", "V", "R", "I", "bolometric"]
    coefficients = {}
    fluxes = {}
    container = SimpleNamespace(**{name: SimpleNamespace() for name in components})
    for index, component in enumerate(components):
        coefficients[component] = {}
        fluxes[component] = {}
        for band_index, band in enumerate(bands):
            a = np.array([.2, .4, .6]) + .03 * index + (0 if equal else .02 * band_index)
            b = np.array([.1, .15, .2])
            coefficients[component][band] = (
                a[:, None] if law in ("linear", "cosine") else np.column_stack((a, b))
            )
            fluxes[component][band] = np.array([1e6, 2e6, 4e6]) * (index + 1) * (band_index + 1)

    calls = []
    def get_coefficients(*_args: object, **kwargs: object) -> dict:
        calls.append(kwargs)
        return coefficients

    monkeypatch.setattr(module, "get_limbdarkening_cfs", get_coefficients)
    flux_function = "_get_normal_radiance" if system_kind == "binary" else "get_normal_radiance"
    monkeypatch.setattr(module, flux_function, lambda *_args, **_kwargs: {
        component: values.copy() for component, values in fluxes.items()
    })
    with settings.override(LIMB_DARKENING_LAW=law):
        intensities, actual_coefficients = module.prep_surface_params(
            container, passband=dict.fromkeys(bands), write_to_containers=True,
        )
    assert len(calls) == 1
    assert actual_coefficients is coefficients
    for component in components:
        assert getattr(container, component).normal_radiance is intensities[component]
        for band in bands:
            for face, coefficient in enumerate(coefficients[component][band]):
                a = coefficient[0]
                b = coefficient[-1]
                def integrand(mu: float, a: float = a, b: float = b) -> float:
                    value = 1 - a * (1 - mu)
                    if law == "logarithmic":
                        value -= b * mu * np.log(mu)
                    elif law == "square_root":
                        value -= b * (1 - np.sqrt(mu))
                    return value * mu
                integral = 2 * np.pi * quad(integrand, 0, 1, epsabs=1e-12, epsrel=1e-12)[0]
                factor = ld.calculate_integrated_limb_darkening_factor(law, coefficient[:, None])[0]
                assert_allclose(factor, integral, rtol=1e-11)
                assert_allclose(intensities[component][band][face] * integral,
                                fluxes[component][band][face], rtol=1e-11)
