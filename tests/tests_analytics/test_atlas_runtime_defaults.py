"""CK04 numerical fitting defaults and optional fitter-level atmosphere configuration."""

from __future__ import annotations

from copy import deepcopy

import pytest

from elisa.analytics.binary_fit import least_squares
from elisa.analytics.params import parameters
from elisa.atm import const as atm_const
from elisa.base.error import ElisaAnalyticsTaskError, ElisaInitialParametersError, ElisaInvalidKwargsError
from tests import set_astropy_units
from tests.tests_analytics.fitting_helpers import lc_parameters

set_astropy_units()

CK04_METALLICITY = (-2.5, 0.5)


def make_star(**kwargs) -> parameters.StarInitialParameters:
    return parameters.StarInitialParameters(**{"t_eff": {"value": 6000.0}, "metallicity": {"value": 0.0}, **kwargs})


def test_default_bounds_use_the_star_default_atmosphere() -> None:
    star = make_star()
    assert (star.t_eff.min, star.t_eff.max) == (3500.0, 50000.0)
    assert (star.metallicity.min, star.metallicity.max) == CK04_METALLICITY


def test_explicit_bounds_are_preserved_and_partial_bounds_fill_defaults() -> None:
    inputs = {
        "t_eff": {"value": 60, "min": 55, "max": 70, "unit": "kK", "sigma": 1},
        "metallicity": {"value": -5.5, "min": -6.0, "max": 2.0},
    }
    original = deepcopy(inputs)
    star = make_star(**inputs)
    assert (star.t_eff.min, star.t_eff.value, star.t_eff.max) == (55000, 60000, 70000)
    assert (star.metallicity.min, star.metallicity.value, star.metallicity.max) == (-6, -5.5, 2)
    assert inputs == original
    lower = make_star(metallicity={"value": 0, "min": -8})
    assert (lower.metallicity.min, lower.metallicity.max) == (-8, CK04_METALLICITY[1])


def test_atmosphere_is_not_a_fitting_parameter() -> None:
    with pytest.raises(ElisaInvalidKwargsError):
        make_star(atmosphere="bb")
    cfg, _ = lc_parameters()
    cfg["primary"]["atmosphere"] = "bb"
    with pytest.raises(ElisaInvalidKwargsError):
        parameters.BinaryInitialParameters(**cfg)


def validated_atmosphere_model(atmosphere_model: object) -> object:
    fit = least_squares.DetachedLightCurveFit()
    fit.atmosphere_model, fit.limb_darkening_coefficients = atmosphere_model, None
    fit._validate_atmosphere_parameters()
    return fit.atmosphere_model


def test_atmosphere_model_is_optional_and_normalized_per_component() -> None:
    assert validated_atmosphere_model(None) is None
    assert validated_atmosphere_model({"primary": "black_body", "secondary": "kurucz"}) == {
        "primary": atm_const.Atmosphere.BB,
        "secondary": atm_const.Atmosphere.K93,
    }
    assert validated_atmosphere_model({"secondary": "castelli"}) == {"secondary": atm_const.Atmosphere.CK04}


@pytest.mark.parametrize("value", [{"primary": "invalid"}, {"primary": ""}, {"primary": None}])
def test_invalid_atmosphere_model_values_raise(value: dict) -> None:
    with pytest.raises(ElisaInitialParametersError):
        validated_atmosphere_model(value)


@pytest.mark.parametrize("value", [{"tertiary": "bb"}, "bb", ["bb"]])
def test_invalid_atmosphere_model_structure_raises(value: object) -> None:
    with pytest.raises(ElisaAnalyticsTaskError):
        validated_atmosphere_model(value)


def test_atmosphere_model_reaches_synthesis_only_for_selected_components() -> None:
    cfg, _ = lc_parameters()
    flat = parameters.BinaryInitialParameters(**cfg).get_fixed(jsonify=False)
    assert not any(key.endswith("@atmosphere") for key in flat)
    extended = parameters.extend_json_with_atm_params(
        {"primary@t_eff": 6000.0, "secondary@t_eff": 5000.0},
        atmosphere_model={"primary": atm_const.Atmosphere.BB},
    )
    assert extended["primary@atmosphere"] == "bb"
    assert "secondary@atmosphere" not in extended
