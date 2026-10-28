"""Fit-result reconstruction uses the same missing defaults as LC synthesis."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pytest
from astropy import units as u
from jsonschema import ValidationError
from numpy.testing import assert_allclose

from elisa import BinarySystem, const
from elisa.analytics.models import lc
from elisa.analytics.params import parameters
from elisa.analytics.params.conf import DEFAULT_FLOAT_UNITS, JOINT_POSTERIOR_PERCENTILES_ESTIMATE


def fit_results(kind: str, method: str) -> dict[str, Any]:
    values = {
        "system@period": 3.5,
        "system@inclination": 85.0,
        "system@argument_of_periastron": 27.0,
        "system@eccentricity": 0.2,
        "system@primary_minimum_time": 0.0,
        "system@phase_shift": 0.03,
        "system@additional_light": 0.05,
    }
    if kind == "standard":
        values.update({"primary@mass": 1.5, "secondary@mass": 0.75})
    else:
        values.update({"system@mass_ratio": 0.5, "system@semi_major_axis": 10.0})
    for component, temperature in (("primary", 6250.0), ("secondary", 5000.0)):
        values.update({
            f"{component}@surface_potential": 10.0,
            f"{component}@t_eff": temperature,
            f"{component}@gravity_darkening": 0.32,
            f"{component}@albedo": 0.5,
            f"{component}@metallicity": -0.1,
        })
    flat = {
        key: {"value": value, "fixed": True, "unit": str(DEFAULT_FLOAT_UNITS[key.split("@")[-1]])}
        for key, value in values.items()
    }
    flat["system@inclination"].update(fixed=False, min=80.0, max=90.0)
    if method == "mcmc":
        flat["system@inclination"]["confidence_interval"] = {"min": 84.0, "max": 86.0}
    flat["secondary@t_eff"] = {
        "value": 5000.0, "constraint": "primary@t_eff * 0.8", "unit": "K",
    }
    if method == "mcmc":
        flat["secondary@t_eff"].update(
            confidence_interval={"min": 4900.0, "max": 5100.0},
            estimate=JOINT_POSTERIOR_PERCENTILES_ESTIMATE,
        )
    flat["r_squared"] = {"value": 0.99, "unit": None}
    return parameters.serialize_result(flat)


@pytest.mark.parametrize("kind", ["standard", "community"])
@pytest.mark.parametrize("method", ["lsq", "mcmc"])
@pytest.mark.parametrize(
    ("gamma", "primary_sync", "secondary_sync"),
    [
        pytest.param(None, None, None, id="missing-defaults"),
        pytest.param(0.0, None, 0.75, id="partly-missing"),
        pytest.param(0.0, 1.5, 0.75, id="explicit-zero-and-asynchronous"),
        pytest.param(12000.0, 1.5, 0.75, id="explicit-nonzero-and-asynchronous"),
    ],
)
def test_reconstruction_matches_synthesis(
    kind: str, method: str, gamma: float | None, primary_sync: float | None, secondary_sync: float | None,
) -> None:
    result = fit_results(kind, method)
    if gamma is not None:
        result["system"]["gamma"] = {
            "value": gamma, "fixed": False, "min": -20000.0, "max": 20000.0, "unit": "m / s",
        }
    if primary_sync is not None:
        result["primary"]["synchronicity"] = {"value": primary_sync, "fixed": True, "unit": ""}
    if secondary_sync is not None:
        result["secondary"]["synchronicity"] = {
            "value": secondary_sync, "constraint": "system@period * 0 + 0.75", "unit": "",
        }
    before = deepcopy(result)
    flat = {key: metadata["value"] for key, metadata in parameters.deserialize_result(result).items()}
    synthesized = lc.prepare_binary(**flat)
    rebuilt = BinarySystem.from_fit_results(result)

    assert result == before
    assert rebuilt.gamma == (0.0 if gamma is None else gamma)
    assert rebuilt.primary.synchronicity == (1.0 if primary_sync is None else primary_sync)
    assert rebuilt.secondary.synchronicity == (1.0 if secondary_sync is None else secondary_sync)
    assert rebuilt.period == 3.5
    assert_allclose((rebuilt.inclination * u.rad).to_value(u.deg), 85.0)
    assert_allclose((rebuilt.argument_of_periastron * u.rad).to_value(u.deg), 27.0)
    assert rebuilt.mass_ratio == pytest.approx(0.5)
    if kind == "standard":
        expected_mass = 1.5 * u.solMass.to(u.kg)
    else:
        expected_mass = 4 * np.pi**2 * (10 * u.solRad.to(u.m))**3 / (const.G * 1.5 * (3.5 * 86400)**2)
        assert_allclose((rebuilt.semi_major_axis * u.m).to_value(u.solRad), 10.0)
    assert_allclose(rebuilt.primary.mass, expected_mass)
    assert_allclose(rebuilt.secondary.mass, expected_mass * 0.5)
    for key in (
        "gamma", "period", "inclination", "argument_of_periastron", "eccentricity", "phase_shift", "additional_light",
    ):
        assert_allclose(getattr(rebuilt, key), getattr(synthesized, key))
    for component in ("primary", "secondary"):
        for key in (
            "mass", "t_eff", "surface_potential", "synchronicity", "gravity_darkening", "albedo", "metallicity", "polar_radius",
        ):
            assert_allclose(getattr(getattr(rebuilt, component), key), getattr(getattr(synthesized, component), key))
    phases = np.array([0.0, 0.25, 0.5])
    assert_allclose(rebuilt.orbit.orbital_motion(phases), synthesized.orbit.orbital_motion(phases))


@pytest.mark.parametrize("kind", ["standard", "community"])
@pytest.mark.parametrize(
    ("component", "parameter"),
    [("system", key) for key in ("period", "inclination", "argument_of_periastron", "eccentricity")]
    + [(component, key) for component in ("primary", "secondary") for key in ("t_eff", "surface_potential")],
)
def test_missing_mandatory_parameters_are_rejected(kind: str, component: str, parameter: str) -> None:
    result = fit_results(kind, "lsq")
    del result[component][parameter]
    with pytest.raises(ValidationError):
        BinarySystem.from_fit_results(result)


@pytest.mark.parametrize(
    ("kind", "component", "parameter"),
    [("standard", "primary", "mass"), ("standard", "secondary", "mass"),
     ("community", "system", "mass_ratio"), ("community", "system", "semi_major_axis")],
)
def test_missing_physical_scale_is_rejected(kind: str, component: str, parameter: str) -> None:
    result = fit_results(kind, "lsq")
    del result[component][parameter]
    with pytest.raises(ValidationError):
        BinarySystem.from_fit_results(result)


@pytest.mark.parametrize("component", ["system", "primary", "secondary"])
def test_missing_component_is_rejected(component: str) -> None:
    result = fit_results("standard", "lsq")
    del result[component]
    with pytest.raises(ValidationError):
        BinarySystem.from_fit_results(result)


@pytest.mark.parametrize(
    ("component", "parameter"),
    [("system", "gamma"), ("primary", "synchronicity"), ("secondary", "synchronicity")],
)
def test_explicit_none_is_not_replaced(component: str, parameter: str) -> None:
    result = fit_results("standard", "lsq")
    result[component][parameter] = {"value": None, "fixed": True}
    with pytest.raises(ValidationError):
        BinarySystem.from_fit_results(result)


def test_from_json_still_requires_explicit_model_inputs() -> None:
    result = fit_results("standard", "lsq")
    data = {component: {key: metadata["value"] for key, metadata in content.items()}
            for component, content in result.items() if component != "r_squared"}
    with pytest.raises(ValidationError):
        BinarySystem.from_json(data)
