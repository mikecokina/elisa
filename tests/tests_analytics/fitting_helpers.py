"""Deterministic LC/RV data for fitting tests; synthesis remains real."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any

import numpy as np
from numpy.typing import NDArray

from elisa import const
from elisa import units as u
from elisa.analytics import LCData, RVData
from elisa.analytics.binary_fit import least_squares, mcmc
from elisa.analytics.binary_fit.shared import AbstractFit, AbstractLCFit
from elisa.analytics.models import lc, rv
from elisa.analytics.params import parameters as p
from elisa.binary_system.curves.community import RadialVelocitySystem
from elisa.binary_system.system import BinarySystem
from elisa.conf.enums import SamplingMethod
from elisa.observer.observer import Observer
from elisa.types import AstropyUnit

V, B = "Generic.Bessell.V", "Generic.Bessell.B"
LC_PHASES = [0.0, 0.04, 0.12, 0.25, 0.47, 0.5, 0.6, 0.8]
SECONDARY_PHASES = [0.0, 0.1, 0.3, 0.5, 0.7]
MANUAL_PHASES = [-0.25, 0.0, 0.04, 0.12, 0.25, 0.47, 0.5, 0.6, 0.8, 1.0, 1.25]
EQUAL_LENGTH_PHASES = [-0.25, 0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25]
BINARY_PARAMETERS = {
    "system": {
        "inclination": 85.0,
        "eccentricity": 0.0,
        "argument_of_periastron": 0.0,
        "period": 3.0,
        "mass_ratio": 0.7,
        "semi_major_axis": 10.0,
    },
    "primary": {
        "t_eff": 6500.0,
        "surface_potential": 6.0,
        "gravity_darkening": 0.32,
        "albedo": 0.5,
        "metallicity": 0.0,
    },
    "secondary": {
        "t_eff": 5000.0,
        "surface_potential": 6.0,
        "gravity_darkening": 0.32,
        "albedo": 0.5,
        "metallicity": 0.0,
    },
}


def normalized(fit: AbstractFit) -> NDArray[np.float64]:
    return np.asarray(p.vector_normalizer(fit.initial_vector, fit.fitable.keys(), fit.normalization))


def physical(fit: AbstractFit) -> dict[str, Any]:
    return p.prepare_properties_set(np.asarray(fit.initial_vector), fit.fitable.keys(), fit.constrained, fit.fixed)


def rv_problem(constrained: bool = False, *, outlier: bool = False) -> tuple[dict[str, Any], dict[str, RVData]]:
    """Independent real kinematic RV observations with different component grids."""
    truth = {
        "eccentricity": 0.15,
        "argument_of_periastron": 25.0,
        "period": 3.0,
        "mass_ratio": 0.7,
        "asini": 10.0,
        "gamma": 12000.0,
    }
    flat = {f"system@{key}": value for key, value in truth.items()}
    cfg: dict[str, Any] = {"system": {key: {"value": value, "fixed": True} for key, value in truth.items()}}
    cfg["system"]["gamma"] = {"value": 9000.0, "fixed": False, "min": 0.0, "max": 25000.0}
    if not outlier:
        cfg["system"]["asini"] = {"value": 9.0, "fixed": False, "min": 8.0, "max": 12.0}
        cfg["system"]["mass_ratio"] = (
            {"constraint": "system@asini * 0.07"}
            if constrained
            else {"value": 0.8, "fixed": False, "min": 0.5, "max": 1.0}
        )
    data = {}
    for component, xs in {
        "secondary": np.array([0.7, 0.0, 0.12, 0.3, 0.5, 0.12]),
        "primary": np.linspace(0.0, 0.98, 21),
    }.items():
        obs = Observer(passband="bolometric", system=None)
        obs.system_cls = RadialVelocitySystem
        curve = rv.central_rv_synthetic(xs, obs, **flat)[component]
        if outlier and component == "primary":
            curve = curve.copy()
            curve[5] += 20000.0
        data[component] = RVData(
            x_data=xs, y_data=curve, y_err=np.full(len(xs), 100.0), x_unit=u.dimensionless_unscaled, y_unit=u.m / u.s
        )
    return cfg, data


def lc_setup(fit: AbstractLCFit, data: dict[str, LCData]) -> None:
    cfg = {
        group: {key: {"value": value, "fixed": True} for key, value in values.items()}
        for group, values in deepcopy(BINARY_PARAMETERS).items()
    }
    cfg["system"]["inclination"] = {"value": 85.0, "fixed": False, "min": 80.0, "max": 90.0}
    fit.set_up(
        p.BinaryInitialParameters(**cfg),
        data,
        passband=data.keys(),
        discretization=10.0,
        interp_treshold=100,
        samples="uniform",
        observer_system_cls=BinarySystem,
    )


def make_lc(
    values: NDArray = (2.0, 4.0, 3.0),
    errors: NDArray | None = None,
    *,
    y_unit: AstropyUnit = u.dimensionless_unscaled,
    reference_magnitude: float | None = None,
) -> LCData:
    return LCData(
        x_data=np.arange(np.asarray(values).size) / 4.0,
        y_data=values,
        y_err=errors,
        x_unit=u.dimensionless_unscaled,
        y_unit=y_unit,
        reference_magnitude=reference_magnitude,
    )


def lc_parameters(standard: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    base = deepcopy(BINARY_PARAMETERS)
    if standard:
        total = (10 * u.solRad).to(u.m).value ** 3 * (2 * np.pi / (3 * 86400)) ** 2 / const.G / u.solMass.to(u.kg)
        del base["system"]["mass_ratio"], base["system"]["semi_major_axis"]
        base["primary"]["mass"], base["secondary"]["mass"] = total / 1.7, total * 0.7 / 1.7
    cfg = {g: {k: {"value": v, "fixed": True} for k, v in d.items()} for g, d in base.items()}
    cfg["system"]["inclination"] = {"value": 85.0, "fixed": False, "min": 80.0, "max": 90.0}
    cfg["primary"]["t_eff"] = {"value": 6500.0, "fixed": False, "min": 6000.0, "max": 7000.0}
    cfg["secondary"]["t_eff"] = {"constraint": "primary@t_eff * 0.7692307692307693"}
    cfg["nuisance"] = {"ln_f": {"value": -2.0, "fixed": True}}
    flat = {f"{g}@{k}": v for g, d in base.items() for k, v in d.items()}
    return cfg, flat


def synthesize_lc(xs: NDArray, bands: Iterable[str], physical: dict[str, Any]) -> dict[str, NDArray[np.float64]]:
    obs = Observer(passband=list(bands), system=None)
    obs.system_cls = BinarySystem
    return lc.synthetic_binary(np.asarray(xs), 10.0, obs, **physical)


def prepare_lc_fits(
    grids: Mapping[str, NDArray],
    threshold: int = 100,
    samples: SamplingMethod | str | list[float] | NDArray = SamplingMethod.UNIFORM,
    *,
    jd: bool = False,
    standard: bool = False,
    atmosphere: dict[str, str] | None = None,
    custom_ld: bool = False,
    free_noise: bool = False,
    missing_errors: bool = False,
) -> tuple[least_squares.DetachedLightCurveFit, mcmc.DetachedLightCurveFit]:
    cfg, _physical = lc_parameters(standard)
    if jd:
        cfg["system"]["primary_minimum_time"] = {"value": 2450000.0, "fixed": True}
    if free_noise:
        cfg["nuisance"]["ln_f"] = {"value": 0.0, "fixed": False, "min": -5.0, "max": 1.0}
    data = {}
    for band, xs in grids.items():
        xs = np.asarray(xs)
        raw = synthesize_lc(xs, [band], _physical)[band]
        data[band] = LCData(
            x_data=2450000.0 + 3 * np.asarray(xs) if jd else np.asarray(xs),
            y_data=raw * (1 + 0.002 * np.arange(len(xs))),
            y_err=None if missing_errors else np.full(len(xs), 0.01),
            x_unit=u.d if jd else u.dimensionless_unscaled,
            y_unit=u.dimensionless_unscaled,
        )
    ld = {c: {b: [0.1] for b in (*grids, "bolometric")} for c in ("primary", "secondary")} if custom_ld else None
    fits: list[Any] = []
    for cls in (least_squares.DetachedLightCurveFit, mcmc.DetachedLightCurveFit):
        fit: Any = cls()
        fit.atmosphere_model, fit.limb_darkening_coefficients = atmosphere, ld
        fit.set_up(
            p.BinaryInitialParameters(**deepcopy(cfg)),
            data,
            passband=data.keys(),
            discretization=10.0,
            interp_treshold=threshold,
            observer_system_cls=BinarySystem,
            samples=samples,
        )
        fits.append(fit)
    return fits[0], fits[1]
