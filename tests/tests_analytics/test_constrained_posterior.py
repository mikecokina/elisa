"""Direct NumPy posterior references; no MCMC sampling or forward synthesis."""

from __future__ import annotations

import json
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from numpy.typing import ArrayLike

from elisa.analytics.binary_fit import io_tools, mcmc
from elisa.analytics.binary_fit.lc_fit import LCFitMCMC
from elisa.analytics.binary_fit.mixins import MCMCMixin
from elisa.analytics.binary_fit.rv_fit import RVFitMCMC
from elisa.analytics.params import parameters
from elisa.analytics.params.parameters import BinaryInitialParameters, InitialParameter
from elisa.analytics.params.transform import BinaryInitialProperties
from elisa.base.error import ElisaMonteCarloError, ElisaInitialParametersError
from tests.tests_analytics.fitting_helpers import BINARY_PARAMETERS, V, make_lc, rv_problem

GAMMA = "system@gamma"
MASS_RATIO = "system@mass_ratio"
ASINI = "system@asini"
NORMALIZATION = {GAMMA: (-4.0, 4.0)}
GAMMA_SAMPLES = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 3.5])
NORMALIZED_CHAIN = ((GAMMA_SAMPLES + 4.0) / 8.0)[:, None]


def posterior_parameters(expression: str = "system@gamma**2 + system@period") -> BinaryInitialParameters:
    return BinaryInitialParameters(
        system={
            "gamma": {"value": 0.0, "fixed": False, "min": -4.0, "max": 4.0},
            "period": {"value": 2.0, "fixed": True},
            "mass_ratio": {"constraint": expression},
        }
    )


def summarize_constraint(
    values: ArrayLike, expression: str, percentiles: Sequence[float] | None = None
) -> dict[str, Any]:
    init = posterior_parameters(expression)
    return MCMCMixin.resolve_constrained_result(
        np.asarray(values, dtype=float),
        init.get_fitable(),
        NORMALIZATION,
        init.get_constrained(),
        init.get_fixed(),
        None if percentiles is None else list(percentiles),
    )[MASS_RATIO]


def assert_posterior_percentiles(
    summary: dict[str, Any],
    samples: ArrayLike,
    percentiles: Sequence[float] = (16, 50, 84),
) -> None:
    expected = np.percentile(np.asarray(samples, dtype=float), percentiles)
    assert_allclose(
        [summary["confidence_interval"]["min"], summary["value"], summary["confidence_interval"]["max"]],
        expected,
        rtol=0,
        atol=1e-14,
    )


@pytest.mark.parametrize(
    ("expression", "expected_samples", "percentiles"),
    [
        pytest.param("3*system@gamma + system@period", 3 * GAMMA_SAMPLES + 2, (16, 50, 84), id="linear-fixed-offset"),
        pytest.param(
            "system@gamma**2 + system@period", GAMMA_SAMPLES**2 + 2, (16, 50, 84), id="quadratic-fixed-offset"
        ),
        pytest.param("system@gamma**2", GAMMA_SAMPLES**2, (5, 50, 95), id="custom-percentiles"),
    ],
)
def test_constraint_summary_propagates_each_joint_sample(expression, expected_samples, percentiles) -> None:
    result = summarize_constraint(NORMALIZED_CHAIN, expression, percentiles)
    assert_posterior_percentiles(result, expected_samples, percentiles)
    assert result["unit"] == posterior_parameters().get_constrained()[MASS_RATIO].to_dict()["unit"]


def test_nonlinear_summary_differs_from_constraint_at_marginal_median() -> None:
    result = summarize_constraint(NORMALIZED_CHAIN, "system@gamma**2 + system@period")
    assert result["value"] == 6.0
    assert np.median(GAMMA_SAMPLES) ** 2 + 2 == 2.25


def test_joint_correlations_and_multiple_constraints() -> None:
    labels = {GAMMA: posterior_parameters().get_fitable()[GAMMA], ASINI: posterior_parameters().get_fitable()[GAMMA]}
    # Perfect anticorrelation: summing marginal intervals would give a false width.
    chain = np.column_stack((NORMALIZED_CHAIN[:, 0], 1 - NORMALIZED_CHAIN[:, 0]))
    constraints = {
        MASS_RATIO: InitialParameter(
            BinaryInitialProperties, param="mass_ratio", constraint=f"{GAMMA}+{ASINI}", unit=None
        ),
        "system@eccentricity": InitialParameter(
            BinaryInitialProperties, param="eccentricity", constraint=f"{GAMMA}*{ASINI}", unit=None
        ),
    }
    before = chain.copy()
    chain.setflags(write=False)
    results = MCMCMixin.resolve_constrained_result(
        chain, labels, {GAMMA: (-4.0, 4.0), ASINI: (-4.0, 4.0)}, constraints, {}
    )
    assert_posterior_percentiles(results[MASS_RATIO], np.zeros(len(GAMMA_SAMPLES)))
    assert_posterior_percentiles(results["system@eccentricity"], -(GAMMA_SAMPLES**2))
    assert_array_equal(chain, before)


def test_constraint_to_constraint_references_are_rejected() -> None:
    # Preserve this existing limitation rather than adding a new dependency engine.
    constraints = {MASS_RATIO: f"{GAMMA}**2", ASINI: f"{MASS_RATIO}+1"}
    with pytest.raises(ElisaInitialParametersError, match="non-variable parameter"):
        parameters.prepare_properties_set(np.array([-1.0]), [GAMMA], constraints, {})


def test_negative_power_substitution_and_constant_samples() -> None:
    chain = np.full((5, 1), 3 / 8)  # physical gamma = -1
    with np.errstate(all="raise"):
        result = summarize_constraint(chain, "system@gamma**2")
        free = MCMCMixin.resolve_mcmc_result(chain, posterior_parameters().get_fitable(), NORMALIZATION)[GAMMA]
    assert_posterior_percentiles(result, np.ones(5))
    assert_posterior_percentiles(free, -np.ones(5))
    exact = 0.123456789012345
    chain = np.full((5, 1), (exact + 4) / 8)
    assert_posterior_percentiles(
        MCMCMixin.resolve_mcmc_result(chain, posterior_parameters().get_fitable(), NORMALIZATION)[GAMMA],
        np.full(5, exact),
    )


@pytest.mark.parametrize("expression", ["exp(1000*system@gamma)", "log(system@gamma)"])
def test_nonfinite_constraints_reject_instead_of_dropping_samples(expression) -> None:
    with (
        np.errstate(over="ignore", invalid="ignore", divide="ignore"),
        pytest.raises(ElisaMonteCarloError, match="Nonfinite posterior constraint"),
    ):
        summarize_constraint(NORMALIZED_CHAIN, expression)


@pytest.mark.parametrize("structured", [False, True])
def test_result_chain_reload_discard_filter_and_reserialize(
    tmp_path: Path, mcmc_result: LCFitMCMC | RVFitMCMC, structured: bool
) -> None:
    fit = mcmc_result
    init = posterior_parameters()
    base = MCMCMixin.resolve_mcmc_result(NORMALIZED_CHAIN, init.get_fitable(), NORMALIZATION)
    base.update(
        {
            key: {"value": item.value, "fixed": True, "unit": item.to_dict()["unit"]}
            for key, item in init.get_fixed().items()
        }
    )
    base[MASS_RATIO] = summarize_constraint(NORMALIZED_CHAIN, "system@gamma**2 + system@period")
    fit.set_result(parameters.serialize_result(base))
    assert fit.flat_result is not None
    assert_posterior_percentiles(fit.flat_result[MASS_RATIO], GAMMA_SAMPLES**2 + 2)
    result_path = tmp_path / "result.json"
    fit.save_result(result_path)
    fit.load_result(result_path)
    assert fit.flat_result is not None
    assert_posterior_percentiles(fit.flat_result[MASS_RATIO], GAMMA_SAMPLES**2 + 2)
    payload = {
        "flat_chain": NORMALIZED_CHAIN.tolist(),
        "fitable_parameters": [GAMMA],
        "normalization": NORMALIZATION,
        "fitable": {key: val.to_dict() for key, val in init.get_fitable().items()},
    }
    if structured:
        payload.update(chain_shape=[4, 2, 1], chain_order="step,walker,parameter")
    path = tmp_path / "chain.json"
    path.write_text(json.dumps(payload))
    fit.load_chain(str(path), discard=1)
    retained = GAMMA_SAMPLES[2 if structured else 1 :]
    assert fit.flat_result is not None
    assert_posterior_percentiles(fit.flat_result[MASS_RATIO], retained**2 + 2)
    io_tools.filter_chain(fit, **{GAMMA: (-1.5, 3.1)})
    retained = retained[(retained > -1.5) & (retained < 3.1)]
    assert fit.flat_result is not None
    assert_posterior_percentiles(fit.flat_result[MASS_RATIO], retained**2 + 2)
    fit.save_result(result_path)
    fit.load_result(result_path)
    assert fit.flat_result is not None
    assert_posterior_percentiles(fit.flat_result[MASS_RATIO], retained**2 + 2)


@pytest.mark.parametrize("fit_cls", [mcmc.DetachedLightCurveFit, mcmc.CentralRadialVelocity])
def test_fit_completion_uses_samplewise_summary(fit_cls) -> None:
    data: dict
    if fit_cls is mcmc.DetachedLightCurveFit:
        config = {
            group: {name: {"value": value, "fixed": True} for name, value in values.items()}
            for group, values in BINARY_PARAMETERS.items()
        }
        data = {V: make_lc(errors=[0.1, 0.1, 0.1])}
    else:
        config, data = rv_problem()
        for metadata in config["system"].values():
            metadata["fixed"] = True
    config["system"].update(
        {
            "gamma": {"value": 0.0, "fixed": False, "min": -4.0, "max": 4.0},
            "period": {"value": 2.0, "fixed": True},
            "mass_ratio": {"constraint": "system@gamma**2 + system@period"},
        }
    )
    initial = BinaryInitialParameters(**config)
    fit = fit_cls()
    sampler = Mock()
    sampler.get_chain.return_value = NORMALIZED_CHAIN
    # Sampling and synthesis for the final R² are the only expensive boundaries.
    with (
        patch.object(fit_cls, "_fit", return_value=sampler),
        patch.object(fit, "prediction_r_squared", return_value=1.0)
        if fit_cls is mcmc.DetachedLightCurveFit
        else nullcontext(),
        patch.object(mcmc, "rv_r_squared", return_value=1.0),
    ):
        result = fit.fit(x0=initial, data=data, nsteps=4, save=False)
    assert_posterior_percentiles(parameters.deserialize_result(result)[MASS_RATIO], GAMMA_SAMPLES**2 + 2)


@pytest.mark.parametrize("missing", ["flat_result", "flat_chain", "normalization"])
def test_update_rejects_missing_state(missing) -> None:
    fit = RVFitMCMC()
    fit.flat_result = {}
    fit.flat_chain = NORMALIZED_CHAIN
    fit.normalization = NORMALIZATION
    setattr(fit, missing, None)
    with pytest.raises(ElisaMonteCarloError, match="requires loaded results, chain and normalization"):
        io_tools._update_solution(fit, {}, None)


@pytest.mark.parametrize("value", [-2.0, -1.0, 0.0, 0.5, 2.0])
def test_constraint_substitution_preserves_negative_power_precedence(value: float) -> None:
    constraints = {"secondary@t_eff": "primary@t_eff ** 2 + sin(system@period)"}
    result = parameters.prepare_properties_set(
        np.array([value]),
        ["primary@t_eff"],
        constraints,
        {"system@period": 3.0},
    )
    assert_allclose(result["secondary@t_eff"], value**2 + np.sin(3.0), rtol=2e-14)
