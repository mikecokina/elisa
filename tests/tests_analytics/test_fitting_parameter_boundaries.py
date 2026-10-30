"""Physical input failures retain their LSQ and MCMC handling at model boundaries."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from numpy.typing import NDArray

from elisa import const, units as u
from elisa.analytics import LCData, RVData
from elisa.analytics.binary_fit import least_squares, mcmc
from elisa.analytics.binary_fit.shared import AbstractFit, AbstractLCFit, AbstractRVFit
from elisa.analytics.models import lc, rv
from elisa.analytics.params.parameters import BinaryInitialParameters
from elisa.base.error import ElisaInitialParametersError, ElisaParametersError
from elisa.binary_system.curves.community import RadialVelocitySystem
from elisa.binary_system.system import BinarySystem
from elisa.binary_system.transform import RadialVelocityObserverProperties
from elisa.observer.observer import Observer
from tests.tests_analytics.fitting_helpers import V, lc_parameters, normalized, physical, rv_problem


INVALID_PHYSICAL_CONSTRAINTS = [
    pytest.param("mass_ratio", "-1", id="singular-negative-ratio"),
    pytest.param("mass_ratio", "0", id="zero-ratio"),
    pytest.param("mass_ratio", "-0.5", id="negative-ratio"),
    pytest.param("period", "0", id="zero-period"),
    pytest.param("period", "-1", id="negative-period"),
]
NONFINITE_CONSTRAINTS = [
    pytest.param("mass_ratio", "log(-1)", id="nan-ratio"),
    pytest.param("mass_ratio", "1e309", id="infinite-ratio"),
    pytest.param("period", "log(-1)", id="nan-period"),
    pytest.param("period", "1e309", id="infinite-period"),
]
INVALID_ASINI = [
    pytest.param(-1.0, ElisaParametersError, id="negative-asini"),
    pytest.param(np.nan, ElisaParametersError, id="nan-asini"),
    pytest.param(np.inf, ElisaParametersError, id="positive-infinite-asini"),
    pytest.param(-np.inf, ElisaParametersError, id="negative-infinite-asini"),
]


@pytest.fixture(
    params=[
        pytest.param(least_squares.DetachedLightCurveFit, id="lsq"),
        pytest.param(mcmc.DetachedLightCurveFit, id="mcmc"),
    ],
)
def lc_fit(request: pytest.FixtureRequest) -> AbstractLCFit:
    return request.param()


@pytest.fixture(
    params=[
        pytest.param(least_squares.CentralRadialVelocity, id="lsq"),
        pytest.param(mcmc.CentralRadialVelocity, id="mcmc"),
    ],
)
def rv_fit(request: pytest.FixtureRequest) -> AbstractRVFit:
    return request.param()


@pytest.fixture
def lc_inputs() -> tuple[dict[str, Any], dict[str, LCData]]:
    config, _ = lc_parameters()
    config["system"]["primary_minimum_time"] = {"value": 2450000.0, "fixed": True}
    config["system"]["period"] = {"constraint": "3.0"}
    config["system"]["mass_ratio"] = {"constraint": "0.7"}
    data = {
        V: LCData(
            x_data=np.array([2450000.0, 2450000.75, 2450001.5]),
            y_data=np.ones(3),
            y_err=np.full(3, 0.1),
            x_unit=u.d,
            y_unit=u.dimensionless_unscaled,
        ),
    }
    return config, data


@pytest.fixture
def rv_inputs() -> tuple[dict[str, Any], dict[str, RVData]]:
    return rv_problem()


def prepare_lc_fit(fit: AbstractLCFit, config: dict[str, Any], data: dict[str, LCData]) -> None:
    initial = BinaryInitialParameters(**config)
    initial.validate_lc_parameters("detached")
    fit.set_up(
        initial,
        data,
        passband=data.keys(),
        discretization=10.0,
        interp_treshold=100,
        samples="uniform",
        observer_system_cls=BinarySystem,
    )
    if isinstance(fit, mcmc.MCMCFit):
        fit.prepare_priors()


def prepare_rv_fit(fit: AbstractRVFit, config: dict[str, Any], data: dict[str, RVData]) -> None:
    initial = BinaryInitialParameters(**config)
    initial.validate_rv_parameters()
    fit.set_up(initial, data, observer_system_cls=RadialVelocitySystem)
    if isinstance(fit, mcmc.MCMCFit):
        fit.prepare_priors()


def construct_binary_without_surface_synthesis(
    phases: NDArray[np.float64], discretization: float, _observer: Observer, **model_parameters: Any,
) -> dict[str, NDArray[np.float64]]:
    """Keep real constructor validation while avoiding LC mesh and atmosphere work."""
    lc.prepare_binary(discretization=discretization, **model_parameters)
    return {V: np.ones(len(phases))}


@pytest.fixture
def lc_model_fit(
    lc_fit: AbstractLCFit,
    lc_inputs: tuple[dict[str, Any], dict[str, LCData]],
    monkeypatch: pytest.MonkeyPatch,
) -> AbstractLCFit:
    config, data = lc_inputs
    prepare_lc_fit(lc_fit, config, data)
    monkeypatch.setattr(lc, "synthetic_binary", construct_binary_without_surface_synthesis)
    return lc_fit


@pytest.fixture
def rv_model_fit(
    rv_fit: AbstractRVFit,
    rv_inputs: tuple[dict[str, Any], dict[str, RVData]],
    monkeypatch: pytest.MonkeyPatch,
) -> AbstractRVFit:
    config, data = rv_inputs
    config["system"]["primary_minimum_time"] = {"value": 2450000.0, "fixed": True}
    config["system"]["period"] = {"constraint": "3.0"}
    config["system"]["mass_ratio"] = {"constraint": "0.7"}
    for dataset in data.values():
        dataset.x_data = 2450000.0 + 3 * dataset.x_data
        dataset.x_unit = u.d
    prepare_rv_fit(rv_fit, config, data)
    return rv_fit


def evaluate_candidate(fit: AbstractFit) -> float | NDArray[np.float64]:
    vector = normalized(fit)
    if isinstance(fit, mcmc.MCMCFit):
        return float(fit.ln_probability(vector))
    if isinstance(fit, least_squares.LightCurveFit):
        return fit.model_to_fit(vector)
    assert isinstance(fit, least_squares.CentralRadialVelocity)
    return fit.central_rv_model_to_fit(vector)


def assert_physical_rejection(fit: AbstractFit) -> None:
    result = evaluate_candidate(fit)
    if isinstance(fit, mcmc.MCMCFit):
        assert result == -np.inf
        return

    observation_count = sum(fit.num_of_points.values())
    expected_penalty = np.full(observation_count, np.sqrt(const.MAX_USABLE_FLOAT / observation_count))
    assert_array_equal(result, expected_penalty)


@pytest.mark.parametrize(("parameter", "constraint"), INVALID_PHYSICAL_CONSTRAINTS)
def test_lc_model_physical_failure_preserves_candidate_rejection(
    lc_model_fit: AbstractLCFit, parameter: str, constraint: str,
) -> None:
    lc_model_fit.constrained[f"system@{parameter}"].constraint = constraint

    assert_physical_rejection(lc_model_fit)


@pytest.mark.parametrize(("parameter", "constraint"), INVALID_PHYSICAL_CONSTRAINTS)
def test_rv_model_physical_failure_preserves_candidate_rejection(
    rv_model_fit: AbstractRVFit, parameter: str, constraint: str,
) -> None:
    rv_model_fit.constrained[f"system@{parameter}"].constraint = constraint

    assert_physical_rejection(rv_model_fit)


@pytest.mark.parametrize(("parameter", "constraint"), NONFINITE_CONSTRAINTS)
def test_lc_model_nonfinite_parameter_rejects_candidate(
    lc_model_fit: AbstractLCFit, parameter: str, constraint: str,
) -> None:
    lc_model_fit.constrained[f"system@{parameter}"].constraint = constraint

    with np.errstate(invalid="ignore"):
        assert_physical_rejection(lc_model_fit)


@pytest.mark.parametrize(("parameter", "constraint"), NONFINITE_CONSTRAINTS)
def test_rv_model_nonfinite_parameter_rejects_candidate(
    rv_model_fit: AbstractRVFit, parameter: str, constraint: str,
) -> None:
    rv_model_fit.constrained[f"system@{parameter}"].constraint = constraint

    with np.errstate(invalid="ignore"):
        assert_physical_rejection(rv_model_fit)


@pytest.mark.parametrize(
    "asini_definition",
    [
        pytest.param({"value": 0.0, "fixed": True}, id="fixed-zero-asini"),
        pytest.param({"value": 0.0, "fixed": False, "min": 0.0, "max": 10.0}, id="free-zero-asini"),
        pytest.param({"constraint": "0 * system@gamma"}, id="constrained-zero-asini"),
    ],
)
def test_zero_asini_is_accepted_and_produces_constant_systemic_velocities(
    rv_fit: AbstractRVFit,
    rv_inputs: tuple[dict[str, Any], dict[str, RVData]],
    asini_definition: dict[str, Any],
) -> None:
    config, data = rv_inputs
    config["system"]["asini"] = deepcopy(asini_definition)
    prepare_rv_fit(rv_fit, config, data)

    result = evaluate_candidate(rv_fit)

    assert np.all(np.isfinite(result))
    observation_count = sum(rv_fit.num_of_points.values())
    rejection_residual = np.sqrt(const.MAX_USABLE_FLOAT / observation_count)
    assert np.max(np.abs(result)) < rejection_residual

    model_parameters = physical(rv_fit)
    curves = rv.central_rv_synthetic(rv_fit.x_data_reduced, rv_fit.observer, **model_parameters)
    expected_velocities = np.full(rv_fit.x_data_reduced.size, model_parameters["system@gamma"])
    for curve in curves.values():
        assert_array_equal(curve, expected_velocities)


@pytest.mark.parametrize("fixed", [True, False], ids=["fixed", "free"])
@pytest.mark.parametrize(("asini", "expected_error"), INVALID_ASINI)
def test_invalid_asini_metadata_uses_the_initial_exception_type(
    rv_inputs: tuple[dict[str, Any], dict[str, RVData]],
    fixed: bool,
    asini: float,
    expected_error: type[Exception],
) -> None:
    config, _ = rv_inputs
    config["system"]["asini"] = {"value": asini, "fixed": fixed, "min": 0.0, "max": 10.0}

    with pytest.raises(ElisaInitialParametersError) as caught:
        BinaryInitialParameters(**config)

    assert type(caught.value.__cause__) is expected_error


@pytest.mark.parametrize(("asini", "expected_error"), INVALID_ASINI)
def test_model_asini_validator_exposes_the_model_exception_type(
    asini: float, expected_error: type[Exception],
) -> None:
    with pytest.raises(expected_error) as caught:
        RadialVelocityObserverProperties.asini(asini)

    assert type(caught.value) is expected_error


def test_model_asini_validator_accepts_zero() -> None:
    assert RadialVelocityObserverProperties.asini(0.0) == 0.0


def test_negative_constrained_asini_is_a_physical_rejection(
    rv_fit: AbstractRVFit, rv_inputs: tuple[dict[str, Any], dict[str, RVData]],
) -> None:
    config, data = rv_inputs
    config["system"]["asini"] = {"constraint": "-1"}
    prepare_rv_fit(rv_fit, config, data)

    assert_physical_rejection(rv_fit)


@pytest.mark.parametrize(
    "constraint",
    [
        pytest.param("1e309", id="positive-infinite-asini"),
        pytest.param("-1e309", id="negative-infinite-asini"),
        pytest.param("log(-1)", id="nan-asini"),
    ],
)
def test_nonfinite_constrained_asini_rejects_candidate(
    rv_fit: AbstractRVFit, rv_inputs: tuple[dict[str, Any], dict[str, RVData]], constraint: str,
) -> None:
    config, data = rv_inputs
    config["system"]["asini"] = {"constraint": constraint}
    prepare_rv_fit(rv_fit, config, data)

    with np.errstate(invalid="ignore"):
        assert_physical_rejection(rv_fit)


def test_lc_input_validation_rejects_a_constrained_mass_that_conversion_would_overwrite(
    lc_fit: AbstractLCFit, lc_inputs: tuple[dict[str, Any], dict[str, LCData]],
) -> None:
    config, data = lc_inputs
    config["primary"]["mass"] = {"constraint": "-1"}
    with pytest.raises(ElisaInitialParametersError, match="LC parameters must use either"):
        prepare_lc_fit(lc_fit, config, data)
