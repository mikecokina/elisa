"""RV fitting selects observed components without changing synthesis output."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from elisa import settings
from elisa.analytics import RVData
from elisa.analytics.binary_fit import least_squares, mcmc
from elisa.analytics.binary_fit.shared import map_rv_predictions, rv_r_squared
from elisa.analytics.models.rv import central_rv_synthetic
from elisa.analytics.params.parameters import BinaryInitialParameters
from elisa.base.error import PredictionAlignmentError
from elisa.binary_system.curves.community import RadialVelocitySystem
from elisa.binary_system.system import BinarySystem
from elisa.observer.observer import Observer
from tests.tests_analytics.fitting_helpers import normalized, physical, rv_problem


@pytest.fixture(
    params=[
        pytest.param(("primary",), id="primary-only"),
        pytest.param(("secondary",), id="secondary-only"),
        pytest.param(("secondary", "primary"), id="both-secondary-first"),
    ],
)
def observed_components(request: pytest.FixtureRequest) -> tuple[str, ...]:
    """Include reversed component order to distinguish it from synthesis order."""
    return request.param


@pytest.mark.parametrize("variant", ["community", "standard"])
def test_real_rv_predictions_residuals_and_likelihood_select_observed_components(
    observed_components: tuple[str, ...],
    variant: str,
) -> None:
    """Both fitter callbacks match independent per-dataset synthesis and costs."""
    config, all_data = rv_problem()
    if variant == "standard":
        del config["system"]["asini"], config["system"]["mass_ratio"]
        config["system"]["inclination"] = {"value": 85.0, "fixed": True}
        config["primary"] = {"mass": {"value": 2.0, "fixed": True}}
        config["secondary"] = {"mass": {"value": 1.0, "fixed": True}}
    config["nuisance"] = {"ln_f": {"value": -2.0, "fixed": True}}
    system_class = RadialVelocitySystem if variant == "community" else BinarySystem

    # Both datasets have unsorted phases and duplicates; their sampling differs.
    primary = all_data["primary"]
    indices = np.array([8, 0, 4, 8, 2])
    all_data["primary"] = RVData(
        x_data=primary.x_data[indices],
        y_data=primary.y_data[indices],
        y_err=primary.y_err[indices],
        x_unit=primary.x_unit,
        y_unit=primary.y_unit,
    )
    data = {component: all_data[component] for component in observed_components}
    lsq = least_squares.CentralRadialVelocity()
    posterior = mcmc.CentralRadialVelocity()
    with settings.override(RV_METHOD="kinematic"):
        for fit in (lsq, posterior):
            fit.set_up(BinaryInitialParameters(**config), data, observer_system_cls=system_class)
        posterior.prepare_priors()
        model_parameters = physical(lsq)

        observer = Observer(passband="bolometric", system=None)
        observer.system_cls = system_class
        # Synthesize each observation grid independently: no union or reducers.
        expected_predictions = {
            component: central_rv_synthetic(dataset.x_data, observer, **model_parameters)[component]
            for component, dataset in data.items()
        }
        raw = lsq.prepare_synthetic(normalized(lsq))
        assert set(raw) == {"primary", "secondary"}
        mapped = map_rv_predictions(raw, lsq.x_data_reduced, lsq.x_data_reducer)
        assert tuple(mapped) == observed_components
        for component in observed_components:
            assert mapped[component].shape == data[component].y_data.shape
            assert_allclose(mapped[component], expected_predictions[component], rtol=2e-14, atol=1e-8)

        expected_residuals = np.concatenate(
            [(dataset.y_data - expected_predictions[component]) / dataset.y_err for component, dataset in data.items()],
        )
        actual_residuals = lsq.central_rv_model_to_fit(normalized(lsq))
        assert actual_residuals.shape == expected_residuals.shape
        assert_allclose(actual_residuals, expected_residuals, rtol=2e-13, atol=1e-8)

        variance_scale = 1.0 + np.exp(-4.0)
        expected_likelihood = -0.5 * sum(
            np.sum(
                (dataset.y_data - expected_predictions[component]) ** 2 / (dataset.y_err**2 * variance_scale)
                + np.log(2.0 * np.pi * dataset.y_err**2 * variance_scale),
            )
            for component, dataset in data.items()
        )
        assert_allclose(posterior.likelihood(normalized(posterior)), expected_likelihood, rtol=2e-13, atol=1e-8)
        assert_allclose(posterior.ln_probability(normalized(posterior)), expected_likelihood, rtol=2e-13, atol=1e-8)

        total_variation = sum(np.sum((dataset.y_data - np.mean(dataset.y_data)) ** 2) for dataset in data.values())
        squared_error = sum(
            np.sum((dataset.y_data - expected_predictions[component]) ** 2) for component, dataset in data.items()
        )
        actual_r_squared = rv_r_squared(
            central_rv_synthetic,
            lsq.x_data_reduced,
            lsq.y_data,
            lsq.x_data_reducer,
            system_class,
            **model_parameters,
        )
        assert_allclose(actual_r_squared, 1.0 - squared_error / total_variation, rtol=2e-13, atol=1e-12)


def test_rv_mapping_ignores_extra_synthetic_components(observed_components: tuple[str, ...]) -> None:
    """Unobserved output is ignored, including its dimensions and values."""
    coordinates = np.array([0.0, 0.1, 0.2])
    reducers = {component: np.array([2, 0, 2]) for component in observed_components}
    synthetic = {"primary": np.array([10.0, 20.0, 30.0]), "secondary": np.array([-10.0, -20.0, -30.0])}
    synthetic["unobserved"] = np.array([np.nan])

    mapped = map_rv_predictions(synthetic, coordinates, reducers)

    assert tuple(mapped) == observed_components
    for component in observed_components:
        assert_array_equal(mapped[component], synthetic[component][[2, 0, 2]])


def test_rv_mapping_rejects_missing_required_components(observed_components: tuple[str, ...]) -> None:
    """Extra output cannot substitute for a required observation component."""
    coordinates = np.array([0.0, 0.1, 0.2])
    reducers = {component: np.array([2, 0, 2]) for component in observed_components}
    synthetic = {"primary": np.ones(3), "secondary": np.ones(3), "unobserved": np.ones(3)}
    del synthetic[observed_components[0]]

    with pytest.raises(PredictionAlignmentError, match="components"):
        map_rv_predictions(synthetic, coordinates, reducers)
