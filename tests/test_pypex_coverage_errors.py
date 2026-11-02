"""PYPEX keeps builtin errors; ELISA coverage assigns numeric failure semantics."""
from unittest.mock import patch

import numpy as np
import pytest

from elisa import const, settings
from elisa.analytics.binary_fit import least_squares
from elisa.analytics.binary_fit.failure_policy import SYNTHESIS_FAILURES
from elisa.base.error import ElisaError, ElisaNumericError
from elisa.binary_system import utils as binary_utils
from elisa.binary_system.surface import coverage
from elisa.conf.enums import IntersectionBackend
from elisa.pypex.base.shape import Shape2D
from elisa.pypex.poly2d.intersection.sat import separating_axis_theorem
from elisa.pypex.poly2d.point import Point, is_point_in_polygon
from tests.tests_analytics.fitting_helpers import V, lc_setup, make_lc, normalized
from tests.tests_analytics.test_mcmc_posterior import make_quadratic_posterior

HULL = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
TRIANGLES = np.array([[[0.1, 0.1], [0.9, 0.1], [0.1, 0.9]]])
STAGES = ["hull_to_pypex_poly", "faces_to_pypex_poly", "pypex_poly_hull_intersection", "pypex_poly_surface_area"]


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("category", [ValueError, TypeError, IndexError, RuntimeError, FloatingPointError, ZeroDivisionError])
def test_pypex_pipeline_errors_become_numeric_only_at_coverage_boundary(stage, category):
    failure = category("geometry failed")
    with settings.override(INTERSECTION_BACKEND=IntersectionBackend.PYPEX):
        with patch.object(binary_utils, stage, side_effect=failure):
            with pytest.raises(ElisaNumericError, match="PyPex failed") as caught:
                coverage._calculate_intersection_areas(HULL, TRIANGLES)
    assert caught.value.__cause__ is failure
    assert isinstance(caught.value, SYNTHESIS_FAILURES)


@pytest.mark.parametrize("areas", [[np.nan], [np.inf], [-1.0], [], [0.0, 0.0], [[0.0]]])
def test_unusable_pypex_results_become_numeric_failures(areas):
    with settings.override(INTERSECTION_BACKEND=IntersectionBackend.PYPEX):
        with patch.object(binary_utils, "pypex_poly_surface_area", return_value=areas):
            with pytest.raises(ElisaNumericError, match="one finite nonnegative"):
                coverage._calculate_intersection_areas(HULL, TRIANGLES)


def test_missing_intersection_is_valid_zero_area():
    with settings.override(INTERSECTION_BACKEND=IntersectionBackend.PYPEX):
        areas = coverage._calculate_intersection_areas(HULL, TRIANGLES + 2.0)
    np.testing.assert_array_equal(areas, [0.0])


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), SystemExit(1)])
def test_process_control_exceptions_do_not_become_numeric_failures(failure):
    with settings.override(INTERSECTION_BACKEND=IntersectionBackend.PYPEX):
        with patch.object(binary_utils, "hull_to_pypex_poly", side_effect=failure):
            with pytest.raises(type(failure)) as caught:
                coverage._calculate_intersection_areas(HULL, TRIANGLES)
    assert caught.value is failure


@pytest.mark.parametrize("operation", [
    lambda: Shape2D.validity_check([[0.0, 0.0]], 3),
    lambda: separating_axis_theorem([[0.0, 0.0]], [[0.0, 0.0]]),
    lambda: is_point_in_polygon(Point(0.0, 0.0), [[0.0, 0.0]]),
])
def test_standalone_pypex_validation_still_raises_builtin_value_error(operation):
    with pytest.raises(ValueError) as caught:
        operation()
    assert type(caught.value) is ValueError
    assert not isinstance(caught.value, ElisaError)


def test_other_intersection_backends_are_outside_the_pypex_error_boundary():
    failure = TypeError("reference backend failure")
    with settings.override(INTERSECTION_BACKEND=IntersectionBackend.REFERENCE):
        with patch("elisa.binary_system.surface.clipping.sutherland_hodgman_intersection_areas", side_effect=failure):
            with pytest.raises(ElisaNumericError) as caught:
                coverage._calculate_intersection_areas(HULL, TRIANGLES)
    assert caught.value.__cause__ is failure


def test_pypex_failure_produces_lsq_penalty_and_mcmc_rejection_then_recovers():
    lsq = least_squares.DetachedLightCurveFit()
    lc_setup(lsq, {V: make_lc(errors=[0.1, 0.1, 0.1])})
    posterior = make_quadratic_posterior()
    failure = ValueError("failed PYPEX geometry")

    def failed_prediction(*args, **kwargs):
        return coverage._calculate_intersection_areas(HULL, TRIANGLES)

    with settings.override(INTERSECTION_BACKEND=IntersectionBackend.PYPEX):
        with patch.object(binary_utils, "pypex_poly_hull_intersection", side_effect=failure):
            with patch.object(lsq, "predict_light_curves", side_effect=failed_prediction):
                residuals = lsq.model_to_fit(normalized(lsq))
            assert residuals.shape == (3,)
            assert np.all(np.isfinite(residuals))
            np.testing.assert_allclose(np.sum(residuals**2), const.MAX_USABLE_FLOAT)
            with patch.object(posterior, "likelihood", side_effect=failed_prediction):
                assert posterior.ln_probability(np.array([0.5])) == -np.inf
    category = f"{ElisaNumericError.__module__}.{ElisaNumericError.__qualname__}"
    assert posterior._candidate_failure_diagnostics.counts[category] == 1
    assert posterior.ln_probability(np.array([0.5])) == -0.25
    with patch.object(lsq, "predict_light_curves", return_value={V: lsq.y_data[V]}):
        np.testing.assert_array_equal(lsq.model_to_fit(normalized(lsq)), np.zeros(3))
