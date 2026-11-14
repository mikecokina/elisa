"""Sutherland-Hodgman adapters preserve numerical and coverage failure semantics."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest

from elisa import settings
from elisa.base.error import ElisaNumericError, ElisaUnexpectedError
from elisa.base.surface.coverage import surface_area_coverage
from elisa.binary_system.surface import clipping, coverage
from elisa.conf.enums import IntersectionBackend
from tests.tests_pypex.test_pypex_coverage_errors import HULL, TRIANGLES

if TYPE_CHECKING:
    from numpy.typing import NDArray

    import elisa.conf.enums


def test_standalone_shape_error_is_builtin_and_coverage_preserves_its_cause() -> None:
    with pytest.raises(ValueError, match="Expected polygon") as original:
        clipping.convex_polygon_triangle_intersection_area(HULL, np.zeros((2, 2)))
    assert type(original.value) is ValueError

    with (
        settings.override(INTERSECTION_BACKEND=IntersectionBackend.REFERENCE),
        patch.object(clipping, "convex_polygon_triangle_intersection_area", side_effect=original.value),
        pytest.raises(ElisaNumericError) as translated,
    ):
        coverage._calculate_intersection_areas(HULL, TRIANGLES)
    assert translated.value.__cause__ is original.value


@pytest.mark.parametrize("cosine", [0.0, np.nan, np.inf])
def test_partial_coverage_rejects_unusable_projection(cosine: float) -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    faces = np.array([[0, 1, 2]])
    with (
        patch.object(coverage, "_calculate_intersection_areas", return_value=np.array([0.1])),
        patch.object(coverage.utils, "calculate_cos_theta_los_x", return_value=np.array([cosine])),
        pytest.raises(ElisaNumericError),
    ):
        coverage.partial_visible_faces_surface_coverage(points, faces, np.zeros((1, 3)), HULL)


def test_partial_coverage_preserves_finite_projected_area_correction() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    faces = np.array([[0, 1, 2]])
    with (
        patch.object(coverage, "_calculate_intersection_areas", return_value=np.array([0.1])),
        patch.object(coverage.utils, "calculate_cos_theta_los_x", return_value=np.array([0.5])),
    ):
        result = coverage.partial_visible_faces_surface_coverage(points, faces, np.zeros((1, 3)), HULL)
    np.testing.assert_allclose(result, [0.8])


def test_degenerate_eclipse_outline_rejects_candidate_with_original_cause() -> None:
    from scipy.spatial import QhullError

    with pytest.raises(ElisaNumericError) as caught:
        coverage.get_eclipse_boundary_path(np.zeros((4, 2)))
    assert isinstance(caught.value.__cause__, QhullError)


def test_missing_prepared_partial_coverage_is_a_contract_failure() -> None:
    with pytest.raises(ElisaUnexpectedError, match="coverage computed before"):
        surface_area_coverage(1, np.array([], dtype=int), np.array([]), partial=np.array([0]))


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_coordinates_are_numeric_failures(compiled: bool, value: float) -> None:
    triangles = TRIANGLES.copy()
    triangles[0, 0, 0] = value
    with pytest.raises(ElisaNumericError, match="finite"):
        clipping.sutherland_hodgman_intersection_areas(HULL, triangles, compiled=compiled)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("triangles", [np.zeros((3, 2)), np.zeros((1, 4, 2)), np.zeros((1, 3, 3))])
def test_standalone_shape_errors_defensively_reject_candidate(compiled: bool, triangles: NDArray[np.generic]) -> None:
    with pytest.raises(ElisaNumericError, match="Expected polygon"):
        clipping.sutherland_hodgman_intersection_areas(HULL, triangles, compiled=compiled)


@pytest.mark.parametrize("backend", [IntersectionBackend.REFERENCE, IntersectionBackend.SUTHERLAND_HODGMAN])
@pytest.mark.parametrize("category", [TypeError, ValueError, IndexError, RuntimeError])
def test_coverage_converts_computation_failures_into_candidate_numeric_errors(
    backend: elisa.conf.enums.IntersectionBackend,
    category: type[IndexError | RuntimeError | TypeError | ValueError],
) -> None:
    failure = category("clipping failed")
    with (
        settings.override(INTERSECTION_BACKEND=backend),
        patch.object(clipping, "sutherland_hodgman_intersection_areas", side_effect=failure),
        pytest.raises(ElisaNumericError, match="Sutherland-Hodgman failed") as caught,
    ):
        coverage._calculate_intersection_areas(HULL, TRIANGLES)
    assert caught.value.__cause__ is failure


@pytest.mark.parametrize("backend", [IntersectionBackend.REFERENCE, IntersectionBackend.SUTHERLAND_HODGMAN])
def test_coverage_preserves_existing_numeric_error(backend: elisa.conf.enums.IntersectionBackend) -> None:
    cause = OverflowError("original overflow")
    failure = ElisaNumericError("nonfinite clipping")
    failure.__cause__ = cause
    with (
        settings.override(INTERSECTION_BACKEND=backend),
        patch.object(clipping, "sutherland_hodgman_intersection_areas", side_effect=failure),
        pytest.raises(ElisaNumericError) as caught,
    ):
        coverage._calculate_intersection_areas(HULL, TRIANGLES)
    assert caught.value is failure
    assert caught.value.__cause__ is cause


@pytest.mark.parametrize("compiled", [False, True])
def test_batch_normalizes_native_floating_point_errors(compiled: bool) -> None:
    target = "_compiled_areas" if compiled else "convex_polygon_triangle_intersection_area"
    failure = FloatingPointError("arithmetic overflow")
    with patch.object(clipping, target, side_effect=failure), pytest.raises(ElisaNumericError) as caught:
        clipping.sutherland_hodgman_intersection_areas(HULL, TRIANGLES, compiled=compiled)
    assert caught.value.__cause__ is failure


@pytest.mark.parametrize("areas", [np.array([np.nan]), np.array([np.inf]), np.array([-1.0]), np.zeros(2)])
def test_batch_rejects_invalid_kernel_results(areas: NDArray[np.generic]) -> None:
    with (
        patch.object(clipping, "_compiled_areas", return_value=areas),
        pytest.raises(ElisaNumericError, match="one finite nonnegative"),
    ):
        clipping.sutherland_hodgman_intersection_areas(HULL, TRIANGLES)


def test_batch_conversion_errors_keep_original_cause() -> None:
    with pytest.raises(ElisaNumericError, match="numeric arrays") as caught:
        clipping.sutherland_hodgman_intersection_areas(HULL, [[["bad", 0], [1, 0], [0, 1]]])
    assert isinstance(caught.value.__cause__, ValueError)


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), SystemExit(1)])
def test_coverage_does_not_swallow_process_control(failure: KeyboardInterrupt | SystemExit) -> None:
    with (
        settings.override(INTERSECTION_BACKEND=IntersectionBackend.SUTHERLAND_HODGMAN),
        patch.object(clipping, "sutherland_hodgman_intersection_areas", side_effect=failure),
        pytest.raises(type(failure)) as caught,
    ):
        coverage._calculate_intersection_areas(HULL, TRIANGLES)
    assert caught.value is failure
