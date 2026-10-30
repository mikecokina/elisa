"""Sutherland-Hodgman adapters preserve numerical and coverage failure semantics."""
from unittest.mock import patch

import numpy as np
import pytest

from elisa import settings
from elisa.base.error import ElisaNumericError, ElisaValueError
from elisa.binary_system.surface import clipping, coverage
from elisa.conf.enums import IntersectionBackend
from tests.test_pypex_coverage_errors import HULL, TRIANGLES


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_coordinates_are_numeric_failures(compiled, value):
    triangles = TRIANGLES.copy()
    triangles[0, 0, 0] = value
    with pytest.raises(ElisaNumericError, match="finite"):
        clipping.sutherland_hodgman_intersection_areas(HULL, triangles, compiled=compiled)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("triangles", [np.zeros((3, 2)), np.zeros((1, 4, 2)), np.zeros((1, 3, 3))])
def test_standalone_shape_errors_keep_contract_category(compiled, triangles):
    with pytest.raises(ElisaValueError, match="Expected polygon"):
        clipping.sutherland_hodgman_intersection_areas(HULL, triangles, compiled=compiled)


@pytest.mark.parametrize("backend", [IntersectionBackend.REFERENCE, IntersectionBackend.SUTHERLAND_HODGMAN])
@pytest.mark.parametrize("category", [TypeError, ValueError, IndexError, RuntimeError])
def test_coverage_converts_computation_failures_into_candidate_numeric_errors(backend, category):
    failure = category("clipping failed")
    with settings.override(INTERSECTION_BACKEND=backend):
        with patch.object(clipping, "sutherland_hodgman_intersection_areas", side_effect=failure):
            with pytest.raises(ElisaNumericError, match="Sutherland-Hodgman failed") as caught:
                coverage._calculate_intersection_areas(HULL, TRIANGLES)
    assert caught.value.__cause__ is failure


@pytest.mark.parametrize("backend", [IntersectionBackend.REFERENCE, IntersectionBackend.SUTHERLAND_HODGMAN])
def test_coverage_preserves_existing_numeric_error(backend):
    cause = OverflowError("original overflow")
    failure = ElisaNumericError("nonfinite clipping")
    failure.__cause__ = cause
    with settings.override(INTERSECTION_BACKEND=backend):
        with patch.object(clipping, "sutherland_hodgman_intersection_areas", side_effect=failure):
            with pytest.raises(ElisaNumericError) as caught:
                coverage._calculate_intersection_areas(HULL, TRIANGLES)
    assert caught.value is failure
    assert caught.value.__cause__ is cause


@pytest.mark.parametrize("compiled", [False, True])
def test_batch_normalizes_native_floating_point_errors(compiled):
    target = "_compiled_areas" if compiled else "convex_polygon_triangle_intersection_area"
    failure = FloatingPointError("arithmetic overflow")
    with patch.object(clipping, target, side_effect=failure):
        with pytest.raises(ElisaNumericError) as caught:
            clipping.sutherland_hodgman_intersection_areas(HULL, TRIANGLES, compiled=compiled)
    assert caught.value.__cause__ is failure


@pytest.mark.parametrize("areas", [np.array([np.nan]), np.array([np.inf]), np.array([-1.0]), np.zeros(2)])
def test_batch_rejects_invalid_kernel_results(areas):
    with patch.object(clipping, "_compiled_areas", return_value=areas):
        with pytest.raises(ElisaNumericError, match="one finite nonnegative"):
            clipping.sutherland_hodgman_intersection_areas(HULL, TRIANGLES)


def test_batch_conversion_errors_keep_original_cause():
    with pytest.raises(ElisaValueError, match="numeric arrays") as caught:
        clipping.sutherland_hodgman_intersection_areas(HULL, [[["bad", 0], [1, 0], [0, 1]]])
    assert isinstance(caught.value.__cause__, ValueError)


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), SystemExit(1)])
def test_coverage_does_not_swallow_process_control(failure):
    with settings.override(INTERSECTION_BACKEND=IntersectionBackend.SUTHERLAND_HODGMAN):
        with patch.object(clipping, "sutherland_hodgman_intersection_areas", side_effect=failure):
            with pytest.raises(type(failure)) as caught:
                coverage._calculate_intersection_areas(HULL, TRIANGLES)
    assert caught.value is failure
