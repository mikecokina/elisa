"""Sutherland-Hodgman area-only clipping; no coordinate quantization or fastmath.

Inputs are finite, cyclically ordered convex polygons and triangles in 2D.
Either winding is accepted, including repeated/collinear boundary vertices.
An exactly zero float64 triangle determinant returns zero; no epsilon widens
half-planes. Extremely ill-conditioned or overflowing coordinates are outside
ordinary float64 accuracy. Malformed arrays raise ElisaValueError;
nonfinite coordinates and results raise ElisaNumericError.
"""

# Geometric dimensions and vertex counts are explicit in the clipping formulas.
# ruff: noqa: PLR2004
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit

from elisa.base.error import (
    ElisaNumericError,
    _register_numba_error_helpers,
    raise_elisa_numeric_error,
    raise_elisa_value_error,
)

_register_numba_error_helpers()

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.types import Float


def _clipping_coordinates(values: NDArray[Float]) -> NDArray[Float]:
    """Convert coordinates while retaining malformed-array diagnostics."""
    try:
        coordinates = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise_elisa_value_error("Clipping coordinates must be numeric arrays", exc)
    return coordinates


def _inputs(
    polygon: NDArray[Float],
    triangle: NDArray[Float],
) -> tuple[NDArray[Float], NDArray[Float]]:
    polygon = _clipping_coordinates(polygon)
    triangle = _clipping_coordinates(triangle)
    polygon_has_xy_vertices = polygon.ndim == 2 and polygon.shape[1] == 2
    triangle_has_three_xy_vertices = triangle.shape == (3, 2)
    if not (polygon_has_xy_vertices and triangle_has_three_xy_vertices):
        message = "Expected polygon (n, 2) and triangle (3, 2)"
        raise_elisa_value_error(message)

    polygon_is_finite = np.all(np.isfinite(polygon))
    triangle_is_finite = np.all(np.isfinite(triangle))
    if not (polygon_is_finite and triangle_is_finite):
        raise_elisa_numeric_error("Coordinates must be finite")
    return polygon, triangle


def _side(point: NDArray[Float], a: NDArray[Float], direction: NDArray[Float], orientation: Float) -> Float:
    value = orientation * (direction[0] * (point[1] - a[1]) - direction[1] * (point[0] - a[0]))
    if not np.isfinite(value):
        message = "Half-plane determinant overflow"
        raise_elisa_numeric_error(message)
    return value


def convex_polygon_triangle_intersection_area(  # noqa: C901 - three clipping stages and numeric guards
    polygon: NDArray[Float],
    triangle: NDArray[Float],
) -> Float:
    """Clip a convex polygon against three triangle edges (reference version).

    Local coordinates and a translated shoelace sum avoid cancellation from
    absolute coordinate offsets. Convexity and cyclic ordering are preconditions.
    """
    polygon, triangle = _inputs(polygon, triangle)
    origin = triangle[0].copy()
    triangle = triangle - origin
    vertices = list(polygon - origin)
    determinant = triangle[1, 0] * triangle[2, 1] - triangle[1, 1] * triangle[2, 0]
    if not np.isfinite(determinant):
        message = "Triangle determinant overflow"
        raise_elisa_numeric_error(message)
    if determinant == 0.0 or len(vertices) < 3:
        return 0.0
    orientation = 1.0 if determinant > 0.0 else -1.0
    for i in range(3):
        if not vertices:
            return 0.0
        a, b = triangle[i], triangle[(i + 1) % 3]
        d = b - a

        output = []
        start = vertices[-1]
        fs = _side(start, a, d, orientation)
        for end in vertices:
            fe = _side(end, a, d, orientation)
            if (fs >= 0.0) != (fe >= 0.0):
                denominator = fs - fe
                if not np.isfinite(denominator):
                    message = "Crossing denominator overflow"
                    raise_elisa_numeric_error(message)
                output.append(start + (fs / denominator) * (end - start))
            if fe >= 0.0:
                output.append(end)
            start, fs = end, fe
        vertices = output
    if len(vertices) < 3:
        return 0.0
    local = np.asarray(vertices) - vertices[0]
    area = 0.5 * abs(sum(local[i - 1, 0] * local[i, 1] - local[i - 1, 1] * local[i, 0] for i in range(1, len(local))))
    if not np.isfinite(area):
        message = "Nonfinite clipping area"
        raise_elisa_numeric_error(message)
    return area


@njit(cache=True)
def _compiled_area(polygon: NDArray[Float], triangle: NDArray[Float]) -> Float:  # noqa: C901, PLR0912
    """Scalar-buffer equivalent of the reference; fastmath is disabled."""
    n = len(polygon)
    if n < 3:
        return 0.0
    tri = triangle - triangle[0]
    det = tri[1, 0] * tri[2, 1] - tri[1, 1] * tri[2, 0]
    if not np.isfinite(det):
        message = "Triangle determinant overflow"
        raise_elisa_numeric_error(message)
    if det == 0.0:
        return 0.0
    sign = 1.0 if det > 0.0 else -1.0
    # Each clipping plane can at most double the vertex count even with
    # redundant vertices; 8*n safely bounds all three stages.
    source = np.empty((8 * n, 2), dtype=np.float64)
    target = np.empty_like(source)
    source[:n] = polygon - triangle[0]
    for edge in range(3):
        if n == 0:
            return 0.0
        ax, ay = tri[edge, 0], tri[edge, 1]
        dx = tri[(edge + 1) % 3, 0] - ax
        dy = tri[(edge + 1) % 3, 1] - ay
        sx, sy = source[n - 1, 0], source[n - 1, 1]
        fs = sign * (dx * (sy - ay) - dy * (sx - ax))
        if not np.isfinite(fs):
            message = "Half-plane determinant overflow"
            raise_elisa_numeric_error(message)
        count = 0
        for j in range(n):
            ex, ey = source[j, 0], source[j, 1]
            fe = sign * (dx * (ey - ay) - dy * (ex - ax))
            if not np.isfinite(fe):
                message = "Half-plane determinant overflow"
                raise_elisa_numeric_error(message)
            if (fs >= 0.0) != (fe >= 0.0):
                denominator = fs - fe
                if not np.isfinite(denominator):
                    message = "Crossing denominator overflow"
                    raise_elisa_numeric_error(message)
                t = fs / denominator
                target[count, 0] = sx + t * (ex - sx)
                target[count, 1] = sy + t * (ey - sy)
                count += 1
            if fe >= 0.0:
                target[count, 0], target[count, 1] = ex, ey
                count += 1
            sx, sy, fs = ex, ey, fe
        source, target = target, source
        n = count
    area = 0.0
    if n < 3:
        return area
    ox, oy = source[0, 0], source[0, 1]
    for j in range(1, n):
        area += (source[j - 1, 0] - ox) * (source[j, 1] - oy) - (source[j - 1, 1] - oy) * (source[j, 0] - ox)
    return 0.5 * abs(area)


def optimized_intersection_area(polygon: NDArray[Float], triangle: NDArray[Float]) -> Float:
    """Validate float64 inputs and evaluate the compiled scalar kernel."""
    polygon, triangle = _inputs(polygon, triangle)
    area = _compiled_area(polygon, triangle)
    if not np.isfinite(area):
        message = "Nonfinite clipping area"
        raise_elisa_numeric_error(message)
    return area


@njit(cache=True)
def _compiled_areas(polygon: NDArray[Float], triangles: NDArray[Float]) -> NDArray[Float]:
    areas = np.empty(len(triangles), dtype=np.float64)
    for i in range(len(triangles)):
        areas[i] = _compiled_area(polygon, triangles[i])
    return areas


def _validated_intersection_inputs(
    polygon: NDArray[Float],
    triangles: NDArray[Float],
) -> tuple[NDArray[Float], NDArray[Float]]:
    """Convert and validate polygon/triangle coordinates once per face batch."""
    polygon = _clipping_coordinates(polygon)
    triangles = _clipping_coordinates(triangles)
    polygon_has_xy_vertices = polygon.ndim == 2 and polygon.shape[1] == 2
    triangles_have_three_xy_vertices = triangles.ndim == 3 and triangles.shape[1:] == (3, 2)
    if not (polygon_has_xy_vertices and triangles_have_three_xy_vertices):
        message = "Expected polygon (n, 2) and triangles (m, 3, 2)"
        raise_elisa_value_error(message)

    polygon_is_finite = np.all(np.isfinite(polygon))
    triangles_are_finite = np.all(np.isfinite(triangles))
    if not (polygon_is_finite and triangles_are_finite):
        raise_elisa_numeric_error("Coordinates must be finite")
    return polygon, triangles


def _validate_intersection_areas(areas: NDArray[Float], face_count: int) -> None:
    """Require one finite nonnegative intersection area per input face."""
    areas_match_faces = areas.shape == (face_count,)
    areas_are_finite = np.all(np.isfinite(areas))
    areas_are_nonnegative = np.all(areas >= 0)
    if not (areas_match_faces and areas_are_finite and areas_are_nonnegative):
        raise_elisa_numeric_error("Sutherland-Hodgman must return one finite nonnegative area per face")


def sutherland_hodgman_intersection_areas(
    polygon: NDArray[Float],
    triangles: NDArray[Float],
    *,
    compiled: bool = True,
) -> NDArray[Float]:
    """Compute Sutherland-Hodgman areas, validating once per face batch.

    :param polygon: Cyclically ordered convex polygon with shape (n, 2).
    :param triangles: Selected face coordinates with shape (m, 3, 2).
    :param compiled: Use the Numba kernel rather than the Python reference.
    :returns: One nonnegative finite area per face, in the same order.
    :raises ElisaValueError: If coordinate arrays have invalid types or shapes.
    :raises ElisaNumericError: If coordinates or computed areas are unusable.
    """
    polygon, triangles = _validated_intersection_inputs(polygon, triangles)

    try:
        if compiled:
            areas = _compiled_areas(polygon, triangles)
        else:
            areas = np.array(
                [convex_polygon_triangle_intersection_area(polygon, triangle) for triangle in triangles],
                dtype=np.float64,
            )
    except ElisaNumericError:
        raise
    except FloatingPointError as exc:
        raise_elisa_numeric_error("Sutherland-Hodgman clipping arithmetic failed", exc)

    _validate_intersection_areas(areas, face_count=len(triangles))
    return areas
