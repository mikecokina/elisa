"""Independent analytic and exact-rational area validation."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction as F
from pathlib import Path
from typing import Any, TypeAlias
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

from elisa import settings
from elisa.binary_system.surface import coverage
from elisa.binary_system.surface.clipping import (
    convex_polygon_triangle_intersection_area as reference,
)
from elisa.binary_system.surface.clipping import (
    intersection_areas,
)
from elisa.binary_system.surface.clipping import (
    optimized_intersection_area as optimized,
)
from elisa.binary_system.surface.coverage import _calculate_intersection_areas
from elisa.conf.enums import IntersectionBackend
from tests.utils import exact_area, signed_double_area

FloatArray: TypeAlias = NDArray[np.float64]
Coordinates: TypeAlias = FloatArray | Sequence[Sequence[int | float | F]]
AreaKernel: TypeAlias = Callable[[FloatArray, FloatArray], float | np.floating[Any]]
BatchAreaKernel: TypeAlias = Callable[[FloatArray, FloatArray], FloatArray]


SQUARE: FloatArray = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


@dataclass(frozen=True)
class KnownAreaCase:
    name: str
    polygon: FloatArray
    triangle: FloatArray
    expected: F
    explanation: str
    intersection_vertices: int
    feature_width: float | None = None


def _known_case(
    name: str,
    polygon: Coordinates,
    triangle: Coordinates,
    expected: int | F,
    explanation: str,
    vertices: int,
    feature_width: float | None = None,
) -> KnownAreaCase:
    return KnownAreaCase(
        name,
        np.asarray(polygon, dtype=np.float64),
        np.asarray(triangle, dtype=np.float64),
        F(expected),
        explanation,
        vertices,
        feature_width,
    )


# These expectations are literal analytic results, never computed by exact_area,
# Pypex, or either production kernel. The first eight geometries also supply the
# original scale/rotation tests below instead of duplicating their definitions.
NARROW_WIDTH = 2.0**-30
KNOWN_AREA_CASES: list[KnownAreaCase] = [
    _known_case(
        "identical-right-triangles",
        [[0, 0], [1, 0], [0, 1]],
        [[0, 0], [1, 0], [0, 1]],
        F(1, 2),
        "The identical right triangles have base and height 1: area = 1*1/2.",
        3,
    ),
    _known_case(
        "separated-square-and-triangle",
        SQUARE,
        [[2, 0], [3, 0], [2, 1]],
        0,
        "The square has x <= 1 and the triangle has x >= 2; no intersection.",
        0,
    ),
    _known_case(
        "square-inside-triangle-with-boundary-contact",
        SQUARE,
        [[0, 0], [2, 0], [0, 2]],
        1,
        "The entire unit square satisfies x >= 0, y >= 0, x+y <= 2; area = 1.",
        4,
    ),
    _known_case(
        "triangle-inside-square-with-boundary-contact",
        SQUARE,
        [[0, 0], [1, 0], [0, 1]],
        F(1, 2),
        "The triangle is one diagonal half of the unit square; area = 1/2.",
        3,
    ),
    _known_case(
        "square-strictly-inside-triangle",
        SQUARE,
        [[-1, -1], [4, -1], [-1, 4]],
        1,
        "All square points satisfy x > -1, y > -1, x+y < 3; area = 1.",
        4,
    ),
    _known_case(
        "contact-along-entire-edge",
        SQUARE,
        [[1, 0], [2, 0], [1, 1]],
        0,
        "The common set is the entire segment x=1, 0<=y<=1, with zero area.",
        2,
    ),
    _known_case(
        "contact-at-one-vertex",
        SQUARE,
        [[1, 1], [2, 1], [1, 2]],
        0,
        "The common set is the single vertex (1,1), with zero area.",
        1,
    ),
    _known_case(
        "partial-collinear-edge-overlap-quadrilateral",
        SQUARE,
        [[F(1, 2), 0], [F(3, 2), 0], [F(1, 2), 1]],
        F(3, 8),
        "A width-1/2 trapezoid has vertical sides 1 and 1/2: area = (1/2)*(1+1/2)/2.",
        4,
    ),
    _known_case(
        "triangle-strictly-inside-square",
        [[0, 0], [4, 0], [4, 4], [0, 4]],
        [[1, 1], [2, 1], [1, 3]],
        1,
        "The interior right triangle has base 1 and height 2: area = 1*2/2.",
        3,
    ),
    _known_case(
        "partially-overlapping-triangles",
        [[0, 0], [4, 0], [0, 4]],
        [[1, 1], [5, 1], [1, 5]],
        2,
        "The overlap is x>=1, y>=1, x+y<=4: a right triangle with legs 2, area = 2.",
        3,
    ),
    _known_case(
        "triangle-crosses-square-no-triangle-vertex-inside",
        SQUARE,
        [[-1, F(1, 2)], [2, F(1, 2)], [F(1, 2), 2]],
        F(1, 2),
        "All triangle vertices are outside. The overlap is [0,1] x [1/2,1], area = 1/2.",
        4,
    ),
    _known_case(
        "square-crosses-triangle-no-square-vertex-inside",
        [[0, 0], [2, 0], [2, 2], [0, 2]],
        [[-1, F(1, 2)], [3, F(1, 2)], [1, F(3, 2)]],
        F(3, 2),
        "All square vertices are outside. Triangle area 2 minus two exterior right triangles of area 1/4 gives 3/2.",
        5,
    ),
    _known_case(
        "crossing-with-no-vertices-inside-either-operand",
        [[-2, F(-1, 4)], [2, F(-1, 4)], [2, F(1, 4)], [-2, F(1, 4)]],
        [[0, -2], [1, 1], [-1, 1]],
        F(2, 3),
        "A height-1/2 trapezoid has widths 7/6 and 3/2: area = (1/2)*(7/6+3/2)/2 = 2/3.",
        4,
    ),
    _known_case(
        "very-narrow-positive-strip",
        SQUARE,
        [[1 - NARROW_WIDTH, 0], [2 - NARROW_WIDTH, 0], [1 - NARROW_WIDTH, 1]],
        F(1, 2**30) - F(1, 2**61),
        "For d=2^-30, the width-d trapezoid has heights 1 and 1-d: area = d-d^2/2.",
        4,
        NARROW_WIDTH,
    ),
    _known_case(
        "very-small-positive-trapezoid",
        SQUARE,
        [[1 - NARROW_WIDTH, 0], [1 + NARROW_WIDTH, 0], [1 - NARROW_WIDTH, NARROW_WIDTH]],
        F(3, 2**62),
        "For d=2^-30, the width-d trapezoid has heights d and d/2: area = 3*d^2/4.",
        4,
        NARROW_WIDTH,
    ),
    _known_case(
        "small-triangular-overlap-at-square-corner",
        SQUARE,
        [[F(3, 4), 1], [1, F(3, 4)], [F(5, 4), F(5, 4)]],
        F(1, 32),
        "The corner overlap has vertices (3/4,1), (1,3/4), (1,1): legs 1/4, area = 1/32.",
        3,
    ),
    _known_case(
        "square-triangle-pentagonal-overlap",
        [[0, 0], [2, 0], [2, 2], [0, 2]],
        [[-2, -2], [5, -2], [-2, 5]],
        F(7, 2),
        "The line x+y=3 removes a unit-leg right triangle from a square of area 4: area = 4-1/2.",
        5,
    ),
    _known_case(
        "convex-pentagon-triangle-partial-overlap",
        [[0, 0], [4, 0], [4, 2], [2, 4], [0, 4]],
        [[0, 0], [5, 0], [0, 5]],
        F(23, 2),
        "The overlap is [0,4]^2 with x+y<=5; subtract a leg-3 right triangle: area = 16-9/2.",
        5,
    ),
    _known_case(
        "convex-hexagon-triangle-heptagonal-overlap",
        [[0, 0], [3, 0], [4, 1], [4, 4], [1, 4], [0, 3]],
        [[-10, -10], [17, -10], [-10, 17]],
        F(29, 2),
        "The hexagon is a 4x4 square minus two unit-leg corner triangles (area 15). x+y<=7 removes one more: 29/2.",
        7,
    ),
    _known_case(
        "opposing-triangles-hexagonal-overlap",
        [[0, 0], [4, 0], [2, 4]],
        [[0, 3], [2, -1], [4, 3]],
        F(21, 4),
        "Split at y=3/2 into two trapezoids, each height 3/2 with widths 1 and 5/2: area = 21/4.",
        6,
    ),
]

# Keep the original rotation/scale and degeneracy coverage. Shared simple cases
# use the same named definitions as the direct analytic tests below.
CASES = [
    pytest.param(case.polygon, case.triangle, float(case.expected), id=case.name) for case in KNOWN_AREA_CASES[:8]
] + [
    pytest.param(SQUARE, [[1 - 2**-24, 0], [2, 0], [1 - 2**-24, 1]], None, id="small-overlap-scale-stability"),
    pytest.param(SQUARE, [[0, 0], [1, 2**-40], [0, 1]], None, id="nearly-parallel-scale-stability"),
    pytest.param(SQUARE, [[0, 0], [1, 0], [2, 0]], 0, id="collinear-degenerate-triangle"),
    pytest.param(SQUARE, [[0, 0], [0, 0], [0, 0]], 0, id="repeated-point-degenerate-triangle"),
    pytest.param(SQUARE, [[0, 0], [2, 0], [0, 2 - 2**-24]], None, id="nearly-complete-scale-stability"),
]


@pytest.fixture(name="known_area_backend", params=["reference", "numba-scalar", "numba-batch"])
def _known_area_backend(request: pytest.FixtureRequest) -> BatchAreaKernel:
    """Give every backend the same multi-triangle inputs and analytic targets."""
    if request.param == "numba-batch":

        def compiled_batch(polygon: FloatArray, triangles: FloatArray) -> FloatArray:
            return np.asarray(intersection_areas(polygon, triangles), dtype=np.float64)

        return compiled_batch
    kernel = reference if request.param == "reference" else optimized

    def scalar_batch(polygon: FloatArray, triangles: FloatArray) -> FloatArray:
        return np.array([kernel(polygon, triangle) for triangle in triangles], dtype=np.float64)

    return scalar_batch


@pytest.mark.parametrize("case", KNOWN_AREA_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize(
    "polygon_cw,triangle_cw",
    [
        pytest.param(False, False, id="ccw-ccw"),
        pytest.param(False, True, id="ccw-cw"),
        pytest.param(True, False, id="cw-ccw"),
        pytest.param(True, True, id="cw-cw"),
    ],
)
@pytest.mark.parametrize(
    "offset",
    [
        pytest.param((0, 0), id="original"),
        pytest.param((-4, 3), id="translated"),
    ],
)
def test_explicit_known_intersection_areas(
    case: KnownAreaCase,
    known_area_backend: BatchAreaKernel,
    polygon_cw: bool,
    triangle_cw: bool,
    offset: tuple[int, int],
) -> None:
    polygon = (case.polygon[::-1] if polygon_cw else case.polygon) + np.asarray(offset)
    triangle = (case.triangle[::-1] if triangle_cw else case.triangle) + np.asarray(offset)
    triangles = np.stack([np.roll(triangle, start, axis=0) for start in range(3)])
    for polygon_start in range(len(polygon)):
        actual = known_area_backend(np.roll(polygon, polygon_start, axis=0), triangles)
        message = f"{case.name}, polygon start={polygon_start}: {case.explanation}"
        assert actual.shape == (3,), message
        _assert_known_area(actual, case, message)


def _assert_known_area(actual: FloatArray | np.float64, case: KnownAreaCase, message: str) -> None:
    # Local coordinates remove the common offset. Ordinary cases allow only a
    # small operation-count budget relative to the exact analytic area, with no
    # absolute tolerance that could accept a positive area as zero.
    relative_budget = 32.0 * float(np.finfo(np.float64).eps)
    if case.feature_width is not None:
        # A feature of width d among O(L) coordinates has relative conditioning
        # O(eps*L/d). This budget is derived before evaluating either kernel.
        extent = float(max(np.ptp(case.polygon, axis=0).max(), np.ptp(case.triangle, axis=0).max()))
        relative_budget *= extent / case.feature_width
    if case.expected > 0:
        assert np.all(actual > 0), message
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        float(case.expected),
        rtol=relative_budget,
        atol=0.0,
        err_msg=message,
    )


@pytest.mark.parametrize("reverse_batch", [False, True], ids=["forward", "reverse"])
def test_explicit_mixed_numba_batch(reverse_batch: bool) -> None:
    """Check ordering and independent results in a batch of different geometries."""
    cases = [case for case in KNOWN_AREA_CASES if np.array_equal(case.polygon, SQUARE)]
    if reverse_batch:
        cases.reverse()
    actual = np.asarray(intersection_areas(SQUARE, np.stack([case.triangle for case in cases])), dtype=np.float64)
    assert actual.shape == (len(cases),)
    for result, case in zip(actual, cases):
        _assert_known_area(result, case, case.explanation)


@pytest.mark.parametrize("kernel", [reference, optimized])
@pytest.mark.parametrize("p,t,known", CASES)
def test_analytic_and_transformations(
    kernel: AreaKernel,
    p: FloatArray,
    t: Coordinates,
    known: float | int | None,
) -> None:
    triangle = np.asarray(t, dtype=np.float64)
    expected = exact_area(p, triangle)
    if known is not None:
        assert expected == known
    for scale in (1e-9, 1.0, 1e9):
        for angle in (0.0, 0.37):
            r = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            for reverse in (False, True):
                pp = np.roll(p[::-1] if reverse else p, 1, axis=0) @ r * scale
                tt = (triangle[::-1] if reverse else triangle) @ r * scale
                offset = np.array([3.0, -7.0]) * scale
                pp += offset
                tt += offset
                oracle = exact_area(pp, tt)
                # Forward error budget: O(n) operations on coordinates of scale L;
                # absolute area error O(eps*n*L^2), not relative to tiny overlap.
                budget = 64 * np.finfo(float).eps * (len(p) + 3) * scale**2
                actual = kernel(pp, tt)
                assert abs(actual - oracle) <= budget
                assert actual >= 0
                assert (
                    actual
                    <= min(
                        float(abs(signed_double_area([(F(float(x)), F(float(y))) for x, y in pp])) / 2),
                        abs(np.linalg.det(tt[1:] - tt[0])) / 2,
                    )
                    + budget
                )


@pytest.mark.parametrize("seed", range(12))
def test_many_vertices_exact_oracle(seed: int) -> None:
    rng = np.random.default_rng(seed)
    angles = np.sort(rng.uniform(0, 2 * np.pi, 120))
    p = np.column_stack((np.cos(angles), np.sin(angles)))
    t = rng.uniform(-1.5, 1.5, (3, 2))
    oracle = exact_area(p, t)
    for kernel in (reference, optimized):
        assert abs(kernel(p, t) - oracle) < 64 * np.finfo(float).eps * len(p)


def test_tiny_positive() -> None:
    delta = 2.0**-35
    t = np.array([[1 - delta, 0], [2, 0], [1 - delta, delta]])
    expected = exact_area(SQUARE, t)
    # Absolute coordinate error is measured in ulps near 1; dividing by
    # feature width gives the relative conditioning of this tiny overlap.
    for kernel in (reference, optimized):
        assert kernel(SQUARE, t) > 0
        assert kernel(SQUARE, t) == pytest.approx(expected, rel=4 * np.spacing(1.0) / delta, abs=0)


def test_invalid_and_empty() -> None:
    for kernel in (reference, optimized):
        with pytest.raises(ValueError):
            kernel(SQUARE, np.full((3, 2), np.nan))
        with pytest.raises(ValueError):
            kernel(SQUARE, np.zeros((4, 2)))
        assert kernel(np.empty((0, 2)), SQUARE[:3]) == 0
    assert intersection_areas(SQUARE, np.empty((0, 3, 2))).shape == (0,)


def test_captured_geometry() -> None:
    data = np.load(Path(__file__).parents[1] / "data/clipping/eclipse_geometry.npz")
    for i in range(int(data["batches"])):
        p, t = data[f"h{i}"], data[f"t{i}"]
        compiled_areas = intersection_areas(p, t)
        reference_areas = intersection_areas(p, t, compiled=False)
        np.testing.assert_allclose(compiled_areas, reference_areas, rtol=0.0, atol=64.0 * float(np.finfo(float).eps))
        if len(compiled_areas) == 0:
            continue
        # Exact independent oracle for both smallest overlap and largest mismatch.
        selected_indices = {
            int(np.argmax(abs(compiled_areas - reference_areas))),
            int(np.argmin(np.where(compiled_areas > 0, compiled_areas, np.inf))),
        }
        for face_index in selected_indices:
            assert abs(compiled_areas[face_index] - exact_area(p, t[face_index])) < 64 * np.finfo(float).eps * len(p)


@pytest.mark.parametrize("kernel", [reference, optimized])
def test_nearly_parallel_thin_overlap(kernel: AreaKernel) -> None:
    d = 2.0**-36
    t = np.array([[0, 1 + d], [1, 1 - d], [1, 2.0]])
    expected = exact_area(SQUARE, t)
    assert expected > 0
    assert kernel(SQUARE, t) == pytest.approx(expected, rel=4 * np.spacing(1.0) / d, abs=0)


@pytest.mark.parametrize("kernel", [reference, optimized])
def test_redundant_vertices_and_large_offset(kernel: AreaKernel) -> None:
    p = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    t = np.array([[0.2, 0.1], [1.2, 0.3], [0.1, 1.3]])
    for offset in (np.zeros(2), np.array([1e9, -1e9])):
        pp, tt = p + offset, t + offset
        # Coordinate subtraction incurs O(eps*offset), and area sensitivity
        # multiplies it by the extent (not by the offset a second time).
        budget = 64 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(offset))))
        assert abs(kernel(pp, tt) - exact_area(pp, tt)) <= budget


def test_private_integration_replay() -> None:
    data = np.load(Path(__file__).parents[1] / "data/clipping/eclipse_geometry.npz")
    assert settings.INTERSECTION_BACKEND == IntersectionBackend.SUTHERLAND_HODGMAN
    first_batch_args = [data[f"{key}0"] for key in ("p", "f", "n", "h")]
    for batch_index in range(int(data["batches"])):
        batch_args = [data[f"{key}{batch_index}"] for key in ("p", "f", "n", "h")]
        with patch.object(settings, "INTERSECTION_BACKEND", IntersectionBackend.REFERENCE):
            reference_coverage = coverage.partial_visible_faces_surface_coverage(*batch_args)
        with patch.object(settings, "INTERSECTION_BACKEND", IntersectionBackend.SUTHERLAND_HODGMAN):
            compiled_coverage = coverage.partial_visible_faces_surface_coverage(*batch_args)
        np.testing.assert_array_equal(reference_coverage, compiled_coverage)
    with pytest.raises(ValueError), patch.object(settings, "INTERSECTION_BACKEND", "invalid"):
        coverage.partial_visible_faces_surface_coverage(*first_batch_args)


@pytest.mark.parametrize("kernel", [reference, optimized])
def test_overflow_is_reported(kernel: AreaKernel) -> None:
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(FloatingPointError):
        kernel(SQUARE, np.array([[0.0, 0.0], [1e308, 0.0], [0.0, 1e308]]))


@pytest.mark.parametrize("backend", list(IntersectionBackend))
@pytest.mark.parametrize("as_string", [False, True], ids=["enum-setting", "string-setting"])
def test_configured_backend_preserves_intersection_areas(backend: IntersectionBackend, as_string: bool) -> None:
    triangles = np.array(
        [
            [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9]],
            [[0.5, 0.2], [1.5, 0.2], [0.5, 1.2]],
            [[2.0, 2.0], [3.0, 2.0], [2.0, 3.0]],
        ]
    )
    # Fully inside, clipped on two sides (1/2 - 1/8 - 1/50), and disjoint.
    expected_areas = np.array([0.32, 0.355, 0.0])
    selected_backend = backend.value if as_string else backend

    with settings.override(INTERSECTION_BACKEND=selected_backend):
        actual_areas = _calculate_intersection_areas(SQUARE, triangles)

    assert actual_areas.shape == expected_areas.shape
    assert actual_areas.dtype == np.float64
    np.testing.assert_allclose(actual_areas, expected_areas, rtol=0.0, atol=1e-14)
