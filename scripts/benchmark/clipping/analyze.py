"""Exact-rational audit of every captured face, independent of Pypex predicates.

Run after experiment.py; use the same PYTHONPATH=src:. and Python environment.
"""

import json
from collections.abc import Sequence
from fractions import Fraction as F
from typing import Literal, TypeAlias, overload
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from elisa.pypex.poly2d.polygon import Polygon

# Keep benchmark validation self-contained, independently of the test package.
FloatArray: TypeAlias = NDArray[np.float64]
RationalPoint: TypeAlias = tuple[F, F]


def signed_double_area(vertices: Sequence[RationalPoint]) -> F:
    """Accumulate a rational shoelace sum without an integer zero fallback."""
    total = F(0)
    for i, (x, y) in enumerate(vertices):
        previous_x, previous_y = vertices[i - 1]
        total += previous_x * y - previous_y * x
    return total


def _cross(first: RationalPoint, second: RationalPoint, point: RationalPoint) -> F:
    return (second[0] - first[0]) * (point[1] - first[1]) - (second[1] - first[1]) * (point[0] - first[0])


@overload
def exact_area(polygon: FloatArray, triangle: FloatArray, return_vertices: Literal[False] = False) -> float: ...


@overload
def exact_area(polygon: FloatArray, triangle: FloatArray, return_vertices: Literal[True]) -> list[RationalPoint]: ...


@overload
def exact_area(polygon: FloatArray, triangle: FloatArray, return_vertices: bool) -> float | list[RationalPoint]: ...


def exact_area(
    polygon: FloatArray,
    triangle: FloatArray,
    return_vertices: bool = False,
) -> float | list[RationalPoint]:
    """Exact rational oracle: clip TRIANGLE against POLYGON (opposite direction).

    Fraction(float) preserves the actual supplied binary64 input exactly.
    """
    polygon_vertices: list[RationalPoint] = [(F(float(x)), F(float(y))) for x, y in polygon]
    clipped_vertices: list[RationalPoint] = [(F(float(x)), F(float(y))) for x, y in triangle]
    sign = F(1) if signed_double_area(polygon_vertices) >= 0 else F(-1)
    for i in range(len(polygon_vertices)):
        output_vertices: list[RationalPoint] = []
        if not clipped_vertices:
            break
        edge_start, edge_end = polygon_vertices[i - 1], polygon_vertices[i]
        start = clipped_vertices[-1]
        fs = sign * _cross(edge_start, edge_end, start)
        for end in clipped_vertices:
            fe = sign * _cross(edge_start, edge_end, end)
            if (fs >= 0) != (fe >= 0):
                t = fs / (fs - fe)
                output_vertices.append((start[0] + t * (end[0] - start[0]), start[1] + t * (end[1] - start[1])))
            if fe >= 0:
                output_vertices.append(end)
            start, fs = end, fe
        clipped_vertices = output_vertices
    if return_vertices:
        return clipped_vertices
    return float(abs(signed_double_area(clipped_vertices)) / 2)


OUT = Path(__file__).resolve().parent


def shoelace(vertices):
    if len(vertices) < 3:
        return 0.0
    v = np.asarray(vertices, dtype=float)
    return 0.5 * abs(np.sum(v[:, 0] * np.roll(v[:, 1], -1) - v[:, 1] * np.roll(v[:, 0], -1)))


def main():
    data = np.load(OUT / "eclipse_geometry.npz")
    metrics = []
    for i in range(int(data["batches"])):
        p, triangles = data[f"h{i}"], data[f"t{i}"]
        hull = Polygon(p)
        for j, t in enumerate(triangles):
            q = exact_area(p, t, return_vertices=True)
            exact = float(abs(signed_double_area(q)) / 2)
            legacy = float(data[f"a_pypex{i}"][j])
            new = float(data[f"a_numba{i}"][j])
            rounded = shoelace(np.round(np.asarray(q, dtype=float), 10))
            high = hull.intersection(Polygon(t), round_tol=15)
            unrounded = high.surface_area() if high is not None else 0.0
            metrics.append(
                dict(
                    batch=i,
                    face=j,
                    exact=exact,
                    legacy_error=legacy - exact,
                    clipping_error=new - exact,
                    rounded_oracle_error=rounded - exact,
                    legacy_minus_rounded_oracle=legacy - rounded,
                    legacy_round15_error=unrounded - exact,
                )
            )
    result = dict(
        faces=len(metrics),
        max_abs={
            key: max(abs(x[key]) for x in metrics)
            for key in (
                "legacy_error",
                "clipping_error",
                "rounded_oracle_error",
                "legacy_minus_rounded_oracle",
                "legacy_round15_error",
            )
        },
        worst_clipping=max(metrics, key=lambda x: abs(x["clipping_error"])),
        worst_legacy=max(metrics, key=lambda x: abs(x["legacy_error"])),
        worst_rounding_residual=max(metrics, key=lambda x: abs(x["legacy_minus_rounded_oracle"])),
        all_faces=metrics,
    )
    (OUT / "exact_audit.json").write_text(json.dumps(result, indent=2))
    print({k: v for k, v in result.items() if k != "all_faces"})


if __name__ == "__main__":
    main()
