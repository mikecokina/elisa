"""Opt-in angular sweep against independent high-precision first intersections."""

# ruff: noqa: INP001, T201
import argparse
import itertools
import json
from math import pi
from pathlib import Path

import numpy as np

from elisa.base.error import ElisaNumericError
from elisa.binary_system.radius import calculate_radius
from scripts.diagnostics.roche_reference import angular_reference

RADIUS_RTOL = 1e-7


def sweep() -> dict:
    """Exercise small/periodic angles, both components and critical neighborhoods."""
    angles = [
        (1e-12, pi / 2),
        (0.01, pi / 2),
        (0.1, pi / 2),
        (0.5, 1.2),
        (1.2, 0.3),
        (pi - 0.01, pi / 2),
        (float(np.nextafter(2 * pi, 0.0)), pi / 2),
    ]
    failures, max_error, count = [], 0.0, 0
    for q, spin, d, component, (phi, theta) in itertools.product(
        [0.001, 0.01, 1.0, 10.0, 1000.0],
        [0.3, 1.0, 3.0],
        [0.4, 1.0, 1.6],
        ["primary", "secondary"],
        angles,
    ):
        critical = angular_reference(q, spin, d, component, phi, theta)
        base = float(critical["critical"])
        targets = [base + 0.1, base + 1e-10, float(np.nextafter(base, np.inf)), base - 1e-5]
        for omega in targets:
            count += 1
            expected = angular_reference(q, spin, d, component, phi, theta, omega)
            case = [q, spin, d, component, phi, theta, omega]
            try:
                obtained = calculate_radius(spin, q, omega, component, d, phi, theta)
            except ElisaNumericError as exc:
                if expected["root"] is not None:
                    failures.append({"case": case, "error": str(exc)})
                continue
            if expected["root"] is None:
                failures.append({"case": case, "error": "Accepted absent root"})
                continue
            root = float(expected["root"])
            relative = abs(obtained - root) / root
            max_error = max(max_error, relative)
            stationary = expected["stationary"]
            if relative > RADIUS_RTOL or (stationary is not None and obtained > float(stationary) + 2e-14 * d):
                failures.append({"case": case, "radius": obtained, "reference": root, "relative": relative})
    return {"cases": count, "max_relative_error": max_error, "failures": failures}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = sweep()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
