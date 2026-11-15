"""Benchmark well-conditioned rays with known stationary brackets.

Run with PYTHONPATH=src:. and a radii.jsonl path. This excludes bracket discovery,
critical points and the cost of validating a candidate: it is not an end-to-end
replacement-solver benchmark.
"""

# ruff: noqa: INP001
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from elisa.binary_system import model
from elisa.opt.fsolver import fsolve


def benchmark(path: Path) -> dict:
    """Time both methods on the same 64 pre-bracketed regular intersections."""
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    selected = [
        row
        for row in rows
        if row["direction"] == row["boundary"]
        and row["reference_radius"] is not None
        and row["delta_label"] == "rel6:+1"
    ]
    arguments = []
    for row in selected:
        function = getattr(model, "potential_" + row["component"] + "_fn")
        precalculate = getattr(model, "pre_calculate_for_potential_value_" + row["component"])
        coefficients = precalculate(row["spin"], row["q"], row["distance"], row["phi"], row["theta"])
        arguments.append(
            (function, ((row["q"], *coefficients), row["omega"]), float(row["stationary"]), row["distance"]),
        )
    result = {"cases": len(arguments), "repeats": 20, "bracket_cost_included": False}
    for name in ("fsolve", "brentq"):
        started, calls = time.perf_counter(), 0
        for _ in range(20):
            for function, args, minimum, distance in arguments:
                if name == "fsolve":
                    _, info, _, _ = fsolve(function, np.array([1e-4]), args=args, full_output=True, xtol=1e-10)
                    calls += info["nfev"]
                else:
                    _, info = brentq(
                        function,
                        distance * 1e-12,
                        minimum,
                        args=args,
                        xtol=5e-15 * distance,
                        rtol=1e-14,
                        full_output=True,
                    )
                    calls += info.function_calls
        result[name] = {
            "microseconds_per_call": (time.perf_counter() - started) * 1e6 / (20 * len(arguments)),
            "mean_nfev": calls / (20 * len(arguments)),
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    print(json.dumps(benchmark(parser.parse_args().input), indent=2))  # noqa: T201 - CLI output
