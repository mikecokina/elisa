"""Reproduce solver timings and compare the frozen independent audit matrix.

Run with PYTHONPATH=src:. in the target environment. Baseline is loaded from
Git into a separate module, without modifying the working tree or index.
"""

# ruff: noqa: INP001, T201, S102 - opt-in diagnostic executes the local committed baseline
import argparse
import gzip
import json
import logging
import statistics
import subprocess
import time
import types
from pathlib import Path
from unittest.mock import patch

from elisa.base.error import ElisaError
from elisa.binary_system import _radius_solver as solver
from elisa.binary_system import radius
from elisa.binary_system.container import OrbitalPositionContainer
from elisa.const import Position
from scripts.diagnostics.roche_reference import reference
from tests.utils import BINARY_SYSTEM_PARAMS, prepare_binary_system

BASELINE = "83082fd4"


def measure(function, repeats: int) -> float:  # noqa: ANN001
    function()
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(repeats):
            function()
        samples.append((time.perf_counter() - start) / repeats)
    return statistics.median(samples)


def critical_four(critical_omega: float) -> list[str]:
    outcomes = []
    for direction in ("forward", "backward", "side", "polar"):
        try:
            getattr(radius, f"calculate_{direction}_radius")(1.0, 0.7, 1.0, critical_omega, "primary")
        except (ElisaError, FloatingPointError) as exc:
            outcomes.append(type(exc).__name__)
        else:
            outcomes.append("returned")
    return outcomes


def benchmark() -> dict:
    baseline = types.ModuleType("baseline_radius")
    source = subprocess.run(  # noqa: S603 - fixed local git invocation
        ["git", "show", f"{BASELINE}:src/elisa/binary_system/radius.py"],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    exec(compile(source, "baseline_radius.py", "exec"), baseline.__dict__)
    system = prepare_binary_system(BINARY_SYSTEM_PARAMS["detached"])

    def mesh() -> None:
        container = OrbitalPositionContainer.from_binary_system(system, Position(0, 1.0, 0.0, 0.0, 0.0))
        container.build_mesh(components_distance=1.0)

    def four(omega: float) -> None:
        for direction in ("forward", "backward", "side", "polar"):
            getattr(radius, f"calculate_{direction}_radius")(1.0, 0.7, 1.0, omega, "primary")

    critical_omega = float(reference(0.7, 1.0, 1.0, "primary", "forward")["critical"])

    results = {}
    for name, implementation in (("baseline", baseline.calculate_radius), ("new", radius.calculate_radius)):
        with patch.object(radius, "calculate_radius", implementation):
            results[name] = {
                "ordinary_four": measure(lambda: four(4.0), 300),
                "near_critical_four": measure(lambda: four(3.243288166372894 + 1e-10), 300),
                "rounded_critical_four": measure(lambda: critical_four(critical_omega), 300),
                "rounded_critical_outcomes": critical_four(critical_omega),
                "initialization": measure(lambda: prepare_binary_system(BINARY_SYSTEM_PARAMS["detached"]), 30),
                "mesh": measure(mesh, 30),
            }
    # Count callbacks separately: instrumentation must not contaminate timings.
    for name, implementation in (("baseline", baseline.calculate_radius), ("new", radius.calculate_radius)):
        for case, omega in (("ordinary", 4.0), ("near_critical", 3.243288166372894 + 1e-10)):
            calls = []
            original = baseline.fsolve if name == "baseline" else solver.brentq

            def record(*args, **kwargs) -> tuple:
                result = original(*args, **kwargs)  # noqa: B023 - called synchronously inside this loop
                calls.append(result[1]["nfev"] if name == "baseline" else result[1].function_calls)  # noqa: B023
                return result

            module, attribute = (baseline, "fsolve") if name == "baseline" else (solver, "brentq")
            with patch.object(radius, "calculate_radius", implementation), patch.object(module, attribute, record):
                four(omega)
            results[name][f"{case}_solver_callback_counts"] = calls
    rays = [
        solver.RocheRay(1.0, 0.7, 1.0, c, rotation, 0.0, 1 - c * c)
        for c, rotation in ((1.0, 0.85), (-1.0, 0.85), (0.0, 0.85), (0.0, 0.0))
    ]
    results["four_geometric_bounds_seconds"] = measure(lambda: [ray.first_turn() for ray in rays], 1000)
    return results


def validate() -> dict:
    errors, rejected_existing, accepted_absent = [], [], []
    accepted_limits = 0
    with gzip.open("docs/audits/data/roche_radius/new-radii.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream]
    for index, row in enumerate(rows):
        try:
            value = radius.calculate_radius(
                row["spin"],
                row["q"],
                row["omega"],
                row["component"],
                row["distance"],
                row["phi"],
                row["theta"],
            )
        except (FloatingPointError, ArithmeticError):
            if row["reference_radius"] is not None:
                rejected_existing.append(index)
            continue
        if row["reference_radius"] is not None:
            expected = float(row["reference_radius"])
            errors.append(abs(value - expected) / expected)
        else:
            critical = reference(row["q"], row["spin"], row["distance"], row["component"], row["direction"])
            if abs(value - float(critical["stationary"])) <= 1e-12 * row["distance"]:
                accepted_limits += 1
            else:
                accepted_absent.append(index)
    return {
        "rows": len(rows),
        "existing_roots": len(errors),
        "max_relative_error": max(errors),
        "rejected_existing": rejected_existing,
        "accepted_absent_nonlimit": accepted_absent,
        "accepted_critical_limits": accepted_limits,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    output = {"baseline": BASELINE, "benchmark": benchmark(), "validation": validate()}
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
