"""Audit x0 sensitivity on a selected, frozen subset of the previous radius audit.

Use PYTHONPATH=src:. and --output DIRECTORY. Only x0 changes: equations, physical
inputs and fsolve options stay fixed. Oracle-based starts are diagnostic controls,
not proposed production initializers. No production code is patched or modified.
"""

# ruff: noqa: INP001
import argparse
import gzip
import hashlib
import json
import platform
import time
import warnings
from collections import defaultdict
from decimal import Decimal, localcontext
from itertools import pairwise
from pathlib import Path

import numpy as np
import scipy

from elisa.binary_system import model
from elisa.opt.fsolver import fsolve
from scripts.diagnostics.roche_reference import Ray, reference

CORE = {
    "circular-q0.001",
    "circular-q0.1",
    "circular-q0.7",
    "circular-q1.0",
    "circular-q1000.0",
    "audit-L1",
    "ecc-periastron-slow",
    "ecc-apastron-slow",
    "ecc-apastron-fast",
    "extreme-fast",
    "inverted-slow",
}
NONPROGRESS_STATUS = 5
MAX_RADIUS = 30.0
KEYS = ("q", "spin", "distance", "component", "direction", "omega")


def select_cases(source: Path) -> list[dict]:
    """Retain representative geometry plus every prior positive/negative anomaly."""
    selected = {}
    with gzip.open(source, "rt") as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            reasons = []
            if row["scenario"] in CORE:
                if row["boundary"] == "forward" and row["delta_label"] in {"rel2:+1", "rounded-critical", "rel6:-1"}:
                    reasons.append("representative-four-directions")
                if row["direction"] == row["boundary"] and row["delta_label"] in {
                    "rel8:+1",
                    "rel12:+1",
                    "ulp1:+1",
                    "ulp1:-1",
                    "rounded-critical",
                    "rel6:-1",
                    "rel2:+1",
                }:
                    reasons.append("critical-rays")
            if row["ier"] == NONPROGRESS_STATUS and row["reference_radius"] is not None:
                reasons.append("all-17-accurate-status5")
            if row["accepted"] and row["correct_branch"] is False and row["reference_radius"] is not None:
                reasons.append("prior-wrong-branch")
            if row["accepted"] and row["no_relevant_root"] and not row["close_to_ray_minimum"]:
                reasons.append("prior-accepted-no-root")
            if not reasons:
                continue
            key = tuple(row[k] for k in KEYS)
            if key not in selected:
                selected[key] = dict(row, source_indices=[], selection_reasons=[])
            selected[key]["source_indices"].append(index)
            selected[key]["selection_reasons"] = sorted(set(selected[key]["selection_reasons"] + reasons))
    return [dict(row, case_id=index) for index, row in enumerate(selected.values())]


def initial_guesses(case: dict) -> dict[str, tuple[float, str]]:
    """Separate algebra, ideal prior results, and unavailable reference information."""
    q, spin, distance, component, direction, omega = (case[k] for k in KEYS)
    rc = float(case["critical_start_anchor"])
    guesses = {f"fixed-{value:g}": (value, "constant") for value in (1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 0.5)}
    a, b, offset = (1.0, q, 0.0) if component == "primary" else (q, 1.0, (1 - q) / 2)
    denominator = omega - offset - b / distance
    if denominator > 0:
        estimate = a / denominator
        guesses["algebra-half"] = (estimate / 2, "algebraic")
        guesses["algebra-full"] = (estimate, "algebraic")
    for fraction in (0.25, 0.5, 0.75, 0.95):
        guesses[f"critical-{fraction:g}"] = (fraction * rc, "reference-stationary")
    for factor in (1.05, 1.5):
        guesses[f"outer-{factor:g}"] = (factor * rc, "reference-stationary-deliberately-wrong")
    if case["reference_radius"] is not None:
        root = float(case["reference_radius"])
        for factor in (0.99, 0.999999, 1.0, 1.000001, 1.01):
            guesses[f"root-{factor}"] = (factor * root, "exact-reference-control")
    # Idealized previously valid results, not a request to run another solver in production.
    previous = reference(q, spin, distance, component, direction, omega + 1e-3 * max(1.0, abs(omega)))["root"]
    if previous is not None:
        guesses["previous-omega"] = (float(previous), "ideal-previous-valid-radius")
    previous_distance = reference(q, spin, distance * 1.02, component, direction, omega)["root"]
    if previous_distance is not None:
        guesses["previous-d1.02"] = (float(previous_distance), "ideal-previous-valid-radius")
    return guesses


def experiment(case: dict, strategy: str, x0: float, source_kind: str) -> dict:
    """Trace each function evaluation and independently evaluate the final candidate."""
    q, spin, distance, component, direction, omega = (case[k] for k in KEYS)
    precalculate = getattr(model, "pre_calculate_for_potential_value_" + component)
    function = getattr(model, "potential_" + component + "_fn")
    args = ((q, *precalculate(spin, q, distance, case["phi"], case["theta"])), omega)
    trials = []

    def traced(radii: np.ndarray, *parameters: tuple) -> np.ndarray:
        trials.append(float(radii[0]))
        return function(radii, *parameters)

    started = time.perf_counter()
    solution, info, ier, message = fsolve(
        traced,
        np.array([x0], dtype=np.float64),
        args=args,
        full_output=True,
        xtol=1e-10,
    )
    elapsed = time.perf_counter() - started
    r = float(solution[0])
    anchor = {
        "stationary": None if case["ray_stationary"] is None else Decimal(case["ray_stationary"]),
        "critical": None if case["ray_critical"] is None else Decimal(case["ray_critical"]),
    }
    with localcontext() as context:
        context.prec = 65
        ray = Ray(q, spin, distance, component, direction)
        try:
            residual = float(ray.potential(Decimal(r)) - Decimal(omega))
        except ArithmeticError:
            residual = None
        expected = None if case["reference_radius"] is None else Decimal(case["reference_radius"])
        absolute = None if expected is None else float(abs(Decimal(r) - expected))
        relative = None if expected is None else float(abs(Decimal(r) - expected) / expected)
        limit_error = None if anchor["stationary"] is None else float(abs(Decimal(r) - anchor["stationary"]))
        limit_delta = None if anchor["critical"] is None else float(Decimal(omega) - anchor["critical"])
    spatial_domain = bool(np.isfinite(r) and r > 0 and (direction != "forward" or r < distance))
    inner_branch = spatial_domain and (
        anchor["stationary"] is None or r <= float(anchor["stationary"]) + 1e-12 * max(1.0, distance)
    )
    accepted = bool(np.isfinite(r) and ier == 1 and 0 <= r <= MAX_RADIUS)
    good = bool(expected is not None and inner_branch and absolute <= max(1e-10 * distance, 1e-7 * float(expected)))
    near = limit_delta is not None and abs(limit_delta) <= 16 * abs(np.spacing(omega))
    exited = any(not np.isfinite(v) or v <= 0 or (direction == "forward" and v >= distance) for v in trials)
    beyond_minimum = anchor["stationary"] is not None and any(
        v > float(anchor["stationary"]) + 1e-12 * max(1.0, distance) for v in trials
    )
    singularities = [0.0, distance] if direction == "forward" else [0.0]
    crossed = [
        singularity
        for singularity in singularities
        if any((a - singularity) * (b - singularity) < 0 or singularity in (a, b) for a, b in pairwise(trials))
    ]
    return {
        "case_id": case["case_id"],
        "strategy": strategy,
        "source_kind": source_kind,
        "x0": x0,
        "radius": r,
        "ier": int(ier),
        "mesg": message,
        "nfev": int(info["nfev"]),
        "seconds_traced": elapsed,
        "residual_float": float(function(solution, *args)[0]),
        "residual_reference": residual,
        "absolute_error": absolute,
        "relative_error": relative,
        "reference_exists": expected is not None,
        "inner_branch": inner_branch,
        "spatial_domain": spatial_domain,
        "current_policy_accepts": accepted,
        "quality_good": good,
        "accepted_correct": accepted and good,
        "accepted_wrong": accepted and ((expected is not None and not good) or (expected is None and not near)),
        "critical_roundoff_band": bool(near),
        "distance_from_stationary": limit_error,
        "delta_from_ray_minimum": limit_delta,
        "trajectory_left_domain": exited,
        "initial_outside_domain": bool(x0 <= 0 or (direction == "forward" and x0 >= distance)),
        "left_domain_after_valid_start": bool(exited and x0 > 0 and (direction != "forward" or x0 < distance)),
        "trajectory_beyond_minimum": bool(beyond_minimum),
        "crossed_singularities": crossed,
        "trial_radii": trials,
    }


def summarize(cases: list[dict], runs: list[dict]) -> dict:
    """Count status, reference quality and domain violations separately."""
    by_strategy = defaultdict(list)
    for row in runs:
        by_strategy[row["strategy"]].append(row)
    strategies = {}
    group17 = {row["case_id"] for row in cases if "all-17-accurate-status5" in row["selection_reasons"]}
    repaired = set()
    for name, rows in by_strategy.items():
        valid = [r for r in rows if r["reference_exists"]]
        repaired_here = {r["case_id"] for r in rows if r["case_id"] in group17 and r["accepted_correct"]}
        repaired |= repaired_here
        strategies[name] = {
            "calls": len(rows),
            "existing_roots": len(valid),
            "accepted_correct": sum(r["accepted_correct"] for r in rows),
            "raw_good": sum(r["quality_good"] for r in rows),
            "accepted_wrong": sum(r["accepted_wrong"] for r in rows),
            "accepted_no_exact_root": sum(r["current_policy_accepts"] and not r["reference_exists"] for r in rows),
            "status_failure_existing": sum(r["ier"] != 1 for r in valid),
            "quality_good_rejected": sum(r["quality_good"] and not r["current_policy_accepts"] for r in rows),
            "median_nfev": float(np.median([r["nfev"] for r in rows])),
            "median_nfev_existing": float(np.median([r["nfev"] for r in valid])),
            "max_relative_error_existing": max((r["relative_error"] for r in valid), default=None),
            "domain_exit": sum(r["trajectory_left_domain"] for r in rows),
            "initial_outside_domain": sum(r["initial_outside_domain"] for r in rows),
            "crossed_singularity": sum(bool(r["crossed_singularities"]) for r in rows),
            "repaired_17": len(repaired_here),
        }
    return {
        "cases": len(cases),
        "runs": len(runs),
        "existing_roots": sum(c["reference_radius"] is not None for c in cases),
        "prior_17_cases": len(group17),
        "repaired_17_any_start": len(repaired),
        "strategies": strategies,
    }


def run(source: Path, output: Path) -> None:
    """Write a reproducible x0-only experiment without repeating the old matrix."""
    if np.__version__ != "2.4.2" or scipy.__version__ != "1.17.1":
        message = "This audit requires NumPy 2.4.2 and SciPy 1.17.1"
        raise RuntimeError(message)
    output.mkdir(parents=True, exist_ok=True)
    cases = select_cases(source)
    for case in cases:
        q, spin, distance, component, direction, _omega = (case[k] for k in KEYS)
        anchor = reference(q, spin, distance, component, direction)
        case["ray_stationary"] = None if anchor["stationary"] is None else str(anchor["stationary"])
        case["ray_critical"] = None if anchor["critical"] is None else str(anchor["critical"])
        scale = anchor["stationary"]
        if scale is None:
            scale = reference(q, spin, distance, component, "forward")["stationary"]
        case["critical_start_anchor"] = str(scale)
    runs = []
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for case in cases:
            for name, (x0, kind) in initial_guesses(case).items():
                runs.append(experiment(case, name, x0, kind))
    # Focused follow-up only where the original constant/algebra strategies did
    # not repair one of the 17 candidates. Do not present this adaptive scan as
    # a uniformly tested new production strategy.
    group17 = {case["case_id"] for case in cases if "all-17-accurate-status5" in case["selection_reasons"]}
    cheap_repaired = {
        row["case_id"] for row in runs if row["source_kind"] in {"constant", "algebraic"} and row["accepted_correct"]
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for case_id in sorted(group17 - cheap_repaired):
            runs.extend(
                experiment(cases[case_id], f"adaptive-fixed-{x0}", x0, "constant-adaptive")
                for x0 in (0.002, 0.005, 0.02, 0.025, 0.05, 0.075)
            )
    summary = summarize(cases, runs)
    summary.update(
        numpy=np.__version__,
        scipy=scipy.__version__,
        python=platform.python_version(),
        seconds=time.perf_counter() - started,
        source=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        fsolve_options={
            "xtol": 1e-10,
            "full_output": True,
            "maxfev": 0,
            "factor": 100,
            "fprime": None,
            "epsfcn": None,
            "diag": None,
            "band": None,
        },
        baseline_max_radius_difference=max(
            abs(r["radius"] - cases[r["case_id"]]["radius"]) for r in runs if r["strategy"] == "fixed-0.0001"
        ),
    )
    (output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    with gzip.open(output / "runs.jsonl.gz", "wt") as stream:
        for row in runs:
            stream.write(json.dumps(row) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "strategies"}, indent=2))  # noqa: T201


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("docs/audits/data/roche_radius/new-radii.jsonl.gz"))
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    run(options.source, options.output)
