"""Run the opt-in Roche experiment; no production solver policy is changed.

Run from repository root with PYTHONPATH=src:. and --output DIRECTORY.
The Decimal oracle is independent of ELISA. Full results are JSONL; stationary
and critical solver comparisons are JSON. Not part of regular pytest collection.
"""

# ruff: noqa: INP001
import argparse
import json
import platform
import time
from collections import Counter
from collections.abc import Callable
from decimal import Decimal, localcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy

from elisa import settings
from elisa.base.error import ElisaError
from elisa.binary_system import model, radius
from elisa.binary_system import system as system_module
from elisa.binary_system.system import BinarySystem
from scripts.diagnostics.roche_reference import Ray, decimal, reference

# Unequal/symmetric/inverted/extreme masses, plus selected eccentric phases and spins.
SCENARIOS = [(f"circular-q{q}", q, 1.0, 1.0, 0.0) for q in (0.001, 0.01, 0.1, 0.7, 1.0, 3.0, 100.0, 1000.0)] + [
    ("audit-L1", 0.6999999999999998, 1.0, 1.0, 0.0),
    ("ecc-periastron-slow", 0.1, 0.6, 0.7, 0.3),
    ("ecc-mid-slow", 0.1, 0.6, 1.0, 0.3),
    ("ecc-apastron-slow", 0.1, 0.6, 1.3, 0.3),
    ("ecc-periastron-fast", 3.0, 1.8, 0.6, 0.4),
    ("ecc-apastron-fast", 3.0, 1.8, 1.4, 0.4),
    ("extreme-fast", 0.01, 3.0, 0.4, 0.6),
    ("inverted-slow", 10.0, 0.3, 1.6, 0.6),
]
DIRECTIONS = {
    "forward": (0.0, np.pi / 2),
    "backward": (np.pi, np.pi / 2),
    "side": (np.pi / 2, np.pi / 2),
    "polar": (0.0, 0.0),
}


def targets(critical: Decimal, *, dense: bool = False) -> dict[str, float]:
    center = float(critical)
    values = {"rounded-critical": center}
    for exp in (2, 4, 6, 8, 10, 12, 13, 14, 15) if dense else (2, 4, 6, 8, 10, 12):
        delta = max(1.0, abs(center)) * 10.0 ** (-exp)
        for sign in (-1, 1):
            values[f"rel{exp}:{sign:+d}"] = center + sign * delta
    for steps in (1, 2, 4, 8, 16, 32, 64, 128) if dense else (1, 4, 16):
        for sign in (-1, 1):
            value = center
            for _ in range(steps):
                value = np.nextafter(value, np.inf if sign > 0 else -np.inf)
            values[f"ulp{steps}:{sign:+d}"] = float(value)
    return values


def run(output: Path, precision: int = 65) -> None:  # noqa: C901, PLR0912, PLR0915 - explicit experiment matrix
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows, critical_rows = [], []
    for name, q, spin, distance, ecc in SCENARIOS:
        for component in ("primary", "secondary"):
            refs = {
                direction: reference(q, spin, distance, component, direction, precision=precision)
                for direction in DIRECTIONS
            }
            for boundary in ("forward", "backward"):
                ref = refs[boundary]
                axis = "L1" if boundary == "forward" else "L3" if component == "primary" else "L2"
                item = {
                    "scenario": name,
                    "q": q,
                    "spin": spin,
                    "distance": distance,
                    "eccentricity": ecc,
                    "component": component,
                    "boundary": boundary,
                    "axis": axis if spin == 1 and distance == 1 else f"{axis}-like component-local",
                    "stationary": str(ref["stationary"]),
                    "critical": str(ref["critical"]),
                }
                if boundary == "forward":
                    calls = []
                    original = scipy.optimize.newton

                    def record_newton(
                        *args,
                        _original: Callable = original,
                        _calls: list[dict] = calls,
                        **kwargs,
                    ) -> tuple:
                        result = _original(*args, **kwargs)
                        _calls.append(
                            {
                                "root": float(result[0]),
                                "converged": bool(result[1].converged),
                                "iterations": result[1].iterations,
                                "function_calls": result[1].function_calls,
                                "flag": result[1].flag,
                                "derivative_residual": float(args[0](result[0], *kwargs["args"])),
                            },
                        )
                        return result

                    try:
                        with patch("elisa.binary_system.system.optimize.newton", side_effect=record_newton):
                            item["production_critical"] = float(
                                BinarySystem.critical_potential_static(component, distance, q, spin),
                            )
                        item["critical_error"] = float(decimal(item["production_critical"]) - ref["critical"])
                        radial_stationary = (
                            calls[-1]["root"] if component == "primary" else distance - calls[-1]["root"]
                        )
                        item["stationary_error"] = float(decimal(radial_stationary) - ref["stationary"])
                    except (ElisaError, RuntimeError, ValueError) as exc:
                        item["error"] = repr(exc)
                    item["newton"] = calls
                critical_rows.append(item)
                for delta_label, omega in targets(
                    ref["critical"],
                    dense=name in {"circular-q1000.0", "circular-q0.001", "audit-L1", "extreme-fast", "inverted-slow"},
                ).items():
                    if spin == 1 and distance == 1:
                        inner = float(refs["forward"]["critical"])
                        local_outer = float(refs["backward"]["critical"])
                        morphology = (
                            "detached"
                            if omega > inner
                            else "critical-limit"
                            if omega == inner
                            else "overcontact-local"
                            if omega > local_outer
                            else "open-local"
                        )
                    else:
                        morphology = "component-local-generalized; global morphology not inferred"
                    for direction, (phi, theta) in DIRECTIONS.items():
                        ray_ref = refs[direction]
                        with localcontext() as ctx:
                            ctx.prec = precision
                            ray = Ray(q, spin, distance, component, direction)
                            root = ray.inner_root(omega, ray_ref["stationary"])
                            separation = None if ray_ref["critical"] is None else decimal(omega) - ray_ref["critical"]
                        record = dict(
                            item,
                            direction=direction,
                            phi=phi,
                            theta=theta,
                            morphology=morphology,
                            delta_label=delta_label,
                            omega=omega,
                            delta_from_boundary=str(decimal(omega) - ref["critical"]),
                            delta_from_ray_minimum=None if separation is None else str(separation),
                            reference_radius=None if root is None else str(root),
                            no_relevant_root=root is None,
                        )
                        precalc = getattr(model, "pre_calculate_for_potential_value_" + component)
                        function = getattr(model, "potential_" + component + "_fn")
                        args = ((q, *precalc(spin, q, distance, phi, theta)), omega)
                        captured = []
                        original_fsolve = radius.fsolve

                        def record_fsolve(
                            *a,
                            _original: Callable = original_fsolve,
                            _captured: list[tuple] = captured,
                            **kw,
                        ) -> tuple:
                            result = _original(*a, **kw)
                            _captured.append(result)
                            return result

                        try:
                            with patch.object(radius, "fsolve", side_effect=record_fsolve):
                                value = radius.calculate_radius(spin, q, omega, component, distance, phi, theta)
                            record.update(accepted=True, returned=float(value), exception=None)
                        except ElisaError as exc:
                            record.update(accepted=False, returned=None, exception=type(exc).__name__)
                        solution, info, ier, message = captured[0]
                        value = float(solution[0])
                        with localcontext() as ctx:
                            ctx.prec = precision
                            try:
                                residual = float(ray.potential(decimal(value)) - decimal(omega))
                            except ArithmeticError:
                                residual = None
                            absolute = None if root is None else float(abs(decimal(value) - root))
                            relative = None if root is None else float(abs(decimal(value) - root) / root)
                            limit_error = (
                                None
                                if ray_ref["stationary"] is None
                                else float(abs(decimal(value) - ray_ref["stationary"]))
                            )
                        correct_branch = (
                            value > 0
                            and (direction != "forward" or value < distance)
                            and (
                                ray_ref["stationary"] is None
                                or decimal(value) <= ray_ref["stationary"] + decimal(1e-12 * max(1.0, distance))
                            )
                        )
                        record.update(
                            solver_coefficients=[float(coefficient) for coefficient in args[0]],
                            initial_radius=1e-4,
                            xtol=1e-10,
                            stationary_curvature=(
                                None if ray_ref["stationary"] is None else float(ray.curvature(ray_ref["stationary"]))
                            ),
                            radius=value,
                            ier=int(ier),
                            mesg=message,
                            nfev=int(info["nfev"]),
                            residual_float=float(function(solution, *args)[0]),
                            residual_reference=residual,
                            absolute_error=absolute,
                            relative_error=relative,
                            correct_branch=correct_branch,
                            distance_from_stationary=limit_error,
                            close_to_ray_minimum=(
                                separation is not None and abs(separation) <= decimal(16 * abs(np.spacing(omega)))
                            ),
                        )
                        rows.append(record)
    lagrange = []
    for _name, q, spin, distance, _ecc in SCENARIOS:
        if spin != 1 or distance != 1:
            continue
        pback = reference(q, spin, distance, "primary", "backward")
        pfront = reference(q, spin, distance, "primary", "forward")
        sback = reference(q, spin, distance, "secondary", "backward")
        expected = [-float(pback["stationary"]), float(pfront["stationary"]), distance + float(sback["stationary"])]
        try:
            attempts = []
            lagrange_solve = system_module.fsolve

            def trace_lagrange(
                *args,
                _solver: Callable = lagrange_solve,
                _attempts: list[dict] = attempts,
                **kwargs,
            ) -> tuple:
                result = _solver(*args, **kwargs)
                _attempts.append(
                    {
                        "x0": float(args[1]),
                        "radius": float(result[0][0]),
                        "ier": int(result[2]),
                        "mesg": result[3],
                        "nfev": int(result[1]["nfev"]),
                        "residual": float(args[0](result[0][0], *kwargs["args"])),
                    },
                )
                return result

            BinarySystem._lagrangian_points_static.cache_clear()  # noqa: SLF001 - trace real uncached solver attempts
            with patch.object(system_module, "fsolve", side_effect=trace_lagrange):
                computed = BinarySystem.lagrangian_points_static(distance, q)
            potentials = BinarySystem.libration_potentials_static(distance, q)
            errors = [abs(a - b) for a, b in zip(sorted(computed), sorted(expected), strict=True)]
            lagrange.append(
                {
                    "q": q,
                    "computed": computed,
                    "reference_by_coordinate": expected,
                    "errors_sorted": errors,
                    "potentials": potentials,
                    "solver_attempts": attempts,
                },
            )
        except ElisaError as exc:
            lagrange.append({"q": q, "error": repr(exc)})
    with (output / "radii.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    (output / "critical.json").write_text(json.dumps(critical_rows, indent=2))
    (output / "lagrange.json").write_text(json.dumps(lagrange, indent=2))
    summary = {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "python": platform.python_version(),
        "reference_precision": precision,
        "scenarios": len(SCENARIOS),
        "calls": len(rows),
        "seconds": time.perf_counter() - started,
        "statuses": dict(Counter(row["ier"] for row in rows)),
        "accepted": sum(row["accepted"] for row in rows),
        "settings": settings.settings_serializer(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "settings"}, indent=2))  # noqa: T201 - CLI summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", type=int, default=65)
    options = parser.parse_args()
    run(options.output, options.precision)
