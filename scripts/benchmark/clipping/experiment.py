"""Reproducible serial clipping experiment. Run from repository root with PYTHONPATH=src.

Uses installed dependencies and configured LD tables; never installs anything.
Writes measurement artifacts only below this directory.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import platform
import pstats
import sys
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
import logging

import numpy as np
import numba
import scipy
import astropy
from astropy import units as u

from elisa import BinarySystem, Observer, Star, settings
from elisa.binary_system.surface import coverage
from elisa.binary_system.surface import clipping as clip
from elisa.binary_system import utils as bsutils
from elisa.binary_system.curves import c_managed
from scripts.benchmark.clipping.analyze import exact_area

BACKENDS = {"pypex": "pypex", "reference": "reference", "numba": "sutherland_hodgman"}

OUT = Path(__file__).resolve().parent


def system(variant):
    primary = Star(
        mass=2.15 * u.solMass,
        surface_potential=3.6,
        synchronicity=1.0,
        t_eff=10000 * u.K,
        gravity_darkening=1.0,
        discretization_factor=8 if variant == "coarse" else 5,
        albedo=0.6,
        metallicity=0.0,
        atmosphere="bb",
    )
    extra = {"discretization_factor": 8} if variant == "coarse" else {}
    secondary = Star(
        mass=1.2 * u.solMass,
        surface_potential=4.0,
        synchronicity=1.0,
        t_eff=7000 * u.K,
        gravity_darkening=1.0,
        albedo=0.6,
        metallicity=0.0,
        atmosphere="bb",
        **extra,
    )
    bs = BinarySystem(
        primary=primary,
        secondary=secondary,
        argument_of_periastron=58 * u.deg,
        gamma=-30.7 * u.km / u.s,
        period=2.5 * u.d,
        eccentricity=0.0,
        inclination=(80 if variant == "inclination80" else 85) * u.deg,
        distance=155 * u.pc,
        primary_minimum_time=2440000.0 * u.d,
        phase_shift=0.0,
    )
    assert bs.morphology == "detached", bs.morphology
    assert bs.eccentricity == 0.0 and bs.primary.synchronicity == bs.secondary.synchronicity == 1.0
    return bs


def pypex(p, t):
    hull = bsutils.hull_to_pypex_poly(p)
    faces = bsutils.faces_to_pypex_poly(t)
    return np.array(bsutils.pypex_poly_surface_area(bsutils.pypex_poly_hull_intersection(faces, hull)))


def stats(times, count=None):
    median = float(np.median(times))
    result = dict(
        median_s=median,
        min_s=min(times),
        max_s=max(times),
        std_s=float(np.std(times)),
        samples_s=times,
    )
    if count is not None:
        result["intersections_per_s"] = count / median
    return result


def timed(fn, repeats, count=1):
    times = []
    for _ in range(repeats):
        start = perf_counter()
        fn()
        times.append(perf_counter() - start)
    return stats(times, count)


def area(p):
    q = p - p[0]
    return 0.5 * abs(np.sum(q[:, 0] * np.roll(q[:, 1], -1) - q[:, 1] * np.roll(q[:, 0], -1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    settings.configure(NUMBER_OF_PROCESSES=1, INTERSECTION_BACKEND="pypex")
    logging.getLogger("observer.observer").setLevel(logging.WARNING)
    result = {
        "environment": dict(
            python=sys.version,
            executable=sys.executable,
            numpy=np.__version__,
            numba=numba.__version__,
            scipy=scipy.__version__,
            astropy=astropy.__version__,
            platform=platform.platform(),
            cpu=Path("/proc/cpuinfo").read_text().split("model name")[1].split("\n")[0],
            workers=settings.NUMBER_OF_PROCESSES,
            ld_tables=settings.LD_TABLES,
            secondary_default_discretization=settings.DEFAULT_DISCRETIZATION_FACTOR
            if hasattr(settings, "DEFAULT_DISCRETIZATION_FACTOR")
            else None,
        ),
        "systems": {},
        "geometry": {},
    }
    dummy_p = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    start = perf_counter()
    clip.intersection_areas(dummy_p, dummy_p[:3][None])
    result["numba_first_call_s"] = perf_counter() - start
    batches = []
    batch_meta = []
    original = coverage.partial_visible_faces_surface_coverage
    phase_context = {}

    def capture(points, faces, normals, hull):
        batches.append((points.copy(), faces.copy(), normals.copy(), hull.copy()))
        batch_meta.append(dict(phase_context))
        return original(points, faces, normals, hull)

    original_compute = coverage.compute_surface_coverage

    def compute(*a, **kw):
        position = (a[0] if a else kw["system"]).position
        phase_context["phase"] = float(position.phase)
        container = a[0] if a else kw["system"]
        phase_context["mesh_faces"] = [len(container.primary.faces), len(container.secondary.faces)]
        return original_compute(*a, **kw)

    kwargs = dict(from_phase=-0.6, to_phase=0.6, phase_step=0.01, normalize=True)
    for variant in ("baseline", "inclination80", "coarse"):
        bs = system(variant)
        observer = Observer(passband=["Generic.Bessell.U"], system=bs)
        rec = {
            "morphology": bs.morphology,
            "eccentricity": bs.eccentricity,
            "synchronicity": [bs.primary.synchronicity, bs.secondary.synchronicity],
            "inclination_deg": float(np.degrees(bs.inclination)),
            "discretization_deg": [
                float(np.degrees(bs.primary.discretization_factor)),
                float(np.degrees(bs.secondary.discretization_factor)),
            ],
        }
        print("warming", variant, flush=True)
        observer.observe.lc(**kwargs)
        phase_context["system"] = variant
        before = len(batches)
        with (
            patch.object(coverage, "partial_visible_faces_surface_coverage", capture),
            patch.object(c_managed, "compute_surface_coverage", compute),
        ):
            phases, flux = observer.observe.lc(**kwargs)
        rec["geometry_batches"] = [before, len(batches)]
        rec["phase_count"] = len(phases)
        rec["mesh_faces"] = batch_meta[before]["mesh_faces"]
        rec["semi_major_axis_m"] = float(bs.semi_major_axis)
        values = {}
        # Identical observer, settings, cache warm-up and interleaved timing order.
        timings = {name: [] for name in ("pypex", "reference", "numba")}
        for name in timings:
            with patch.object(settings, "INTERSECTION_BACKEND", BACKENDS[name]):
                observer.observe.lc(**kwargs)
        for repeat in range(args.repeats):
            order = list(timings) if repeat % 2 == 0 else list(reversed(timings))
            for name in order:
                with patch.object(settings, "INTERSECTION_BACKEND", BACKENDS[name]):
                    start = perf_counter()
                    ph, fl = observer.observe.lc(**kwargs)
                    timings[name].append(perf_counter() - start)
                    values[name] = fl["Generic.Bessell.U"].copy()
        rec["timing"] = {name: stats(ts) for name, ts in timings.items()}
        rec["flux_comparison"] = {}
        for name in ("reference", "numba"):
            difference = values[name] - values["pypex"]
            contacts = (abs(phases - np.round(phases * 2) / 2) > 0.04) & (abs(phases - np.round(phases * 2) / 2) < 0.14)
            rec["flux_comparison"][name] = dict(
                max_abs_normalized=float(max(abs(difference))),
                ppm=float(max(abs(difference)) * 1e6),
                max_contact_abs=float(max(abs(difference[contacts]))),
                max_difference_phase=float(phases[np.argmax(abs(difference))]),
                depth_differences=[
                    float(
                        np.min(values["pypex"][abs(phases - center) < 0.1])
                        - np.min(values[name][abs(phases - center) < 0.1])
                    )
                    for center in (0.0, 0.5)
                ],
            )
        # Also compare unnormalized physical flux: same calculation, normalize=False.
        raw = {}
        for name in ("pypex", "numba"):
            with patch.object(settings, "INTERSECTION_BACKEND", BACKENDS[name]):
                _, fl = observer.observe.lc(**(kwargs | {"normalize": False, "flux_unit": u.W / u.m**2}))
                raw[name] = fl["Generic.Bessell.U"]
        rec["max_absolute_raw_flux_difference"] = float(max(abs(raw["pypex"] - raw["numba"])))
        for backend in ("pypex", "numba"):
            profiler = cProfile.Profile()
            with patch.object(settings, "INTERSECTION_BACKEND", BACKENDS[backend]):
                profiler.enable()
                observer.observe.lc(**kwargs)
                profiler.disable()
            buf = io.StringIO()
            pstats.Stats(profiler, stream=buf).strip_dirs().sort_stats("cumulative").print_stats(40)
            (OUT / f"profile_{variant}_{backend}.txt").write_text(buf.getvalue())
        np.savez_compressed(
            OUT / f"flux_{variant}.npz", phases=phases, **values, raw_pypex=raw["pypex"], raw_numba=raw["numba"]
        )
        result["systems"][variant] = rec
        print("finished", variant, rec["timing"], flush=True)
        (OUT / "results.json").write_text(json.dumps(result, indent=2))

    saved = {"batches": len(batches)}
    comparisons = []
    exact_cases = []

    for i, (points, faces, normals, hull) in enumerate(batches):
        triangles = points[faces]
        saved[f"h{i}"] = hull
        saved[f"t{i}"] = triangles
        saved[f"n{i}"] = normals
        saved[f"f{i}"] = faces
        saved[f"p{i}"] = points
        aa = {
            name: fn(hull, triangles)
            for name, fn in (
                ("pypex", pypex),
                ("reference", lambda p, t: clip.intersection_areas(p, t, compiled=False)),
                ("numba", clip.intersection_areas),
            )
        }
        if not len(triangles):
            continue
        a = aa["numba"]
        old = aa["pypex"]
        diff = abs(a - old)
        bounds = np.minimum(area(hull), np.array([area(t) for t in triangles]))
        budget = 64 * np.finfo(float).eps * len(hull) * max(np.ptp(hull, axis=0)) ** 2
        types = np.where(a == 0, "empty", np.where(abs(a - bounds) <= budget, "containment", "partial"))
        rel = np.divide(diff, a, out=np.zeros_like(a), where=a > budget)
        significant = diff > budget
        # Exact arithmetic investigates the largest legacy and ref discrepancies,
        # and a tiny positive intersection, using actual binary64 coordinates.
        chosen = {int(np.argmax(diff)), int(np.argmax(abs(aa["reference"] - a)))}
        positive = np.flatnonzero(a > 0)
        if len(positive):
            chosen.add(int(positive[np.argmin(a[positive])]))
        for j in chosen:
            oracle = exact_area(hull, triangles[j])
            exact_cases.append(
                dict(
                    batch=i,
                    face=j,
                    exact=oracle,
                    pypex=float(old[j]),
                    reference=float(aa["reference"][j]),
                    numba=float(a[j]),
                    error_budget=budget,
                )
            )
        comparisons.append(
            dict(
                batch=i,
                **batch_meta[i],
                faces=len(a),
                outline_vertices=len(hull),
                max_abs=float(max(diff)),
                max_relative_meaningful=float(max(rel)),
                max_reference_numba=float(max(abs(aa["reference"] - a))),
                bound_violations=int(np.sum((a < 0) | (a > bounds + budget))),
                legacy_bound_violations=int(np.sum((old < 0) | (old > bounds + budget))),
                significant_faces=int(sum(significant)),
                mismatched_face_indices=np.flatnonzero(significant).tolist(),
                empty=int(sum(types == "empty")),
                containment=int(sum(types == "containment")),
                partial=int(sum(types == "partial")),
                min_positive=float(min(a[positive])) if len(positive) else None,
                coverage_max_abs=float(max(abs((old - a) / np.abs(normals[:, 0])))),
                coverage_max_relative_meaningful=float(
                    max(np.divide(diff, bounds - a, out=np.zeros_like(a), where=(bounds - a) > budget))
                ),
            )
        )
        for name, v in aa.items():
            saved[f"a_{name}{i}"] = v
        saved[f"type{i}"] = types
    np.savez_compressed(OUT / "eclipse_geometry.npz", **saved)
    result["geometry"]["batches"] = comparisons
    result["geometry"]["exact_checks"] = exact_cases
    # Time all captured input batches: kernel includes conversion/preparation;
    # complete partial coverage includes normals and output processing.
    total = sum(len(b[1]) for b in batches)
    result["kernel_timing"] = {}
    result["coverage_timing"] = {}
    for name, fn in [
        ("pypex", pypex),
        ("reference", lambda p, t: clip.intersection_areas(p, t, compiled=False)),
        ("numba", clip.intersection_areas),
    ]:
        result["kernel_timing"][name] = timed(lambda: [fn(b[3], b[0][b[1]]) for b in batches], args.repeats, total)
        with patch.object(settings, "INTERSECTION_BACKEND", BACKENDS[name]):
            result["coverage_timing"][name] = timed(lambda: [original(*b) for b in batches], args.repeats, total)
    result["group_timing"] = {}
    for size in sorted(set(len(b[3]) for b in batches)):
        for kind in ("empty", "containment", "partial"):
            subset = [
                (b[3], b[0][b[1]][saved[f"type{i}"] == kind])
                for i, b in enumerate(batches)
                if len(b[3]) == size and f"type{i}" in saved
            ]
            count = sum(len(t) for p, t in subset)
            if not count:
                continue
            result["group_timing"][f"{size}/{kind}"] = {"count": count}
            for name, fn in [
                ("pypex", pypex),
                ("reference", lambda p, t: clip.intersection_areas(p, t, compiled=False)),
                ("numba", clip.intersection_areas),
            ]:
                result["group_timing"][f"{size}/{kind}"][name] = timed(
                    lambda: [fn(p, t) for p, t in subset if len(t)], args.repeats, count
                )
    (OUT / "results.json").write_text(json.dumps(result, indent=2))
    print("artifacts saved", total, "triangles", flush=True)


if __name__ == "__main__":
    main()
