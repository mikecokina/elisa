"""Render measured artifacts directly into the experiment report."""

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]


def table(headers, rows):
    return (
        "\n".join(
            ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
            + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
        )
        + "\n"
    )


def main():
    r = json.loads((OUT / "results.json").read_text())
    audit = json.loads((OUT / "exact_audit.json").read_text())
    compilation = json.loads((OUT / "compilation.json").read_text())
    data = np.load(OUT / "eclipse_geometry.npz")
    batches = r["geometry"]["batches"]
    env = r["environment"]
    text = """# Pypex clipping experiment

Measured on 2026-10-22. Following the user-requested adoption,
Sutherland-Hodgman is now the default; Pypex remains selectable. The experiment adds an independent
float64 Sutherland-Hodgman reference, a Numba batch implementation without fastmath,
and a settings-based selector. No geometry function API, face selection, coverage correction,
threshold, physical model, SAT routine, or existing Pypex implementation was changed.
No dependencies were installed and no changes were committed.

## Verified geometry contracts

Inspected `src/elisa/binary_system/surface/coverage.py`,
`src/elisa/binary_system/utils.py`, `src/elisa/pypex/poly2d/polygon.py`,
`src/elisa/pypex/poly2d/intersection/linter.py`, `src/elisa/pypex/base/shape.py`,
and the available audit at `docs/audits/PYPEX_ALGORITHM_AUDIT.md`.

* `get_eclipse_boundary_path` explicitly constructs `ConvexHull` of the projected
  foreground visible points and takes its ordered boundary vertices. Convexity is
  established at this call site; it is not inferred from `Polygon`'s name or its
  structural validity check. Arbitrary `Shape2D` inputs are not proven convex.
* `compute_surface_coverage` projects points with `plane_projection(..., "yz",
  keep_3d=False)`. Selected faces index `(3, 2)` projected triangles. The captured
  arrays confirm this, with all coordinates finite.
* The coverage docstring says clockwise, but the 2D SciPy hull yields an ordered
  boundary and the implementation does not require that documented winding.
  Pypex's sorting name is also insufficient evidence of orientation: the actual
  angle sort is increasing. The new algorithm establishes triangle winding from
  its signed determinant and accepts either polygon winding without sorting.
* Only `pypex_poly_surface_area` consumes the reconstructed intersections here.
  The downstream expression remains `(projected_triangle_area - intersection_area)
  / cosine`, followed by the existing surface noise threshold and SMA² conversion.
* There is no general nondegeneracy guarantee for a projected face. The new
  kernel explicitly handles zero float64 determinant triangles. No assumption
  about Pypex boundary predicates is needed by the new kernel.

## Implementation and scope

`src/elisa/binary_system/surface/clipping.py` contains:

* `convex_polygon_triangle_intersection_area`: readable list-based reference.
* `optimized_intersection_area`: validated scalar Numba entry point.
* `sutherland_hodgman_intersection_areas`: validated batch adapter, compiling the triangle loop.

The outline is the subject polygon, clipped against exactly three triangle
half-planes. For an edge `a -> b`, compute `f(x) = sign * cross(b-a, x-a)`, where
`sign` is the triangle winding. The boundary is included with `f >= 0`. An edge
crossing adds `s + f(s)/(f(s)-f(e)) * (e-s)`. Coincident boundaries do not divide
by zero because interpolation only occurs when the inside classifications differ.
Both implementations translate to the triangle origin before clipping and use
shoelace coordinates relative to the first surviving vertex, reducing cancellation.
They take the absolute signed area, with no rounding, sampling, geometric epsilon,
area estimate, clamp to bounds, or fastmath. Time is O(outline vertices) per face.

Inputs must be finite cyclically ordered convex boundary vertices, in either
winding; convexity is a precondition, not an expensive runtime reconstruction.
Repeated vertices and collinear boundary vertices are permitted. Invalid dimensions
or nonfinite inputs raise `ValueError`. A triangle whose signed determinant is
exactly zero in float64 returns zero. Fewer than three polygon vertices return
zero. Nonfinite determinant, crossing, and output arithmetic is reported as
`FloatingPointError`, rather than turning an overflow into an empty intersection.
This is ordinary floating-point geometry, not exact predicates: underflow,
unrepresentable offsets, or an ill-conditioned determinant can exhaust float64
precision. No near-zero determinant threshold is introduced.

The optimized version uses two scalar buffers. A half-plane clip can at most
double the current vertex count, so allocating `8*n` slots safely bounds three
stages even with redundant vertices. Numba is already a repository dependency;
compilation is cached. No parallel Numba loop or GPU is used.

`coverage.py` reads `settings.INTERSECTION_BACKEND`, defaulting to
`"sutherland_hodgman"`; the alternate values are `"reference"` and `"pypex"`. The new module is imported
lazily only on selection. Both INI loading and `settings.configure` reject unknown
backend values, and the setting is included in settings serialization. It replaces
only the intersection-area array inside the original partial coverage function.

```python
from elisa import settings
settings.configure(INTERSECTION_BACKEND="sutherland_hodgman")
```

The equivalent configuration, documented in `src/elisa/conf/elisa_conf_docs.ini`
and the example `src/elisa/conf/config.ini`, is:

```ini
[computational]
intersection_backend = sutherland_hodgman
```

The default is `sutherland_hodgman` (the compiled clipping implementation).
Numba is the compiler, not the geometric algorithm name. Settings are process-global, like other ELISA options;
configure before starting an observation. The experiment scripts patch the setting
and restore it after each comparison, using one worker as originally measured.
`Polygon.intersection`, `linter.intersection`, `linter.intersections`, and SAT are
unchanged. No unrelated bug fixes are included.

The timing data below were collected with the original private module selector,
before its promotion to settings and subsequent user-requested default switch.
Historical artifact keys `numba` identify the compiled Sutherland-Hodgman backend;
`numba` is no longer a valid setting value. The geometry kernel and
coverage formulas are unchanged; timings were not remeasured after this settings
integration. The 174 geometry/regression tests were rerun after migration, and
39 configuration tests passed, including backend INI loading, default fallback,
programmatic selection, serialization and rejection of invalid values.

## Independent correctness checks

`unittests/test_convex_clipping.py` supplies analytic cases and uses the independent
exact-rational oracle in `unittests/utils.py`. The oracle converts every supplied binary64 coordinate
exactly to `Fraction`, clips the TRIANGLE against every POLYGON edge (the opposite
operand direction), and evaluates area with rational arithmetic before converting
the final result to float. Pypex is never the sole reference. This shares the
half-plane mathematical principle but not floating-point predicates, rounding,
clip order, or arithmetic with the implementation.

Cases cover identical triangles, disjoint inputs, partial overlap, containment in
both directions, vertex and edge contact, coincident edges, redundant and collinear
vertices, nearly parallel boundaries, a tiny strictly positive overlap, nearly
complete overlap, reversed winding, cyclic shifts, rotation by 0.37 radians,
translation, scales 1e-9/1/1e9, a unit geometry translated by 1e9, 120-vertex convex
outlines with 12 deterministic seeds, collinear/repeated-point triangles, malformed
inputs, empty batches, and reported overflow. Input arrays are not rounded.

The ordinary transformation tests compare against the oracle for the ACTUAL
transformed binary64 coordinates. Their conservative forward-error budget is
`64 * eps * (n+3) * scale²`; the large-offset test uses
`64 * eps * offset * extent`, reflecting subtraction error. These are validation
error budgets, not clipping tolerances. The tiny positive examples also require
strict positivity and relative agreement with budgets derived from coordinate
ulps divided by feature width (no absolute tolerance that accepts zero).
Area bounds are checked against independently calculated operand areas. A rotated
exactly collinear mathematical triangle can become slightly noncollinear after
coordinate rounding; the oracle evaluates the represented coordinates.

The retained fixture `unittests/data/clipping/eclipse_geometry.npz` contains all
projected points, selected face vertex indices, normals, outlines, triangles,
legacy/reference/compiled areas, and nonempty overlap classifications from the
capture runs. Tests replay every batch, compare reference and Numba, check exact
rational samples including tiny overlaps, and run the original partial coverage
function through both optional backends. `analyze.py` additionally audits EVERY
captured face against rational geometry and diagnoses legacy quantization.

Final targeted validation: **174 passed**, including all new geometry tests,
the existing Pypex suite, binary light-curve support tests, and face visibility
regressions. One pre-existing dateutil UTC deprecation warning occurred. The full
repository suite was not run. Ruff passes on both modified production modules;
benchmark/test scripts were formatted and checked for syntax and undefined names.

## Environment and measurement protocol

"""
    text += table(
        ["Item", "Measured value"],
        [
            ["Python", env["python"].split("\n")[0]],
            ["Executable", env["executable"]],
            ["CPU", env["cpu"].strip(":\t ")],
            ["Platform", env["platform"]],
            ["NumPy / SciPy", env["numpy"] + " / " + env["scipy"]],
            ["Numba / Astropy", env["numba"] + " / " + env["astropy"]],
            ["ELISA workers", env["workers"]],
            ["Limb-darkening tables", env["ld_tables"]],
        ],
    )
    text += """
The existing project environment `/home/mike/Work/envs/elisa312/bin/python` was
used with `PYTHONPATH=src:.`, and the configured local limb-darkening data.
Installed versions are recorded above, rather than inferred from current project
dependency floors. All backends use identical physical/approximation settings and
the same observer per system. Each backend receives a full warm-up; five wall-clock
runs alternate forward/reverse backend ordering. Primary timings exclude capture,
exact arithmetic, cProfile, and first compilation. There is no CPU affinity or
frequency lock; min/max and standard deviations are retained in `results.json`.
Small differences in unchanged functions' profiles can reflect instrumentation,
CPU/cache state, and timing noise; they are not claimed as new optimizations.

"""
    text += f"Fresh-cache JIT first call: **{compilation['cold_first_call_s']:.3f} s** (including batch and scalar compilation, excluding Python imports). "
    text += f"The experiment-process first call took {r['numba_first_call_s']:.3f} s; its default cache state is not a cold-compile guarantee. "
    text += f"Median warm one-face validated dispatch in the separate compilation probe was {np.median(compilation['warm_samples_s']) * 1e6:.2f} µs. "
    text += "Compilation overhead is separate from all warm timing tables.\n\n## Binary configurations\n\n"
    text += """The baseline is exactly the requested construction: primary mass 2.15 solMass,
potential 3.6, temperature 10000 K, gravity darkening 1, explicit 5 degree
mesh; secondary mass 1.2 solMass, potential 4, temperature 7000 K, gravity
darkening 1, secondary discretization omitted. Both have albedo 0.6, metallicity
0, and blackbody atmosphere. Orbital parameters are argument of periastron 58 deg,
gamma -30.7 km/s, period 2.5 d, eccentricity 0, inclination 85 deg, distance 155 pc,
primary minimum 2440000 d, and phase shift 0. The observer uses Generic.Bessell.U,
from -0.6 to 0.6 with step 0.01 and normalize=True. The repository generates
**120 requested phases** (the upper endpoint is excluded), and its existing phase
reduction/symmetry evaluates 51 positions in the measured profile.

The omitted secondary mesh is automatically adjusted by existing ELISA logic to
8 degrees. This is the result of the mandated configuration, not an experiment
substitution. Additional systems change only inclination or mesh; masses and
potentials remain identical. Every construction asserts `morphology == "detached"`,
`eccentricity == 0`, and both synchronicities equal 1. No prohibited physical
regime was constructed. The 80 degree variant is labeled by inclination, without
claiming that it is a mathematically grazing eclipse.

"""
    text += table(
        ["System", "Inclination deg", "Meshes deg P/S", "Mesh faces P/S", "Phases", "Verified morphology"],
        [
            [
                n,
                v["inclination_deg"],
                " / ".join(f"{x:g}" for x in v["discretization_deg"]),
                " / ".join(map(str, v["mesh_faces"])),
                v["phase_count"],
                v["morphology"],
            ]
            for n, v in r["systems"].items()
        ],
    )
    text += "\n## Captured geometry and scientific area differences\n\n"
    text += f"Captured **{int(data['batches'])} batches**, including {int(data['batches']) - len(batches)} with no selected triangles; **{sum(b['faces'] for b in batches)} face intersections** across {len(batches)} nonempty batches. "
    text += "All selected face intersections in this workload are partial; there are zero individual empty or full-containment intersections. Empty face batches are retained and timed. Containment and empty-intersection correctness are established analytically, but no real-workload speed for those absent classes is claimed.\n\n"
    rows = []
    for name, v in r["systems"].items():
        bs = [b for b in batches if b["system"] == name]
        rows.append(
            [
                name,
                sum(b["faces"] for b in bs),
                f"{min(b['outline_vertices'] for b in bs)}–{max(b['outline_vertices'] for b in bs)}",
                f"{max(b['max_abs'] for b in bs):.6e}",
                f"{max(b['max_relative_meaningful'] for b in bs):.6e}",
                sum(b["significant_faces"] for b in bs),
                f"{min(b['min_positive'] for b in bs):.6e}",
            ]
        )
    text += table(
        [
            "System",
            "Triangles",
            "Outline vertices",
            "Max |Pypex-new| SMA²",
            "Max relative overlap diff",
            "Significant faces",
            "Smallest positive overlap SMA²",
        ],
        rows,
    )
    text += """
The reference and Numba areas are bitwise identical for every captured triangle.
All new and legacy areas obey `0 <= intersection <= min(operand areas)` within
the scale-dependent forward-error budget; there are **zero bound violations**.
"Significant" here means exceeding `64*eps*h*outline_extent²`, not an astrophysical
acceptance threshold; it does not alter geometry or face selection. Relative
intersection differences omit intersections smaller than this budget. Complete
face offsets and phase distributions are in `results.json`; offsets index the
selected triangle arrays, whose original vertex triplets are retained in the NPZ.

"""
    text += table(
        [
            "System",
            "Max corrected coverage difference SMA²",
            "Max relative visible coverage diff",
            "Max corrected difference m²",
        ],
        [
            [
                n,
                f"{max(b['coverage_max_abs'] for b in batches if b['system'] == n):.6e}",
                f"{max(b['coverage_max_relative_meaningful'] for b in batches if b['system'] == n):.6e}",
                f"{max(b['coverage_max_abs'] for b in batches if b['system'] == n) * v['semi_major_axis_m'] ** 2:.6e}",
            ]
            for n, v in r["systems"].items()
        ],
    )
    text += """
These corrected differences precede the unchanged noise threshold and SI scaling.
Relative visible coverage uses projected residual area above the same error
budget, and the cosine factor cancels. Near the limb, division by a small cosine
amplifies absolute area differences; nearly hidden triangles can also have large
relative residual differences. Therefore the overlap, corrected coverage, and
flux errors are reported separately rather than dismissed on one small scalar.

### Independent discrepancy investigation

"""
    text += table(
        ["Exact-rational audit over all 2550 faces", "Maximum absolute error SMA²"],
        [[k, f"{v:.9e}"] for k, v in audit["max_abs"].items()],
    )
    text += """
`legacy_error` and `clipping_error` compare with the exact intersection of the
represented input coordinates. `rounded_oracle_error` takes exact intersection
vertices, rounds to 10 decimals, and applies an uncentered shoelace sum.
`legacy_minus_rounded_oracle` is the remaining difference from Pypex. This
reproduces the legacy discrepancies to approximately 1e-18, demonstrating that
coordinate quantization dominates this captured workload. As a separate
read-only diagnostic, calling existing `Polygon.intersection(round_tol=15)`
reduces the maximum legacy error to the value in `legacy_round15_error`.
Neither setting nor Pypex was modified in production or primary benchmarks.

No captured disagreement requires a changed face classifier, substantial clipping
failure, or legacy bug fix. This is evidence for this workload, not proof that
all legacy predicate behavior is accurate. Analytic touching/degenerate cases are
separate from genuine tiny positive intersections; no small area is zeroed by the
new algorithm. The exact audit's worst legacy and clipping examples, with batch
and face offsets, are retained in `exact_audit.json`.

### Distribution over sampled eclipse phases

Each row refers to the existing circular symmetry/reduction evaluation, not all
120 remapped output phases. Phase 0.07–0.08 and phase 0.42–0.43 sample the outer
contacts in these runs; the opposite branches are remapped by existing ELISA
logic. The complete zero-face batches are in the NPZ.

"""
    text += table(
        ["System", "Evaluated phase", "Faces", "Significant differences", "Max area diff SMA²"],
        [
            [b["system"], f"{b['phase']:.2f}", b["faces"], b["significant_faces"], f"{b['max_abs']:.3e}"]
            for b in batches
        ],
    )
    text += "\n## A. Isolated geometry performance\n\n"
    text += """The measured workload is the same captured outline and triangle arrays for all
three algorithms. Pypex timing includes hull/face objects, intersection
reconstruction, and area extraction; the new paths include float64 conversion,
finite/shape validation, buffers and area arrays. This is an area-interface
comparison, not a claim about arithmetic-only instruction throughput. Each run
replays all batches including zero-face preparation. Five runs per backend.

"""
    text += table(
        ["Backend", "Median ms", "Min–max ms", "Std ms", "Intersections/s", "Speedup vs Pypex"],
        [
            [
                n,
                f"{t['median_s'] * 1e3:.4f}",
                f"{t['min_s'] * 1e3:.4f}–{t['max_s'] * 1e3:.4f}",
                f"{t['std_s'] * 1e3:.4f}",
                f"{t['intersections_per_s']:,.0f}",
                f"{r['kernel_timing']['pypex']['median_s'] / t['median_s']:.2f}x",
            ]
            for n, t in r["kernel_timing"].items()
        ],
    )
    text += "\nPerformance by outline size (all observed classes are partial intersections):\n\n"
    text += table(
        ["Vertices/type", "Faces", "Pypex median µs/face", "Reference µs/face", "Numba µs/face", "Numba speedup"],
        [
            [
                k,
                v["count"],
                *[f"{v[n]['median_s'] / v['count'] * 1e6:.3f}" for n in ("pypex", "reference", "numba")],
                f"{v['pypex']['median_s'] / v['numba']['median_s']:.1f}x",
            ]
            for k, v in r["group_timing"].items()
        ],
    )
    text += """
Each size group retains original batches; batch overhead is divided by actual
face count. Small groups are sensitive to timer and allocation overhead. Full
sample ranges, standard deviations and rates for every group are in
`results.json`. There were no actual empty/contained faces from which to measure
those category rates; no synthetic category timings are presented as real data.

## B. Complete partial eclipse coverage

The original `partial_visible_faces_surface_coverage` is replayed with identical
projected points, selected face triplets, normals and outlines, changing only the
backend selector. These timings include polygon preparation/validation, projected
triangle area, cosine correction, subtraction/division and output arrays. They
do not include the unchanged face classifier; complete coverage and hull costs
are separately visible in the full-model profiles.

"""
    text += table(
        ["Backend", "Median ms", "Min–max ms", "Std ms", "Speedup vs Pypex"],
        [
            [
                n,
                f"{t['median_s'] * 1e3:.4f}",
                f"{t['min_s'] * 1e3:.4f}–{t['max_s'] * 1e3:.4f}",
                f"{t['std_s'] * 1e3:.4f}",
                f"{r['coverage_timing']['pypex']['median_s'] / t['median_s']:.2f}x",
            ]
            for n, t in r["coverage_timing"].items()
        ],
    )
    text += "\n## C. Complete warm light curves\n\n"
    text += table(
        ["System", "Backend", "Median s", "Min–max s", "Std s", "Speedup vs Pypex"],
        [
            [
                n,
                backend,
                f"{t['median_s']:.6f}",
                f"{t['min_s']:.6f}–{t['max_s']:.6f}",
                f"{t['std_s']:.6f}",
                f"{v['timing']['pypex']['median_s'] / t['median_s']:.3f}x",
            ]
            for n, v in r["systems"].items()
            for backend, t in v["timing"].items()
        ],
    )
    text += """
The clear Python reference is approximately as expensive as legacy Pypex here;
the large speedup requires compilation and batching. Kernel speedup does not
translate directly to whole-model speedup because other calculations remain.

Separate cProfile runs for both backends are retained in
`profile_{system}_{backend}.txt`. For the baseline, Pypex's partial coverage has
0.271 s cumulative profile time (928 `Polygon.intersection` calls, 0.236 s;
`linter.intersections`, 0.137 s). Complete `compute_surface_coverage` drops from
0.308 s to 0.038 s in the optimized profile. After clipping optimization,
`move_sys_onpos` (0.071 s), deep copies (0.043 s), and reflection (0.036 s) are
prominent. These cumulative times overlap and must not be summed or substituted
for unprofiled speedups. Changes in timing of these unchanged functions are not
attributed to the new geometry code.

## Complete light-curve scientific comparison

Both optional backends give bitwise identical flux arrays on these systems.
The table compares compiled Sutherland-Hodgman with the original Pypex baseline. Raw flux is separately
computed with `normalize=False, flux_unit=u.W/u.m**2`: explicitly specifying
units is necessary because Observer retains dimensionless units after a
normalized observation. Normalized flux uses the same observer normalization as
the requested baseline. Ppm is `1e6 * abs(normalized flux difference)`.

"""
    text += table(
        [
            "System",
            "Max raw |ΔF| W/m²",
            "Max normalized |ΔF|",
            "Max ppm",
            "Phase of max",
            "Contact-neighborhood max |ΔF|",
        ],
        [
            [
                n,
                f"{v['max_absolute_raw_flux_difference']:.9e}",
                f"{v['flux_comparison']['numba']['max_abs_normalized']:.9e}",
                f"{v['flux_comparison']['numba']['ppm']:.9e}",
                f"{v['flux_comparison']['numba']['max_difference_phase']:.2f}",
                f"{v['flux_comparison']['numba']['max_contact_abs']:.9e}",
            ]
            for n, v in r["systems"].items()
        ],
    )
    text += """
The contact-neighborhood metric uses `0.04 < abs(phase - round(2*phase)/2) < 0.14`
on the sampled output grid. This is an explicitly defined window around ingress
and egress, not a solved contact-time estimate. The following depth changes use
`depth = 1 - min(normalized flux)` within 0.1 phase of each eclipse center;
positive means the new curve is deeper. NPZ flux arrays retain every phase for
independent inspection and recomputation.

"""
    text += table(
        ["System", "Depth(new)-depth(Pypex), eclipse near 0", "Depth difference, eclipse near 0.5"],
        [
            [n, *[f"{x:.9e}" for x in v["flux_comparison"]["numba"]["depth_differences"]]]
            for n, v in r["systems"].items()
        ],
    )
    text += """
These nonzero flux differences have a demonstrated geometric source: legacy
intersection coordinate rounding, subsequently propagated through the original
cosine, surface thresholds, radiance weighting and normalization. They are not
merely assumed harmless floating-point noise. Exact-rational area validation
supports the new areas on every captured face. This experiment does not establish
a universal astrophysical acceptance criterion for other systems or passbands.

## Files and reproduction

Modified:

* `src/elisa/binary_system/surface/coverage.py`: `_calculate_intersection_areas`
  encapsulates settings-based backend selection; partial coverage calls it once.
* `src/elisa/managers/settings_manager.py`: default `INTERSECTION_BACKEND="sutherland_hodgman"`.
* `src/elisa/conf/settings.py`: INI loading, validation and serialization.
* `src/elisa/conf/elisa_conf_docs.ini`, `src/elisa/conf/config.ini`: documented
  `[computational]` option and configuration example.
* `unittests/cfg/test_settings.py`: configuration coverage for the backend option.

Added:

* `src/elisa/binary_system/surface/clipping.py`: reference and compiled kernels.
* `unittests/test_convex_clipping.py`: analytic, rational, transformed, captured
  and private integration tests.
* `unittests/data/clipping/eclipse_geometry.npz`: exact real-input fixture.
* `unittests/utils.py`: independent rational geometry.
* `scripts/benchmark/clipping/eclipse_geometry.npz`: generated benchmark capture.
* `scripts/benchmark/clipping/experiment.py`: mandatory baseline first, two
  detached variants, capture, scientific comparisons, all timing levels/profiles.
* `scripts/benchmark/clipping/analyze.py`: every-face exact audit and rounding diagnosis.
* `scripts/benchmark/clipping/compilation.py`: separate fresh-cache compilation probe.
* `scripts/benchmark/clipping/report.py`: renders this report from artifacts.
* `scripts/benchmark/clipping/results.json`, `exact_audit.json`, `compilation.json`:
  measured timing samples, geometry/phase distributions, oracle checks.
* `scripts/benchmark/clipping/flux_{system}.npz`: normalized and raw output arrays.
* `scripts/benchmark/clipping/profile_{system}_{backend}.txt`: separate profiles.
* `scripts/benchmark/clipping/tests.log`, `run.log`, `compilation.log`,
  `exact_audit.log`: validation and measurement logs.
* `PYPEX_CLIPPING_EXPERIMENT.md`: this report.

Existing untracked user files, including the audit documents, were preserved.
Only obsolete artifacts created by this experiment inside the repository were
removed; nothing outside the authorized project or `/tmp` was deleted.

From the repository root:

```bash
export PYTHONPATH=src:.
/home/mike/Work/envs/elisa312/bin/python scripts/benchmark/clipping/compilation.py
/home/mike/Work/envs/elisa312/bin/python scripts/benchmark/clipping/experiment.py --repeats 5
/home/mike/Work/envs/elisa312/bin/python scripts/benchmark/clipping/analyze.py
/home/mike/Work/envs/elisa312/bin/python -m pytest unittests/test_convex_clipping.py unittests/pypex unittests/binary/test_binary_system_lc.py::SupportMethodsTestCase unittests/test_face_visibility.py -q
/home/mike/Work/envs/elisa312/bin/python scripts/benchmark/clipping/report.py
```

The scripts write only repository artifacts and a fresh Numba cache under `/tmp`.
Benchmark captures are written to `scripts/benchmark/clipping/eclipse_geometry.npz`,
independently of the committed test fixture. Unit tests keep their reference in
`unittests/utils.py`; benchmark validation has its own reference inside
`analyze.py`. Neither side imports the other. Benchmark scripts do not overwrite
test data, and deleting the benchmark directory does not affect unit tests.
The environment's existing configured external LD tables are read, not modified.

## Limitations and recommendation

Measured: accurate geometric areas on analytic/transformed tests and every
captured triangle, zero scale-budget bound violations, roughly twofold whole-model
speedup, and much larger area-interface and partial-coverage speedups. No actual
captured containment or empty-face intersections exist in this selected workload;
no performance conclusion for those classes is inferred from nonexistent data.
Only three systems, one passband, and the stated mesh/phase resolutions were
measured. Coarse contact sampling is not an exhaustive near-tangency campaign.
The unchanged face selector, silhouette discretization, thresholds and physical
model were deliberately not audited or improved by this implementation.

Ordinary binary64 clipping does not promise exact topology under arbitrarily
ill-conditioned inputs, huge offsets, determinant cancellation, or underflow.
Tests and exact captures establish normal floating-point accuracy in the tested
domain, not exact-predicate robustness over all finite binary64 values. Convexity
and cyclic ordering remain required private-input contracts. Settings integration
was regression-tested, but multiprocessing performance/propagation was not part
of the one-worker experiment.

The original experimental recommendation was to keep Pypex as the default pending
wider validation. The user subsequently requested adoption: **Sutherland-Hodgman
is now the default**, with Pypex still available through `intersection_backend =
pypex`. Wider detached geometries, contact sampling, and an explicit scientific
accuracy policy for legacy quantization differences remain useful follow-up work. The expected benefit for
other mesh sizes is an inference from O(h) clipping, not an additional measured
speedup. The later default switch is explicitly user-requested; it adds no geometric approximation.
"""
    (ROOT / "PYPEX_CLIPPING_EXPERIMENT.md").write_text(text)


if __name__ == "__main__":
    main()
