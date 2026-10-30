# Limb-darkening performance and science benchmark

Edit the uppercase settings in `limb_darkening.py`, then run it **without arguments** from the repository root:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
NUMBA_CACHE_DIR=/tmp/elisa-ld-benchmark/numba \
/home/mike/Work/envs/elisa/elisa314/bin/python -B scripts/benchmark/speed/limb_darkening.py
```

Use your existing project interpreter on another machine. Nothing is installed. A complete six-round comparison of all
four models takes several minutes on one worker. The script patches helpers only inside its process and restores them;
it does not modify production files or physics.

## Settings and inputs

```python
MODE = "all"  # "all", "baseline", "direct", "optimized"
OUTPUT_DIR = Path("/tmp/elisa-ld-benchmark/review")
REPETITIONS = 6
SCIENCE_ONLY = False
MICRO_ONLY = False
PROFILE = False
COMPARE_DIR: Path | None = None
CASES = ("circular", "eccentric", "multiband", "single")

```

| Setting        | Purpose                                                                                                                                                                          |
|----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `MODE`         | `all` compares all three implementations in one balanced/interleaved run. A single mode is useful for reference generation or diagnostics, not standalone fair cross-run timing. |
| `OUTPUT_DIR`   | Created automatically. Matching output filenames are overwritten.                                                                                                                |
| `REPETITIONS`  | Measurement rounds, not total calls. Use a multiple of **6** for full three-mode order balance; incomplete balance emits a warning and is recorded.                              |
| `SCIENCE_ONLY` | Skip full-curve timing samples while retaining scientific captures and comparison checks.                                                                                        |
| `MICRO_ONLY`   | Run only the balanced component-interpolation microbenchmark. Takes precedence over `SCIENCE_ONLY`; no full curves/profiles are produced.                                        |
| `PROFILE`      | Separate cProfile runs after measurements, with an untimed full-curve warmup. Profiling is never included in wall-clock samples.                                                 |
| `COMPARE_DIR`  | Optional directory containing versioned baseline/direct NPZs. Requires optimized full curves. `None` compares freshly generated modes in memory when `MODE="all"`.               |
| `CASES`        | Choose from `circular`, `eccentric`, `multiband`, `single`; changing this selection changes the strict reference fingerprint.                                                    |

Required inputs are this ELISA checkout and packaged data/passbands, installed dependencies, and these Van Hamme CSVs under `settings.LD_TABLES`:

- `lin.Generic.Bessell.U.p00.csv`
- `lin.Generic.Bessell.V.p00.csv`
- `lin.bolometric.p00.csv`

On the audited machine, the directory is `/home/mike/.elisa/limbdarkening/ld`. These are physical input tables, not generated benchmark results. The models use blackbody atmospheres; external CK04 grids are unnecessary. Neither `profil.py`, unit tests nor `data/test_binary_*.json` is imported/read.

The script uses normal ELISA configuration and requires cosine LD, one worker, per-face coefficients, reflection enabled
and two reflection iterations. It does not change those settings. The binary modes use the same model factory: the
audit's 2.15/1.2 solar-mass, 10000/7000 K binary, potentials 3.6/4.0, synchronous rotation, inclination 85°, primary
discretization 5°, secondary discretization omitted. Binary cases are circular U, eccentricity 0.2 U, and circular U+V.

The `single` case is a non-pulsating, unspotted `SingleSystem`: mass 2.15 solar masses, effective temperature 10000 K,
polar log(g/cgs) 4.2, gravity-darkening exponent 1, rotation period 0.7 days, inclination 65°, discretization 5°, solar
metallicity, blackbody atmosphere and distance 155 pc, observed in U. Rotation produces non-uniform face temperatures
and gravity. Scientific checks require temperature variation greater than 1 K, log(g) variation greater than 0.001 dex,
varying LD coefficients and a coefficient row for every face. All four cases use phases -0.6 to 0.6 with step 0.01.

## What each implementation measures

| Mode | Component/result cache | Source interpolation |
|---|---|---|
| `baseline` | Benchmark-local reproduction of the original flattened, nine-decimal-rounded Python float tuples and 128-entry LRU; original reconstruction and mirroring/ownership behavior. No new byte keys, source identity checks or read-only/copy handling are charged to this mode. | Original filename-keyed frame buffer, ordered concatenation, full-selected-row deduplication, column selection and `griddata(method="linear")` on every necessary evaluation. |
| `direct` | Current production exact float64 keys, source/configuration identity, 128-entry result LRU and ownership handling. | The same legacy preparation/`griddata` path as baseline. Queries are unrounded. |
| `optimized` | Current production component/result cache, exactly as in direct. | Current production source-interpolator cache. |

Direct pre-reflection bolometric queries stay unrounded in **all** modes. Baseline rounding occurs only at
component-result key generation, before evaluation, including misses. The baseline patches the imported component-helper
aliases in both binary and single-system curve utilities; patching only the common helper would leave SingleSystem
accidentally running the optimized path. Its helper reproduces the original
`ravel → round(9) → astype(float) → tolist → tuple` sequence. Native coefficient shapes, coefficient order, source row
order, metallicity table concatenation, band aliases, gravity conversion and error messages are preserved.

**Interpretation:** `direct` versus `optimized` isolates structure-cache reuse at identical coordinates and current component-cache machinery. `baseline` versus `direct` includes both rounding removal and the new result-key/identity/ownership implementation; its timing difference is not a pure rounding cost. `baseline` versus `optimized` measures the combined LD changes. The local baseline ports the original LD helpers for these fixed, non-custom models; the rest of the engine is the current checkout. It does not recreate an entire historical release, and explicitly rejects custom coefficients in its component helper.

Because baseline intentionally retains the old filename-only source buffer and incomplete result key, assets/settings must remain fixed. Each measurement resets both old/new LD caches; metadata is rechecked at the end. Baseline is not a proposed safe application cache implementation.

## Balanced timing and cache states

All modes run in one process, with a separately constructed equivalent observer per mode. Model construction, metadata collection, warmups, cache setup, instrumentation and profiling are outside the timers. An initial untimed curve also stabilizes the engine's NumPy error policy before provenance is recorded; each mode then receives an untimed full curve to warm imports, Numba and non-LD model caches. The production Lagrange-point code can leave divide/invalid handling set to `raise`; the benchmark records and verifies the resulting common policy rather than changing production code.

Each round uses a different permutation of the three modes, cycling all six permutations. Every mode occupies each timing position equally over six rounds; predecessor order within a block is balanced too. The three full-curve scenarios rotate through first/second/third position across rounds. Exact order, round, position, raw seconds and cache deltas are saved. Compare matched rounds/states and inspect distributions, not a single minimum or separate single-mode batches.

| Full-curve state | Setup outside timer | What the timed curve contains |
|---|---|---|
| `cold_source` | Clear LD frames, source interpolators and both result LRUs. | CSV parsing/source preparation, structure initialization, current queries and the rest of the LC. OS page caches and Numba may already be warm. |
| `warm_source_new_result` | Clear LD caches; interpolate one supported query for all needed source bands, without filling a component result LRU. | Empty result LRU, loaded source frames, and (optimized only) initialized source interpolators. Surface queries differ from the priming query. Result hits must be zero; optimized source misses must be zero. |
| `repeated_curve` | Clear LD caches; compute a complete untimed normalized curve with that observer. | A complete repeat with whatever exact result-cache reuse the real working set allows. Source preparation stays warm. |

In baseline/direct, “warm source” means **loaded DataFrames**, since those implementations have no source-structure cache. Their triangulation is still rebuilt on every necessary evaluation. Cold samples do not claim cold filesystem, interpreter, compilation or CPU hardware caches.

A repeated circular curve has four result hits and two uncached reflection evaluations. An eccentric curve generally exceeds the 128-entry result capacity and can have **zero result hits even on a complete repeat**. Its `repeated_curve` timing must not be labelled a warm-result-cache speedup. Actual per-sample deltas make this distinction explicit.

The non-pulsating single star prepares observed U and bolometric-correction coefficients once before its rotational
phase loop. It has no binary reflection calls. Each cold/new-result curve must have two result misses; a complete repeat
must have two result hits and no evaluations or constructions. Optimized cold sources require two constructions,
warm-source/new-result requests require two source hits with no source misses, and exact result hits bypass the source
cache entirely. Timing-state assertions enforce these counts in addition to the independently instrumented scientific
runs.

With `MICRO_ONLY=True`, the benchmark instead interleaves requests for 400 supported query points through the same component-helper call site. It separately measures `cold_source`, `warm_source_new_result` (perturbed temperatures, exactly one result miss), and `warm_result_cache` (identical request, exactly one result hit). Cache states are asserted. These microseconds are not whole-LC speedups, and individual warm-hit samples can be noisy; use more rounds for microbenchmark precision.

Interleaving reduces order/thermal drift; it does not lock CPU frequency, pin a core, flush hardware caches or remove operating-system noise. Run without competing workloads. Library builds, thread environment and process CPU affinity are recorded; repeated paired/interleaved measurements remain local evidence, not portable performance guarantees.

## Scientific captures and actual work counts

Scientific runs are separate from timed/profiled runs, beginning with empty LD caches. Arrays are labelled by **stage and sequence**, independently of result misses:

- `reflection_NNN_*`: temperatures, log(SI) gravity and native `(coefficient, face)` bolometric matrices **before reflection**, at every reflection request.
- `surface_NNN_observed_*`: full-face temperatures/gravity, selected component query coordinates and full-face
  observed-band coefficient matrices. Binary values are post-reflection; single-star values are gravity-darkened without
  irradiation.
- `surface_NNN_bolometric_correction_*`: the corresponding bolometric-correction stage.
- `geometry_NNN`: position record associated with that surface preparation. Single-star captures contain only meaningful
  `(index, azimuth, phase)` fields; binary-only distance and true-anomaly placeholders are NaN and deliberately omitted.
- `phases`, `raw_flux_BAND`, `normalized_flux_BAND`: copied arrays, with raw flux explicitly requested in W/m². Passing `normalize=False` alone is insufficient after an Observer has retained dimensionless units.

Stage wrappers select `single_system.curves.utils` or `binary_system.curves.utils` from the observer's actual system
type. They capture every component group, **including result-cache hits**. Nonzero preparation counts are mandatory, so
missed instrumentation cannot pass vacuously. Single-star captures must contain exactly one preparation, two component
groups, zero reflection requests, two cold misses and two repeat hits; constructor/evaluation counts are reconciled
independently against these cache deltas. A separate normalized LC is run with the warm state left by the raw LC. Its
complete set of intrinsic stage keys, phase, geometry, temperatures, coordinates and coefficient captures must match
their raw-curve stages exactly before its normalized flux is accepted. Raw/normalized work counts and LRU deltas are
reported separately.

Constructor/evaluation wrappers instrument both SciPy's `griddata` constructor reference and the production direct constructor. Counts therefore reflect actual constructions and evaluations, not an assumed equality between result misses and source builds. Point counts, reflection requests, surface preparations and component groups are also recorded. Assertions reconcile those counts against each cache layer; no instrumented timing is used for performance claims.

Direct and optimized artifacts must agree **exactly** on every array, shape and dtype in this environment. Baseline
differences in rounded query coordinates/coefficients/flux are measured, while phases, geometry, physical temperatures
and surface gravity must still match. Surface-gravity differences are reported separately from query-coordinate
differences. The baseline/direct numerical difference comes from evaluating rounded versus original component
coordinates; exact direct/optimized equality isolates the structure-cache change. The axisymmetric unspotted single star
has a constant normalized curve, so normalized flux alone is a weak regression check: raw flux and the full spatially
varying coefficient/temperature/gravity arrays are also required. This case does not cover spots, pulsations or changing
single-star surface parameters. No tolerance, clipping or extrapolation hides discrepancies. Built-in checks require the
original `LimbDarkeningError` and message for out-of-hull/NaN queries and `FileNotFoundError` for missing tables; native
lower-level interpolation exceptions propagate.

## Provenance and reference compatibility

Every NPZ embeds its model case, implementation mode and versioned metadata. `benchmark.json`/`micro.json` record the common metadata too:

- Actual serialized models, separately recorded resolved component discretizations in radians, passbands, phase
  arguments, uppercase settings and LD column configuration. The schema registry is identified by its stable class name,
  not its process-specific object address; schema files are hashed too.
- SHA-256 of production Python/packaged data, configured passband CSVs and selected LD source CSVs; resolved paths are included.
- Benchmark implementation fingerprint, excluding editable top-level configuration assignments, plus an artifact format version.
- Python executable/version, NumPy/SciPy/Pandas/Astropy/Numba versions, NumPy/SciPy build configuration, Astropy constants selection, solar-mass conversion, CPU model/platform/affinity, NumPy error policy and thread environment.

Hashing happens during run setup and final validation, not on timed interpolation hits. Metadata/settings must remain identical through the run. Timing samples and source-cold measurements consequently have OS-warm file contents. Only a run whose `benchmark.json` has `completed=true` is accepted as a reference. External reference loading checks the recorded fingerprint against both its contents and the current environment, and rejects wrong-mode/wrong-case artifacts. Old unversioned NPZs must be regenerated. Strict compatibility may reject numerically equivalent environments that use different paths/builds; this is intentional.

For a normal comparison, use `MODE="all"` and `COMPARE_DIR=None`: all three modes are captured and compared in the same run. For external references, first generate versioned baseline/direct artifacts with the same `CASES` and environment, then run optimized with `COMPARE_DIR` set to their directory. Do not compare independent single-mode timing batches as if they were interleaved samples.

## Files created and older `.dat` files

| File under `OUTPUT_DIR` | Purpose                                                                                                                                                                                                                                                                                   |
|-------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `benchmark.json`        | Full-curve raw timing samples/order/cache deltas, scientific construction/evaluation counts, comparisons and metadata. Updated after each completed case; a failed run can leave partial output with `completed=false`. Final provenance validation must pass before this becomes `true`. |
| `MODE_CASE.npz`         | Versioned scientific arrays and embedded provenance. Cases: circular, eccentric, multiband, single.                                                                                                                                                                                       |
| `MODE_CASE.prof`        | Separate warmed cProfile run, only if `PROFILE=True`.                                                                                                                                                                                                                                     |
| `micro.json`            | Interleaved component microbenchmark samples/cache deltas and provenance, only if `MICRO_ONLY=True`.                                                                                                                                                                                      |

`SCIENCE_ONLY=True` leaves timing sample lists empty. Existing profiles are not deleted when profiling is disabled; use a fresh directory to avoid confusing old files with new outputs. Logs are printed through normal logging/stdout; shell redirection creates log files. Numba caches use `NUMBA_CACHE_DIR`, or its normal location if unset. This version no longer generates the old per-mode summary JSON, `comparisons.json`, or hard-coded retained-memory estimates.

The older `runtime.py` uses `data/test_binary_circ.json` and `data/test_binary_ecc.json` as inputs and writes `benchmark_circ.dat` / `benchmark_ecc.dat` in its working directory. Those `.dat` files contain timing columns (`Alpha`, `n_phases`, `mean_time`, `std_time`, `min_time`, `max_time`), not physical input data. At documentation time, the circular file has one measurement and the eccentric file only a header. They predate this LD benchmark and are neither read nor changed by it. `__pycache__/` is generated bytecode. None of these older files is required for the LD benchmark.

Earlier investigation artifacts live under `/tmp/elisa-ld-optimization`: old mode outputs, logs, test results, source backups and Numba files. The current script does not depend on them. Generated benchmark outputs can be recreated with the same code/assets; `/tmp` is not permanent storage. Earlier measured results remain historical in [LIMB_DARKENING_INTERPOLATION_AUDIT.md](../../../LIMB_DARKENING_INTERPOLATION_AUDIT.md); their old separate-batch timings should not be substituted for new interleaved measurements.

## Earlier binary review checks

On the configured Python 3.14 / SciPy 1.17.1 environment:

- Completed six balanced rounds for all three full-curve states and all three models: **162 timed curves**, with exact order balance verified and final provenance validation passing.
- Direct and optimized phases, geometry, coordinates, coefficients, reflected temperatures, raw flux and normalized flux agreed exactly. Rounded-baseline differences were recorded separately.
- Independently checked **506 baseline coefficient matrices** and their rounded query keys against the saved pre-optimization interpolation functions. Also reproduced a complete original circular raw/normalized LC, all eight actual interpolation outputs across the two calls, and the four-miss/four-hit LRU sequence exactly. These frozen source copies were used only for this independent review check; the benchmark has no runtime dependency on them.
- Verified actual eccentric counts: **492 evaluations, 82 preparations and two optimized constructions** for the raw curve; **492 evaluations and zero new constructions** for its normalized repeat. Circular repeats retained four result hits and two reflection evaluations.
- Completed **54 microbenchmark samples** covering cold sources, warm-source/new-query misses and exact result hits, and exercised the separate profiling path in all three modes.
- Accepted matching references across processes; deliberately rejected changed settings, library versions, source hashes, wrong-mode/unversioned artifacts and incomplete runs. Out-of-hull, NaN and missing-source checks passed. `git diff --check` passed.

Review outputs are in `/tmp/elisa-ld-optimization/review-final`, with separate micro/profile validation outputs under `review-micro` and `review-profile`. An earlier attempt correctly failed final validation because provenance was captured before the engine stabilized NumPy's error policy; it is not an accepted reference run. Use the completed run and its metadata, not that rejected attempt.

## SingleSystem extension results

The extension was checked on the same configured Python 3.14 / SciPy 1.17.1 installation. Only this script and README
were changed. The SingleSystem and LD unit suites passed: **147 tests**. The complete four-case benchmark finished **216
timed curves** (six rounds × three states × three modes × four cases), passed final provenance validation, and
independently verified each mode occupied every timing position twice per state/case. Direct and optimized matched
exactly on all captured arrays in every case; no scientific regression was observed. Out-of-hull, NaN and missing-source
checks passed in all modes.

The single star had 3,080 full-surface faces, temperatures 9807.51–10348.75 K, log(SI gravity) 2.10606–2.19937, and U
coefficients 0.453831–0.465729. Its 14 captured arrays include 120 phases (-0.6 through 0.59), both observed and
bolometric-correction stages, raw flux and normalized flux. Removing rounding changed query coordinates by at most
4.98403e-10, LD coefficients by 8.88023e-12 absolute, and raw flux by 1.21048e-24 W/m² (3.51959e-13 relative). Physical
temperatures, gravity, phases, geometry and normalized flux were unchanged. Comparing baseline against direct unrounded
`griddata` gives the same differences; structure caching introduced zero additional numerical difference.

Median complete single-star LC times, in milliseconds (six unprofiled samples per cell):

| Source/result state     | Baseline | Direct | Optimized |
|-------------------------|---------:|-------:|----------:|
| Cold source             |   14.134 | 14.342 |    14.290 |
| Warm source, new result |   13.121 | 13.183 |     5.540 |
| Repeated curve          |    5.103 |  5.189 |     5.243 |

Warm-source/new-result matched-round direct/optimized speed ratios had a median of **2.37×** for the whole LC. Cold
initialization was effectively unchanged because both implementations still need two interpolators. Repeated-result
timings showed a small increase, not a speedup: optimized was about 1.0% above direct and 2.7% above baseline by median.
The direct/optimized ranges overlapped (5.124–5.495 versus 5.050–5.413 ms); six samples do not establish a statistically
significant regression at this scale. Exact result hits bypass interpolation in all modes, leaving surface/model
overhead and result-key/ownership costs.

Independent scientific instrumentation recorded the same single-star work counts in each mode: one preparation, two
component groups and zero reflection calls per curve; the cold raw curve performed two constructions, two evaluations (
770 selected query points total) and two result misses. Its normalized repeat performed zero constructions/evaluations
and two result hits. The optimized source cache had two cold misses; warm-source/new-result timing samples had two
source hits and no misses, while repeated-result hits did not access the source interpolator cache. A deliberate missed
SingleSystem baseline-alias patch was rejected by the LRU assertions, confirming that the new path cannot silently
escape instrumentation. Versioned single-star baseline/direct reference artifacts also loaded successfully.

Results and metadata: `/tmp/elisa-ld-optimization/single-extension-balanced/benchmark.json` and `MODE_CASE.npz` in that
directory. The same run also revalidated the circular, eccentric and multiband scientific comparisons and cache counts.
Unit-test output is `/tmp/elisa-ld-optimization/single-extension-tests.txt`; benchmark output is
`/tmp/elisa-ld-optimization/single-extension-benchmark.log`. These are generated local artifacts, not required inputs.

The retired production CSV readers `get_ld_table` and `get_ld_table_by_name` are
preserved locally in this benchmark script. The historical baseline uses the local
filename reader; both readers raise builtin `FileNotFoundError` directly and are
independent of ELISA's exception helpers and production source cache.
