# Polygon intersection benchmarks and report

This directory contains four runnable scripts:

- `compilation.py`: measures Numba compilation overhead.
- `experiment.py`: runs the algorithm comparisons and light-curve benchmarks.
- `analyze.py`: checks the captured geometry against exact rational arithmetic.
- `report.py`: turns the measurement files into a Markdown report.

## Execution order

For a fresh report, run these steps in order and wait for each to finish:

1. **`compilation.py`** creates `compilation.json`. This measurement is independent
   of the main experiment and keeps compilation overhead separate from warm timings.
2. **`experiment.py --repeats 5`** creates `results.json`, `eclipse_geometry.npz`,
   flux arrays, and profiles. This is the main measurement step.
3. **`analyze.py`** reads the geometry from step 2 and creates `exact_audit.json`.
4. **`report.py`** reads the results from steps 1–3 and writes
   `PYPEX_CLIPPING_EXPERIMENT.md` in the repository root.

`report.py` does not run measurements or regenerate missing inputs. If the raw
results have been deleted, repeat all four steps. If you only edit the report
template, rerun just `report.py` while its input files are still available.

## Run the complete experiment

Run these commands from the repository root using the existing ELISA environment:

```bash
cd /home/mike/Work/projects/elisa
source /home/mike/Work/envs/elisa312/bin/activate
export PYTHONPATH=src:.

# 1. Measure compilation overhead.
python scripts/benchmark/clipping/compilation.py

# 2. Run the main experiment and capture geometry.
python scripts/benchmark/clipping/experiment.py --repeats 5

# 3. Analyze the geometry captured in step 2.
python scripts/benchmark/clipping/analyze.py

# 4. Generate the final report from all three result sets.
python scripts/benchmark/clipping/report.py
```

If you use a different existing Python environment, activate that environment
instead. It needs the project dependencies, including Numba; the
scripts do not install anything. ELISA must be configured with accessible
limb-darkening tables (`LD_TABLES`). The scripts use the usual ELISA configuration,
or the configuration selected through `ELISA_CONFIG`.

## What each file does

| File                | Purpose                                                                                                                                                                                            | Inputs                                                                              | Outputs                                                                    |
|---------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| `compilation.py`    | Measures the first Numba compilation and subsequent calls separately, using a fresh cache directory under `/tmp`.                                                                                  | Geometry kernel; no complete binary model.                                          | `compilation.json`                                                         |
| `experiment.py`     | Compares Pypex, the Python reference, and compiled Sutherland-Hodgman. Benchmarks intersection areas, partial coverage, and complete light curves. Captures geometry and runs cProfile separately. | ELISA configuration and data; systems are defined in `system()`.                    | `results.json`, `flux_*.npz`, `profile_*.txt`, and `eclipse_geometry.npz`. |
| `analyze.py`        | Compares captured areas against exact rational geometry and investigates Pypex coordinate rounding.                                                                                                | `eclipse_geometry.npz` from the experiment.                                         | `exact_audit.json`                                                         |
| `report.py`         | Fills the report template with tables of measured results.                                                                                                                                         | `results.json`, `exact_audit.json`, `compilation.json`, and `eclipse_geometry.npz`. | `PYPEX_CLIPPING_EXPERIMENT.md` in the repository root.                     |
| `.gitignore`        | Excludes raw results, profiles, logs, and caches from commits.                                                                                                                                     | -                                                                                   | -                                                                          |

`experiment.py` runs the mandatory baseline first, followed by a variant with
80° inclination and a variant with a coarser mesh. It verifies that every system
is circular, synchronous, and detached. It uses one worker, warms up the
calculations, and then collects five timing repetitions by default. Complete
light-curve timing runs are separate from profiling.

The `numba` label in JSON and NPZ results is the historical name of the compiled
backend. Its configuration value is `sutherland_hodgman`; the script handles this
mapping. It explicitly selects Pypex for comparison without modifying the user's
configuration file.

## Files overwritten by a new run

A new experiment overwrites the raw results and `eclipse_geometry.npz` in this
directory. The geometry archive is generated benchmark output, not a versioned
input. `report.py` also overwrites `PYPEX_CLIPPING_EXPERIMENT.md` in the repository
root.

Use the geometry archive and results from the same experiment. Whenever you rerun
`experiment.py`, rerun `analyze.py` before generating the report.

The generator uses a template: numerical tables come from the result files, but
the date and some commentary are hardcoded in `report.py`. Update those sections
to match the actual run when preparing a report from new measurements.

## What to commit

Keep the scripts, this README, and `.gitignore`. Raw JSON/NPZ results, profiles,
and logs in this directory are generated artifacts and are ignored by Git. Logs
are created only if you redirect command output to files.

The final Markdown report preserves the findings in a readable form; review its
changes after a new experiment.
