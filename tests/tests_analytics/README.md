# Analytics test cleanup report

Changes are confined to analytics tests and their local infrastructure. Production
code, numerical tolerances and dependencies are unchanged. Existing analytics
modules outside the fitting work were left alone.

## Module mapping

| Original module                          | Destination                                                                                                            |
|------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| `test_phase1_uncertainties.py`           | `test_observation_preparation.py`, `test_observation_normalization.py`                                                 |
| `test_phase2_mcmc_posterior.py`          | `test_mcmc_posterior.py`                                                                                               |
| `test_phase3a_chain_diagnostics.py`      | `test_chain_io.py`, `test_chain_diagnostics.py`                                                                        |
| `test_phase3b1_constrained_posterior.py` | `test_constrained_posterior.py`                                                                                        |
| `test_fitting_hot_path.py`               | `test_fitting_evaluation.py`, `test_likelihood.py`; scalar constraint arithmetic joins `test_constrained_posterior.py` |

The existing `test_lc_prediction_consistency.py`, `test_lc_preparation_and_lsq.py`
and `test_mcmc_diagnostic_plotting.py` retain their functional scope.

## Readability and infrastructure

- Shared LC/RV data factories and parameter conversions live in
  `fitting_helpers.py`; test modules no longer import other runnable test modules.
- Two duplicate science fixtures and a manual test-case lifecycle are replaced by
  one settings fixture with teardown. A second fixture supplies real LC/RV result
  wrappers where both public APIs are tested.
- The synthesis test class becomes ordinary pytest tests. Independent mapping,
  statistical cost and R² checks are small, named functions.
- Forwarding wrappers replace runtime profiling and inspection of stack locals;
  the actual SciPy solver, synthesis and numerical calculations still execute.
- Real parameter metadata and result containers replace fabricated namespaces and
  serializer lambdas. Sampling and expensive final-score synthesis are mocked
  only at their boundaries.
- Named parameter cases replace tuple slicing and conditional smoothing
  expectations. Posterior quantities have descriptive labels, and separate
  linear, nonlinear and custom-percentile cases have explicit expectations.

## Coverage and verification

Python **3.14.6**, using the installed ELISA 3.14 environment and `PYTHONPATH=src`:

| Check                                                      | Before                               | After                                |
|------------------------------------------------------------|--------------------------------------|--------------------------------------|
| Complete analytics suite                                   | 645 collected; 605 passed, 40 failed | 604 collected; 584 passed, 20 failed |
| Affected tests                                             | 374 collected; 334 passed, 40 failed | 333 collected; 313 passed, 20 failed |
| Existing unittest subtests in complete suite               | 9 passed                             | 9 passed                             |
| Pyright, with repository root and `src` on the import path | 135 errors in 8 original files       | 0 errors in 13 modified Python files |

The collection difference is accounted for: remove 41 accidental duplicate
synthesis cases collected through an imported test class, five duplicate
fixed/free preparation cases for an API absent from the source, and two slot
cases referring to absent prepared attributes. Existing slot behavior remains
covered in observation preparation. Expand scalar constraint arithmetic from
one case to five and the combined posterior-summary test from one case to four.
All 41 distinct synthesis cases remain collected once.

The numerical checks retain uncertainty fallbacks and units, normalization and
coordinate alignment, LSQ residuals and robust loss, Gaussian likelihoods,
priors and physical rejection, current Roche boundaries, joint constrained
posteriors and precision, chain ordering/discard/filter/serialization, temporal
autocorrelation and trace plotting. No failures were skipped or marked xfail.

Ruff was explicitly run on every modified Python file with exclusion overridden.
`E,F,I,UP,B,C901,SIM` pass, including **C901 maximum 8**. A separate audit with the
repository's full production `ALL` selection falls from 1,304 to 406 findings;
it is **not clean**. Remaining findings include pytest assertions (`S101`),
unannotated test parameters, numeric test constants and formatter-conflicting
trailing-comma rules. No lint suppressions or type-ignore comments were added.
`git diff --check` passes, and explicit pytest collection discovers the renamed
modules.

## Production defects deliberately left outside scope

The 20 retained failures express existing scientific expectations previously
failing in `test_fitting_hot_path.py`, sometimes masked by absent preparation APIs:

- **13 likelihood cases:** direct variance squaring and exponentiation underflow
  or overflow for extreme positive errors/noise, a subnormal nonzero residual
  fails its independent log-coordinate reference, and unrepresentable density
  returns a nonfinite value instead of raising a numerical failure.
- **7 synthesis-output cases:** RV missing/column/short output does not produce
  the expected alignment exception; LC and RV NaN/Inf output is not rejected.

The obsolete preparation calls were replaced with current public calculations so
these expectations reach the actual source. Their assertions and original
numerical tolerances remain intact. Fixing these failures requires a separate
production change.
