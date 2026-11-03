# Roadmap

Updated: 2026-11-03

This roadmap collects deferred work and technical debt, not commitments for a specific release. Items are unprioritized and should be revisited when supported by a concrete need, bug report, or performance evidence.

## Open / deferred

- [ ] **Limb-darkening metallicity interpolation** - Audit the treatment of metallicity between atmosphere/LD tables. The earlier implementation combined bracketing tables; confirm the intended physical interpolation and its numerical behavior before changing anything.
- [ ] **Cache keys and settings dependence** - Audit caches whose results can change with ELISA settings. Ensure all result-affecting settings are represented in cache identity, or that invalidation is correct. Avoid gratuitous cache changes without a reproducible failure.
- [ ] **Atmosphere / reflection performance** - Profile realistic systems, especially eccentric, asynchronous and spotted configurations. Previous work provided limited additional acceleration. Preserve configurations where symmetries cannot safely be used.
- [ ] **Notebooks, examples and tutorials** - Review for obsolete APIs, outdated numerical output, and the eccentric-system convention that input `surface_potential` is referenced to periastron (`d = 1 - e`).
- [ ] **GUI: remaining exception-boundary issues** - Review the broad exception catch in the LC fitting ephemeris fallback (`_extract_from_lsqr()` / `task.get_result()`) which can turn an unexpected programming failure into `None`. Also add a focused malformed/non-object JSON regression for the shared modeling-result loader if useful. Do not introduce blanket exception handling.
- [ ] **GUI browser E2E Python compatibility** - Remove or justify the hardcoded Python 3.14 fixture requirement. ELISA's supported Python floor is 3.11. Verify GUI E2E behavior on available supported interpreters; do not claim untested compatibility.
- [ ] **Static type/IDE inspections in tests** - Ruff is clean for the test tree, but an independent mypy/pyright and PyCharm inspection pass was not completed. Address concrete issues as they arise, especially intentional invalid inputs in negative tests. Avoid speculative suppressions.
- [ ] **Ruff outside tests** - A previous audit reported about 1,915 diagnostics outside the test tree, principally in ancillary notebooks/scripts and a few production files. Treat as separate scope; confirm current counts before any cleanup.

## Explicitly deprioritized / do not reopen without a reason

- **MCMC initial walker positions** - No further initialization changes planned. Preserve the existing finite-posterior initialization and candidate-failure handling.
- **Post-fit critical-potential / boundary adjustment** - Further changes to the critical-potential boundary or its shifting/adjustment after fitting are **not planned**. Do not confuse this with the already completed fix for computing physically correct critical potentials in the fit summary.
- **Further classical Lagrange solver cleanup** - No broad refactor merely for naming, exception hierarchy or cosmetic cleanup.
- **Historical near-critical fit-bias experiments** - Earlier volume-reference investigations were sufficient for the present purpose; no additional experiments planned.
