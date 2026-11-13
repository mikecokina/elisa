# Roadmap

Updated: 2026-11-13

This roadmap tracks remaining work and deliberately deferred technical debt, not release commitments. Completed audits, tutorials, diagrams and refactors are omitted.

## Next

- [ ] **Standalone binary LC visual diagnostics** - Create one independent, production-quality Python script for each of the eight documented binary LC configurations in `docs/diagrams/`, one system at a time. Each script defines its own fixed, physically valid system, calculates and displays the light curve, saves `lightcurve.png`, and generates `temperature_animation.gif` showing absolute surface temperature with a fixed 45° azimuth / 45° elevation camera and consistent frame-to-frame colour scale. Keep outputs separate. **No CLI, shared base, common utilities, configuration modules or imports between examples; deliberate code duplication is required.** Preserve elapsed orbital cycles where spots or pulsations require them. Start with circular synchronous unspotted.

## Open / deferred

- [ ] **Logging lifecycle and spawned workers (follow-up to PR #29)** - Separate loading/validating `LOG_CONFIG` from applying logging during import. Under spawn, a custom logging JSON selected in the INI can be applied before worker initialization; a `FileHandler(mode="w")` may truncate the parent log. Defer the broader lifecycle refactor; it is not a merge requirement for PR #29. Preserve fork-inherited handlers and explicit parent logging configuration. Cover INI-selected custom logging in spawn regressions when revisited.
- [ ] **GUI: targeted exception-boundary follow-ups** - Check the broad catch in the LC fitting ephemeris fallback (`_extract_from_lsqr()` / `task.get_result()`), which may turn an unexpected programming failure into `None`. Consider a focused malformed/non-object JSON regression for the shared modeling-result loader. These are narrow follow-ups, not a reason to repeat the completed GUI audit. Do not introduce blanket exception handling.
- [ ] **GUI browser E2E Python compatibility** - Remove or justify the hardcoded Python 3.14 fixture requirement. ELISA supports Python 3.11+. Verify E2E behavior on available supported interpreters; do not claim compatibility without testing it.

## Explicitly deferred / do not reopen without evidence

- **Limb-darkening metallicity interpolation** - Intentionally deferred. Define and validate a physically meaningful metallicity interpolation before changing the existing table-selection/bracketing behavior; no speculative change now.
- **Post-fit critical-potential boundary adjustments and classical Lagrange solver** - No further boundary-shifting changes or broad solver/exception-hierarchy cleanup are planned without a concrete correctness problem.