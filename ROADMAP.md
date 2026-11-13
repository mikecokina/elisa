# Roadmap

Updated: 2026-11-13

This roadmap tracks remaining work and deliberately deferred technical debt, not release commitments. Completed audits, tutorials, diagrams and refactors are omitted.

## Next

## Open / deferred

- [ ] **Logging lifecycle and spawned workers (follow-up to PR #29)** - Separate loading/validating `LOG_CONFIG` from applying logging during import. Under spawn, a custom logging JSON selected in the INI can be applied before worker initialization; a `FileHandler(mode="w")` may truncate the parent log. Defer the broader lifecycle refactor; it is not a merge requirement for PR #29. Preserve fork-inherited handlers and explicit parent logging configuration. Cover INI-selected custom logging in spawn regressions when revisited.

- [ ] **LC fitting: preserve user-supplied phase coordinates** - When users provide phase-domain observations, preserve their original phases without implicit modulo-one folding in the fitting time layer. Ensure direct synthesis and interpolation support the supplied phase range consistently in both LSQ and MCMC. Preserve existing JD/time-domain folding and candidate period/epoch recomputation. Add focused regression tests. No fitting refactor planned.

## Explicitly deferred / do not reopen without evidence

- **Limb-darkening metallicity interpolation** - Intentionally deferred. Define and validate a physically meaningful metallicity interpolation before changing the existing table-selection/bracketing behavior; no speculative change now.
- **Post-fit critical-potential boundary adjustments and classical Lagrange solver** - No further boundary-shifting changes or broad solver/exception-hierarchy cleanup are planned without a concrete correctness problem.