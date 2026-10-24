# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Historical release identifiers are preserved as originally published.
Planned work is tracked in the [roadmap](ROADMAP.md).

## Unreleased

### Added

- Public `elisa.analytics.DiscardUnit` enum with `STEPS` and `ROWS` for MCMC chain
  loading; existing string values remain supported.
- Optional MCMC export shape and ordering metadata to restore ensemble step,
  walker, and parameter axes while preserving the existing flat sample format.
- Public `elisa.analytics.SamplingMethod` enum with `UNIFORM` and `ADAPTIVE`
  strategies shared by LC least-squares and MCMC fitting. Existing string values
  and manual phase lists/arrays remain supported.
- A reproducible LC prediction overhead benchmark separating validation,
  mapping/interpolation, and normalization from physical synthesis.
- `ld.clear_ld_caches()` to clear loaded LD tables, prepared interpolators, and
  evaluated coefficient results together.
- Independent Sutherland-Hodgman polygon/triangle intersection area kernels: a
  float64 reference and a Numba implementation without fastmath, with analytic,
  exact-rational, transformation, and captured eclipse geometry validation.
- Gradio desktop UI, launched with `python -m elisa.ui`, with tabs for light curve
  modeling, radial velocity modeling, light curve fitting, radial velocity fitting,
  and system visualization.
- Least Squares and MCMC fitting methods in the light curve and radial velocity UI tabs.
- Surface, mesh, orbit, and equipotential rendering modes in System Visualization.
- Kinematic and radiometric computation methods in Radial Velocity Modeling.
- UI configuration through `ELISA_UI_SERVER_PORT`, `ELISA_UI_SERVER_HOST`,
  `ELISA_UI_THEME`, `ELISA_UI_MAX_SPOTS`, and `ELISA_UI_MAX_PULSE_MODES`.
- [UI guide](UI_GUIDE.md) covering tabs, parameters, workflows, and troubleshooting.
- Type annotations throughout the codebase for improved type checking and IDE support.

### Changed

- Retain structured MCMC chains on LC/RV fit wrappers. Chain loading defaults to
  discarding ensemble steps for structured exports and flat rows for legacy files;
  `discard_unit` can select an explicit unit. Share production-chain retrieval
  and validation between LC and RV wrappers.
- Validate static light-curve observation mappings once per fitter setup and reuse
  them for LS, MCMC, and internal R² calculations. Revalidate on normalization or
  explicit `validate_observations()` after manual edits; standalone prediction
  and public R² helpers continue validating their inputs.
- Reuse the prepared LC fitter's prediction method and Observer for internal R²
  calculations, retaining compatibility of the standalone `lc_r_squared()` API.
- Require one Jacobian or `jac_sparsity` row per observation in LC/RV least squares,
  ordered by dataset insertion order. Robust-loss `f_scale` now applies to
  individual standardized residuals; finite-tolerance optimizer behavior can change.
- Share light-curve synthesis, observation mapping, and mean normalization across
  LS, MCMC, and R² calculations while preserving physical synthesis, parameter
  constraints, and statistical noise models.
- Keep settings mutable with atomic, validated updates and consistent attribute
  access and serialization.
- Use module-based loggers and configurable logging profiles, with scoped changes
  and host-handler preservation for built-in profiles.
- Use explicit fork contexts for observation and fitting workers without changing
  the application's multiprocessing default.
- Skip unused symmetric reflection blocks and a redundant limb-darkening matrix
  copy while preserving numerical results.
- Reduce passband preparation overhead by removing a redundant deep copy and
  reusing exact numeric bandwidth-trimming results in a bounded LRU cache.
  Store 128-bit BLAKE2b content digests instead of complete spectra in cache keys
  to reduce retained memory, preserving exact boundary bits and cache limits.
  Preserve independent writable outputs, input isolation, Akima interpolation,
  and bitwise-identical spectra, radiances, and light curves.
- Deduplicate selected CK04 and K93 atmosphere grid combinations before
  constructing filenames and paths, preserving table selection, ordering,
  interpolation results, and error behavior.
- Reuse bounded, cached `LinearNDInterpolator` instances for unchanged
  limb-darkening sources while retaining the evaluated-result LRU and evaluating
  current surface parameters on eccentric-orbit rebuilds.
- Evaluate limb-darkening queries at their original float64 coordinates instead
  of rounding to nine decimal places. Cache keys preserve exact values, array
  shape, and ordering
- Use Sutherland-Hodgman clipping by default for eclipse intersection areas.
  Configure it through `INTERSECTION_BACKEND` or `[computational].intersection_backend`;
  `pypex` and `reference` remain available.
- Eliminate intersection coordinate rounding. Measured maximum area error decreased
  from **3.36e-12 to 1.87e-18 SMA²** against exact geometry. Reference and compiled
  results agreed bitwise.
- Measured normalized light-curve differences versus Pypex were at most
  **4.27e-11 (0.0000427 ppm)**, reflecting removal of legacy geometric rounding.
  The physical model and mesh accuracy are unchanged.
- Measured speedups: **236x** for intersection areas, **124x** for partial coverage,
  and approximately **2x** for complete light curves.
- Reduced fitting overhead in MCMC likelihood evaluation, parameter serialization,
  curve integration, and logging-dependent R² calculations.
- Cached albedo and correction factor calculations to reduce repeated I/O.
- Vectorized surface transformations and spot temperature scaling.
- Reduced redundant array allocation while constructing surface meshes.
- Used partitioning where full sorting is unnecessary in observer calculations.

### Fixed

- Compute MCMC autocorrelation functions and integrated times along ensemble
  steps, averaging individual walker ACFs instead of treating flattened walkers
  as consecutive times. Mark unreliable integrated times and parameters with
  constant walkers as unavailable.
- Preserve temporal ordering through structured MCMC reloads and aligned row
  cuts. Clear temporal chains after value-dependent filtering, partial-ensemble
  cuts, or legacy reloads; reject temporal diagnostics without trustworthy axes.
- Validate MCMC sample matrices, labels, temporal metadata and discard arguments
  before slicing or overwriting exported samples. Document discard units and
  export shapes across the chain-loading and serialization APIs.
- Reject explicitly invalid MCMC physical configurations with `-np.inf`,
  independently of previous likelihood evaluations. Propagate programming,
  input-structure, and unexpected numerical errors instead of penalizing them.
- Initialize MCMC walkers only at finite-posterior states within prior bounds,
  with bounded automatic sampling attempts. Validate supplied ensembles without
  clipping or mutating caller arrays.
- Validate free-parameter bounds and Gaussian widths before MCMC sampling;
  normalize widths directly by the physical bound span to avoid cancellation,
  preserving existing prior definitions and fixed/constrained parameter handling.
- Accept finite LC normalization fractions in `(0, 1]` for `maximum`,
  `global_maximum`, and `minimum`; ignore the fraction for mean normalization.
  Prevent empty or out-of-range extrema selections while preserving existing
  normalization formulas and `global_maximum` selection semantics.
- Pass per-observation standardized residual vectors to SciPy for LC and RV least
  squares, so default linear loss minimizes `0.5 * chi2` instead of
  `0.5 * chi2**2`. Apply robust losses to individual observations and use a
  fixed-length, finite chi-squared penalty vector for invalid LC synthesis.
- Apply per-passband observation reducers in direct MCMC synthesis, preserving
  unsorted and repeated coordinates and differently sampled passbands. Prevent
  singleton observations from broadcasting against an entire synthetic curve.
- Apply explicit per-fit atmosphere models and custom limb-darkening coefficients
  to MCMC synthesis, matching LS and R².
- Use prepared uniform, adaptive, and manual phase support in both fitting methods
  and their final R² calculations, including the interpolation-threshold boundary.
  Accept manual support supplied as a NumPy array.
- Determine light-curve interpolation from actual coordinate correspondence rather
  than array shape, fixing equal-length support and observation grids with
  different phases in LS, MCMC, and R².
- Normalize synthetic curves only after mapping or interpolation onto each
  passband's observation points, ensuring consistent LS/MCMC flux scales.
- Validate observation, reducer, uncertainty, support, and prediction alignment
  before evaluating light-curve costs. Propagate structural errors explicitly
  instead of converting them into MCMC invalid-physical-state penalties.
- Reject reuse of custom limb-darkening coefficients under an incompatible law,
  including logarithmic/square-root switches. Preserve linear/cosine compatibility
  and runtime law switching for automatically interpolated coefficients.
- Resolve stellar fitting temperature and metallicity defaults from the current
  `ATM_ATLAS` when creating parameters, preserving existing instances and explicit
  user bounds. Support CK04/K93 aliases and use CK04 fitting bounds for blackbody.
- Prevent stale RV passband, atmosphere, correction-factor, albedo, and photometric
  calibration caches after configuration or source-file changes.
- Preserve atmospheric spectral flux precision during bandwidth boundary
  interpolation by removing decimal rounding that could erase small nonzero
  flux values.
- Prevent stale limb-darkening table, interpolator, and coefficient reuse after
  directory, law, column configuration, or source-file revision changes.
- Prevent caller mutations of returned LD coefficient arrays from corrupting
  cached results.
- Restored Pypex point-side checks on NumPy 2.5 by explicitly computing
  2D cross-product components instead of calling `np.cross` with 2D vectors.
- Prevented multiprocessing worker hangs by explicitly closing and joining pools.
- Guarded zero-point lookups against missing keys.
- Corrected parallel-line distances in vectorized Pypex segment intersections,
  including parallel versus collinear classification near the distance tolerance.
- Made collinear segment overlap detection independent of endpoint orientation,
  preserving the configured endpoint-contact policy.

## 0.7.0dev0 - 2026-03-02

### Added

- Support for Python 3.13 and 3.14.
- `pyproject.toml` with PEP 517 build isolation using the setuptools backend.
- Development extras for building and publishing packages.

### Changed

- Raised the minimum supported Python version to 3.10.
- Modernized packaging configuration and dependency declarations.
- Replaced deprecated NumPy APIs, including `np.row_stack`, `np.chararray`, and
  `np.NaN`, with NumPy 2.x compatible implementations.
- Improved NumPy 2.x compatibility in geometry and intersection utilities.
- Adopted the Keep a Changelog format.

### Removed

- Support for Python versions below 3.10.

### Fixed

- Included required data files in source distributions.

## 0.6.2 - 2026-03-01

### Fixed

- Included all required files from the package data directory in build distributions.

## 0.6.1 - 2026-02-04

### Fixed

- Aligned INI configuration parsing with documented setting types, correcting uses
  of `getfloat`, `getint`, and `getboolean`.
- Preserved integer and float types when safely parsing `rv_lambda_interval` tuples.
- Allowed valid configurations that previously failed during settings loading.

## 0.6.0 - 2026-02-04

### Added

- First-run configuration wizard for the minimum required settings.
- Download manager for limb-darkening tables and atmosphere models.
- Support for Python 3.9 through 3.12.
- Gaia passbands using the Van Hamme 2019 limb-darkening table notation:
  `Gaia.2010.G` (equivalent to the previous `GaiaDR2`), `Gaia.2010.BP`, and `Gaia.2010.RP`.
- Configurable `default_discretization_factor` to control surface mesh fidelity.
- Automatic semi-major axis estimation when loading light curve fitting parameters
  without `semi_major_axis`. `LCBinaryAnalyticsTask.set_result()` and `load_result()`
  accept `autofill_sma=True` to select a fixed value that keeps component surface
  gravities within the range supported by atmosphere models.
- Custom atmosphere models and per-passband limb-darkening coefficients through
  the `Star` arguments `atmosphere` and `limb_darkening_coefficients`, and through
  `AnalyticsTask` arguments for fitting. Custom coefficients remain constant over
  the component surface.
- Observer distance through the `distance` parameter of `BinarySystem` and
  `SingleSystem`, defaulting to 10 pc.
- Light curves in magnitudes through `Observer.flux_unit = u.mag` or the
  `flux_unit` argument of `Observer.observe.lc()`.
- `MAGNITUDE_SYSTEM` configuration with `vega` (default), `ab`, and `st` zero-point sets.

### Changed

- Updated dependencies and implementation compatibility across Python 3.6 through 3.12.

### Fixed

- Prevented configuration parser crashes when `general.home` is set.
- Clipped normal-distribution priors to the configured `min` and `max` bounds.

## 0.5.1 - 2021-11-04

### Fixed

- Corrected dependency requirements to avoid the incompatibility between
  `packaging 21.2`, which requires `pyparsing<3,>=2.0.2`, and `pyparsing 3.0.4`.

## 0.5 - 2021-10-20

### Added

- Black-body atmospheres through `black_body`.
- Different atmosphere models for individual celestial objects.
- `velocity` and `radial_velocity` colormaps for `BinarySystem.plot.surface()`
  and `SingleSystem.plot.surface()`.
- Uniform and normal parameter priors, with normal standard deviation set by `sigma`.
- Built-in `default` and `fit` logging configurations, or a custom configuration
  file path, selected through `LOG_CONFIG`. The `fit` configuration limits logging
  to analytics classes.
- Adaptive and custom fitting samples through the `samples` argument.
- Figure return values through the surface plot argument `return_figure_instance`.
- Flat-chain filtering with `AnalyticsTask.filter_chain()` to examine multiple solutions.
- R² evaluation with `AnalyticsTask.coefficient_of_determination()`.
- Loading radius-based JSON models with `equivalent_radius` as an alternative to
  the standard `polar_log_g` format.
- `BinarySystem.build_container()` and `SingleSystem.build_container()` for complete
  system models at a photometric `phase` or observational `time`.

### Changed

- Conserved the solar constant across surface discretization levels.
- Improved trapezoidal discretization.
- Checked relative irradiation changes before reusing approximations at similar
  orbital positions during eccentric light curve integration.
- Used Numba for computationally intensive calculations such as reflection effects,
  in preparation for GPU support.
- Returned dictionaries from `elisa.analytics.tasks.load_results()`.
- Removed the need to call `set_up_logging()` when changing logging configurations.
- Tuned surface underestimation corrections separately for single stars, detached
  binaries, and over-contact binaries.
- Scaled default component discretization to produce similar flux contributions
  per surface element within the permitted discretization range, reducing excessive
  sampling of smaller, dimmer components.

### Fixed

- Reinitialized binary system parameters correctly, including pulsation parameters.
- Applied inclination rotation in the positive direction.
- Aligned the line-of-sight convention with negative radial velocities for approaching
  bodies: changed the vector from `[1, 0, 0]` to `[-1, 0, 0]`, measured azimuth from
  the y-axis, and placed the observer at `[-inf, 0, 0]`.
- Used flux-based rather than temperature-based weights for atmosphere interpolation.
- Corrected surface element visibility during eclipses by stars smaller than the
  eclipsed surface elements.
- Prevented solver crashes near the neck of over-contact stars by changing the
  cylindrical-coordinate initial value from `polar_radius` to `0.25 * polar_radius`.

## 0.4 - 2020-10-01

### Added

- Radiometric radial velocity curves that model the Rossiter effect and spot effects.

### Changed

- Updated dependencies and support for Python 3.6, 3.7, and 3.8.
- Exposed configuration through a singleton imported with `from elisa import settings`.
- Enabled observation plotting with `DataSet.plot.display_observation()`.

### Fixed

- Removed faulty curve points produced by multiprocessing integration.
- Conserved component volumes in eccentric systems with spots.
- Reduced surface area artifacts caused by numerical noise during total eclipses.
- Corrected temperature renormalization using fourth powers.

## 0.3.1 - 2020-08-19

### Changed

- Enabled full error propagation in `fit_summary` (`result_summary`) through
  the `propagate_errors` argument.

### Fixed

- Normalized light curves on demand.
- Prevented MCMC chain evaluation crashes near Roche-lobe filling by snapping
  surface potentials to critical values when they fall within the fitted errors.
- Used confidence intervals rather than fitting bounds in corner and trace plots.
- Improved the least squares cost function.
- Corrected the secondary potential derivative.
- Accounted for libration when positioning spots in eccentric orbits.
- Conserved volumes in eccentric systems with spots.

## 0.3 - 2020-06-17

### Added

- Single-star light curves with spots and pulsations.
- Improved analytics API, fitting summaries, and fitting input/output.
- TESS passband with limb-darkening tables.

### Fixed

- Prevented missing-`param` crashes caused by invalid backend constraints when
  fitting over-contact binaries.
- Normalized each fitted light curve to its own maximum.
- Returned a large negative MCMC penalty for invalid binary systems instead of
  a value close to zero.
- Raised `elisa.base.error.AtmosphereError` rather than `FileNotFoundError` for
  missing atmosphere files.
- Removed the hardcoded `ck04` atmosphere from light curve observations.
- Prevented crashes caused by small spots.
- Prevented MCMC chain evaluation crashes near Roche-lobe filling by snapping
  surface potentials to critical values when they fall within the fitted errors.

## 0.2.3 - 2020-05-27

### Fixed

- Prevented missing-`param` crashes caused by invalid backend constraints when
  fitting over-contact binaries.
- Normalized each fitted light curve to its own maximum.
- Returned a large negative MCMC penalty for invalid binary systems instead of
  a value close to zero.
- Raised `elisa.base.error.AtmosphereError` rather than `FileNotFoundError` for
  missing atmosphere files.

## 0.2.2 - 2020-01-29

### Fixed

- Corrected radial velocity curve orientation.
- Corrected dependency requirements in `setup.py` and documentation.

## 0.2.1 - 2020-01-17

### Fixed

- Inherited spot discretization from the parent object when not set explicitly.
- Corrected spot detection on the neck of over-contact systems.

## 0.2 - 2019-12-29

### Added

- Binary radial velocity curves based on center-of-mass motion and the community
  parameters `q` and `asini`.
- Parallel light curve calculation by splitting phases into batches across up to
  `N` available processes.
- MCMC and nonlinear least squares fitting of binary light curves using standard
  masses `M_1`, `M_2` or community parameters `q` (mass ratio) and `a` (semi-major axis).
- MCMC and nonlinear least squares fitting of radial velocities using the
  parameters `M_1`, `M_2`, `e`, `i`, `omega`, and `gamma`.
- More specific exception types in `elisa.base.errors`.

### Fixed

- Accepted NumPy arrays for `phases` in `Observer.observe.lc()` and `Observer.observe.rv()`.
- Kept adaptive binary discretization within its prescribed bounds to avoid
  insufficient surface points and triangulation crashes.
- Prevented `phase_interval_reduce` crashes when `BinarySystem` was not imported directly.
- Removed a constant PI multiplier from the output flux; further investigation
  was still required.
- Prevented crashes when using the `bolometric` passband.
- Accepted NumPy integer and floating-point scalar types during binary system initialization.

## 0.1 - 2019-11-06

### Added

- Binary surface point generation from generalized surface potentials and
  triangulation into component surface elements.
- Gravity and temperature distributions over component surfaces.
- Surface spots and temperature pulsations.
- Light curves for circular synchronous and asynchronous binary orbits, with
  or without spots.
- Radial velocity curves based on center-of-mass motion.
- Binary visualization through surface point plots, wire meshes, surface maps of
  physical quantities, orbit plots, and orbital animations.
