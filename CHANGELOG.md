# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Historical release identifiers are preserved as originally published.
Planned work is tracked in the [roadmap](ROADMAP.md).

## 1.0.0

### Scientific modeling and fitting

#### Added

- Cartesian pulsation regression checks for pole continuity, signed amplitudes,
  radius-weighted normalization, frozen-equilibrium time derivatives, physical
  vector units, colormap projections and single/binary observable synthesis.
- Aggregated MCMC synthesis-failure diagnostics across all workers, with concise
  periodic INFO summaries instead of individual candidate warnings and tracebacks.
  `MCMC_DIAGNOSTICS_INTERVAL_STEPS` /
  `[computational].mcmc_diagnostics_interval_steps` sets a positive reporting interval
  (default `100` steps). With `save=True`, a separate atomic `.diagnostics.json`
  file records initialization, burn-in, and production statistics with interval
  history. Diagnostics are best-effort and do not alter sampling or chain saving.
- Physical validation of initial LC MCMC walkers before sampling: each candidate
  initializes the fitted binary, checks its morphology, and synthesizes one
  unnormalized LC point at periastron (a deterministic reference phase for circular
  orbits), requiring finite, positive fluxes in every fitted passband. Valid `x0`
  values are retained; invalid automatic candidates are replaced by uniform draws.
  `MCMC_INITIAL_WALKER_MAX_ATTEMPTS` / `[computational].mcmc_initial_walker_max_attempts`
  sets a positive per-walker attempt limit (default `100`, including the original
  candidate), with rejection diagnostics on exhaustion. Supplied ensembles are
  validated without changing their coordinates. RV initialization and MCMC sampling
  behavior remain unchanged.
- Configurable Kepler solving through `MEAN_ANOMALY_SOLVER` or
  `[computational].mean_anomaly_solver`: `vectorized` (default) solves arrays of
  orbital phases together for supported float64 arrays, with scalar fallback
  when batch solving fails. The `scalar` option uses the original SciPy solver
  for each phase separately.
- MCMC chain exports can now retain sampling-step and walker dimensions, allowing
  reloads to preserve the time ordering needed by trace and autocorrelation plots.
  Discard controls distinguish complete steps from individual sample rows through
  `DiscardUnit.STEPS` / `ROWS` in `elisa.conf.enums`. The same module provides
  `SamplingMethod.UNIFORM` / `ADAPTIVE` for LC fitting phase sampling; existing
  string arguments and manual phase arrays remain supported.
- Configurable reliability checks for MCMC integrated autocorrelation-time
  estimates through `MCMC_AUTOCORR_TOL`: a positive value requires the retained
  chain to span at least that many estimated autocorrelation times, otherwise
  the estimate is reported as unavailable. The default `0` disables this
  chain-length check; it does not establish that a chain has converged.
- Shared `ElisaJSONEncoder` lets parameter, fitting-result, and MCMC-chain JSON
  exports serialize NumPy integer, floating-point, and boolean scalars, arrays
  of JSON-compatible values, Astropy unit objects, and ISO 8601 dates/datetimes
  consistently. Astropy quantities remain unsupported and raise `TypeError`.
- Explicit cache controls: `ld.clear_ld_caches()` clears loaded limb-darkening
  tables, interpolators, and coefficient results; atmosphere
  `clear_atm_strip_cache()` clears reused bandwidth-trimming results, while
  `atm_strip_cache_info()` reports cache usage and capacity.
- Optional `PENALIZE_ALL_EXCEPTIONS` lets fitting continue by rejecting candidates
  whenever their evaluation raises an ordinary exception. By default, only
  recognized physical and numerical failures reject candidates; unexpected errors
  propagate. The option can hide programming errors and does not cover fit setup
  or process-control exceptions.
- Explicit-input tracking (`elisa.base.parameter_tracking`) for stars, spots, pulsation
  modes, and binary/single systems: constructor arguments and later user assignments are
  recorded as explicit, while omitted parameters stay automatic. `init()` now preserves
  explicit values and recalculates automatic ones (e.g. albedo, discretization factor,
  gamma) after related inputs change. Spots and pulsation modes gained `init()`, and
  `Star.init()` rebuilds nested spots and modes. See
  `src/elisa/base/README_PARAMETER_TRACKING.md`.

#### Changed

- Clarified pulsation mesh-to-metre conversions and the velocity-coefficient
  meaning of mode amplitudes. Updated the pulsation tutorial to distinguish
  coefficient amplitudes from physical RMS velocities and document preserved
  elapsed cycles in the Observer's `times=` interface.
- Replaced finite spherical-coordinate pulsation increments with physical Cartesian
  displacement vectors evaluated on the flattened equilibrium mesh, including spot
  replacement vertices. Position, intrinsic velocity and acceleration now share
  the same complex displacement and component-frame rotation; displaced face
  centers, normals and areas are rebuilt before surface integration.
- Breaking pulsation API change: removed `spherical_perturbation` and the legacy
  angular-displacement and pole-treatment helpers. Perturbation functions return
  Cartesian physical vectors; spherical diagnostic components require explicit
  projection with `spherical_vector_components`.
- Optimized tabulated-atmosphere radiances by integrating unique aligned,
  passbanded spectra before mapping and temperature interpolation across surface
  faces. Integrated fluxes and T^4 interpolation weights use float64 precision,
  reducing temporary atmosphere allocations by about 88% and complete light-curve
  synthesis time by 4-11% in representative benchmarks.
- Optimized binary surface coverage outside eclipses by skipping geometric
  projections and partial-eclipse preparation, reducing allocations and light-curve
  synthesis time while preserving identical coverage and flux values.
- Added default Sutherland-Hodgman eclipse clipping without legacy coordinate
  rounding; `INTERSECTION_BACKEND` also supports `pypex` and `reference`.
- Reduced orbital calculation overhead with vectorized Kepler solving and cached
  collinear Lagrange points.
- Reused limb-darkening interpolators and atmospheric bandwidth strips, deduplicated
  atmosphere table lookups, and avoided redundant bolometric passband preparation.
- Reduced repeated LC fitting validation, observation mapping, and R² preparation,
  and skipped unused reflection calculations.
- Moved atmosphere APIs from `elisa.atm` to `elisa.atm.atm`.
- Replaced legacy exception names and many generic validation errors with explicit
  domain exceptions in `elisa.base.error`; callers must update imports and handlers.
- Custom least-squares Jacobians and `jac_sparsity` must now have one row per
  observed data point, in dataset insertion order; robust losses assess each
  point separately, with `f_scale` measured in units of residual divided by
  measurement uncertainty.
- When loading MCMC chains with saved step/walker metadata, the discard count
  now removes complete sampling steps, including all walkers: discarding 100
  steps with 20 walkers removes 2,000 samples. Older flat files still discard
  individual sample rows; use `discard_unit` to select steps or rows explicitly.
- Made settings updates atomic and validated, and preserved host logging handlers
  when applying built-in logging profiles.
- `reinitialization_kwargs()` on stars, systems, spots, and pulsation modes now returns only
  explicit inputs; `PulsationMode.reinitialization_kwargs()` serialization moved to
  `PulsationMode.kwargs_serializer()`.

#### Fixed

- Fixed the `radius` colormap in binary orbital animations, which incorrectly
  included the offset to the system barycenter in each star's radius. Radii are
  now measured from the corresponding star's center in both pulsating and
  non-pulsating systems, while the rendered surface geometry remains unchanged.
- Corrected `temperature` colormaps with `subtract_equilibrium=True` in single-
  and binary-star surface plots and animations. They now show the temperature
  change caused by pulsations relative to the same surface before pulsations,
  including spots and binary irradiation. Previously, the color values were
  calculated from already perturbed temperatures, giving incorrect differences.
- Fixed non-finite surface temperatures at zero pulsation velocity amplitude,
  which could cause light-curve and radial-velocity synthesis to fail. An
  explicitly configured temperature perturbation still produces thermal
  oscillations at zero displacement; automatically calculated temperature
  perturbations become zero.
- Evaluate pulsation harmonics and Cartesian tangent gradients with a regular
  solid-harmonic recurrence that remains defined at both poles. Horizontal
  displacement uses local equilibrium radii and complex vertex RMS normalization.
- Corrected pulsation colormaps to project physical Cartesian vectors at
  equilibrium vertices or face centers, avoid duplicate velocity/acceleration
  scaling, and leave the surface mesh unchanged when displaying displacement.
- Reset flatten state before complete single-star container rebuilds so spot
  replacement vertices and faces are merged on every build, including pulsating
  spotted stars.
- Corrected single-star rotational velocities for stellar and spot surfaces to
  use `omega × position`, matching the time derivative of the rotating geometry
  and the established observer-frame RV convention. Added independent geometry,
  analytic rotation, spot and RV regression checks across phases and inclinations,
  and refreshed the affected pulsating single-star RV reference curve.
- Preserved orbital cycle counts in absolute-time inputs to `Observer.lc()`
  and `Observer.rv()`, making pulsating and asynchronously spotted systems
  consistent with equivalent unwrapped-phase observations. Time-derived
  returned phases now retain elapsed cycles; strictly periodic systems keep
  their existing phase reduction and deduplication.
- Corrected orbital light-travel-time delays for binary pulsations: evaluate
  harmonics at emission time by subtracting the observer-frame barycentric
  propagation delay, without applying the inclination projection twice.
  Added geometric and synthesis regression tests, refreshed the pulsating-binary
  RV reference, and corrected the line-of-sight vector in its utility docstring.
- Removed whole-kelvin rounding from blackbody surface temperatures so
  least-squares finite differences retain the Planck-flux temperature
  response and recover noiseless light-curve parameters after the passband
  limb-darkening correction. Preserved the solver and recovery tolerances.
- Updated the obsolete single-star V-band pulsation reference for
  passband-specific limb-darkening normalization, with a reproducible
  generator that checks the historical bolometric-normalization shape.
- Corrected photometric surface-flux normalization to use each passband's
  limb-darkening coefficients for both binary and single stars, preserving
  hemispheric passband flux conservation. Bolometric limb darkening remains
  in use for irradiation and reflection. Added independent conservation
  checks for all supported limb-darkening laws.
- Fixed single-phase eccentric LC synthesis to use exact integration without
  attempting curve approximations that require multiple orbital phases.
- Fixed phase-dependent Roche geometry in binary `plot.mesh()`, `plot.wireframe()`,
  and `build_container(phase=... / time=...)`: eccentric snapshots now apply
  volume-corrected instantaneous surface potentials before mesh construction,
  preserving stored periastron reference potentials and matching `plot.surface()`.
  Mesh plots and public containers also update spot longitudes for the requested
  phase; mesh and wireframe plotting remain independent of atmosphere and
  limb-darkening tables.
- Fixed rebuild-mask indexing in the eccentric similar-neighbour synthesis
  approximation: geometry and irradiation decisions now follow original orbital
  position indices after separation sorting and multiprocessing partitioning,
  preventing stale surface physics in LC and radiometric RV synthesis.
- Corrected eccentric surface-potential reference geometry and phase-dependent
  volume conservation: input `surface_potential` now refers to periastron rather
  than unit separation; reusing old numerical potentials can change radii and
  light curves, so historical fitted potentials may need reinterpretation or conversion.
- Fixed near-critical Roche radius root selection and convergence, and generalized
  critical potentials for eccentric and asynchronous binaries.
- Corrected secondary-star polar gravity at non-unit component separations.
- Fixed star and system reinitialization to preserve current parameters, spot and
  pulsation units, and `polar_log_g` without repeated 2 dex shifts.
- Fixed single-star rotation-period unit conversion and period initialization.
- Fixed Kepler nonconvergence and nonfinite-result handling, incomplete Lagrange
  solutions, and unintended changes to NumPy floating-point error settings.
- Corrected LC/RV prediction alignment and LC interpolation and normalization at
  observation coordinates, including repeated, unsorted, and differently sampled data.
- Fixed LC/RV least squares to minimize per-observation standardized residuals
  instead of squaring the total chi-squared, including finite penalties for invalid candidates.
- Made LSQ/MCMC candidate rejection consistent for recognized synthesis failures,
  while propagating unexpected errors by default.
- Fixed MCMC likelihood numerical edge cases, prior validation, and initialization
  of walkers at finite-posterior states within prior bounds.
- Corrected constrained MCMC summaries using joint posterior samples, preserving
  correlations, full precision, and constant-sample credible intervals.
- Fixed MCMC chain validation, reloads, filtering, walker traces, and autocorrelation
  diagnostics to preserve or explicitly require trustworthy ensemble time axes.
- Fixed standard/community fitting parameter validation, model reconstruction from
  fit results, and result round trips preserving constraints, units, and R².
- Corrected fitted pulsation frequency conversion and validation of pulsation modes,
  spot construction, nonfinite physical inputs, and LC normalization fractions.
- Applied custom atmosphere and limb-darkening settings consistently in MCMC fitting,
  and rejected custom coefficients reused under an incompatible limb-darkening law.
- Fixed stale atmosphere, limb-darkening, RV passband, and calibration caches after
  configuration or source changes, and protected cached LD coefficients from caller mutation.
- Preserved small atmospheric fluxes at interpolated bandwidth boundaries and exact
  limb-darkening query coordinates instead of rounding them.
- Improved errors for missing or malformed atmosphere, limb-darkening, and passband
  sources.
- Fixed Pypex 2D point-side checks for NumPy 2.5, parallel-line classification,
  and orientation-independent collinear segment overlap detection.

#### Removed

- Deprecated `[support].atlas` INI option and the first-run atmosphere-selection
  prompt; select atmospheres per star or through fitting's `atmosphere_model` mapping.
- Global `settings.ATM_ATLAS`. `Star.atmosphere` (default `ck04`) is the only atmosphere selection;
  canonical identifiers are `ck04`, `k93` and `bb`.

### GUI

#### Added

- Gradio desktop UI, launched with `python -m elisa.ui`, with tabs for light curve
  modeling, radial velocity modeling, light curve fitting, radial velocity
  fitting, and system visualization.
- Least-squares and MCMC fitting in both fitting tabs, kinematic and radiometric
  RV modeling, and surface, mesh, orbit, and equipotential visualization.
- UI configuration through `ELISA_UI_SERVER_PORT`, `ELISA_UI_SERVER_HOST`,
  `ELISA_UI_THEME`, `ELISA_UI_MAX_SPOTS`, and `ELISA_UI_MAX_PULSE_MODES`.
- `UI_GUIDE.md` covering parameters, workflows, and troubleshooting.

#### Fixed

- Improved GUI handling of validation errors and numerical synthesis failures.

### Python compatibility and packaging

Includes changes previously listed under the unreleased `0.7.0dev0` version.

#### Added

- Support for Python 3.13 and 3.14.
- `pyproject.toml` with PEP 517 build isolation using the setuptools backend.
- Development extras for building and publishing packages.

#### Changed

- Raised the minimum supported Python version to 3.11.
- Modernized packaging configuration and dependency declarations.
- Replaced deprecated NumPy APIs, including `np.row_stack`, `np.chararray`, and
  `np.NaN`, and improved NumPy 2.x compatibility in geometry and intersection utilities.
- Adopted the Keep a Changelog format.

#### Fixed

- Included required data files in source distributions.

## 0.6.2 - 2026-03-01

### Fixed

- Fixed package builds to include all required files from the data directory.

## 0.6.1 - 2026-02-04

### Fixed

- Fixed INI settings parsed with the wrong getter so numeric and boolean values
  use their documented types.
- Parsed `rv_lambda_interval` tuples safely while preserving integer and
  floating-point values.
- Fixed settings-loading failures for otherwise valid configurations.

## 0.6.0 - 2026-02-04

### Added

- A first-run wizard to configure the settings required to start ELISA.
- A download manager for installing limb-darkening tables and atmosphere models
  without downloading them manually.
- Support for Python 3.9 through 3.12.
- Gaia passbands using the Van Hamme 2019 limb-darkening table notation:
  `Gaia.2010.G` (equivalent to the previous `GaiaDR2`), `Gaia.2010.BP`, and `Gaia.2010.RP`.
- A `default_discretization_factor` setting to control the resolution of surface meshes.
- Automatic semi-major axis estimation when loading light curve fitting parameters
  without `semi_major_axis`. `LCBinaryAnalyticsTask.set_result()` and `load_result()`
  accept `autofill_sma=True` to select a fixed value that keeps component surface
  gravities within the range supported by atmosphere models.
- Custom atmosphere models and per-passband limb-darkening coefficients through
  the `Star` arguments `atmosphere` and `limb_darkening_coefficients`, and through
  `AnalyticsTask` arguments for fitting. Custom coefficients remain constant over
  the component surface.
- Configurable observer distance through the `distance` parameter of
  `BinarySystem` and `SingleSystem`, with a default of 10 pc.
- Light curves in magnitudes through `Observer.flux_unit = u.mag` or the
  `flux_unit` argument of `Observer.observe.lc()`.
- A `MAGNITUDE_SYSTEM` setting to select `vega` (default), `ab`, or `st`
  photometric zero points for magnitude calculations.

### Changed

- Updated dependencies and code for compatibility with Python 3.6 through 3.12.

### Fixed

- Prevented configuration parser crashes when `general.home` is set.
- Restricted normal-distribution priors to the configured `min` and `max` bounds.

## 0.5.1 - 2021-11-04

### Fixed

- Corrected dependency requirements to avoid the incompatibility between
  `packaging 21.2`, which requires `pyparsing<3,>=2.0.2`, and `pyparsing 3.0.4`.

## 0.5 - 2021-10-20

### Added

- Black-body atmosphere modeling with the `black_body` option.
- Per-object atmosphere selection, allowing different stars to use different models.
- `velocity` and `radial_velocity` colormaps for `BinarySystem.plot.surface()`
  and `SingleSystem.plot.surface()`.
- A choice of uniform or normal parameter priors, with `sigma` setting the
  standard deviation of a normal prior.
- Built-in `default` and `fit` logging configurations, or a custom configuration
  file path, selected through `LOG_CONFIG`. The `fit` configuration limits logging
  to analytics classes.
- Adaptive phase sampling and user-supplied phase samples for fitting through
  the `samples` argument.
- An option to return the surface plot figure for further customization through
  `return_figure_instance`.
- Filtering of flattened MCMC samples with `AnalyticsTask.filter_chain()` to
  examine different fitted solutions.
- Fit-quality evaluation using R² through
  `AnalyticsTask.coefficient_of_determination()`.
- Loading radius-based JSON models with `equivalent_radius` as an alternative to
  the standard `polar_log_g` format.
- `BinarySystem.build_container()` and `SingleSystem.build_container()` for complete
  system models at a photometric `phase` or observational `time`.

### Changed

- Preserved the solar constant when changing surface mesh resolution.
- Improved trapezoidal discretization.
- Checked relative changes in irradiation before reusing calculations at similar
  orbital positions, improving eccentric light curve integration accuracy.
- Used Numba for computationally intensive calculations, including reflection effects.
- Changed `elisa.analytics.tasks.load_results()` to return a dictionary.
- Removed the need to call `set_up_logging()` when changing logging configurations.
- Calibrated corrections for underestimated mesh surface areas separately for
  single stars, detached binaries, and over-contact binaries.
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

- Radiometric radial velocity curves that account for the Rossiter effect and
  velocity shifts caused by spots.

### Changed

- Updated dependencies and support for Python 3.6, 3.7, and 3.8.
- Made configuration available through a shared settings instance imported with
  `from elisa import settings`.
- Added observation plotting through `DataSet.plot.display_observation()`.

### Fixed

- Removed faulty curve points produced by multiprocessing integration.
- Fixed component volume conservation in eccentric systems with spots.
- Reduced surface area artifacts caused by numerical noise during total eclipses.
- Corrected temperature renormalization to use the fourth power of temperature.

## 0.3.1 - 2020-08-19

### Changed

- Enabled full error propagation in `fit_summary` (`result_summary`) through
  the `propagate_errors` argument.

### Fixed

- Fixed on-demand light curve normalization.
- Prevented MCMC chain evaluation crashes near Roche-lobe filling by snapping
  surface potentials to critical values when they fall within the fitted errors.
- Corrected corner and trace plot ranges to use fitted confidence intervals
  instead of parameter search bounds.
- Improved the least squares cost function.
- Corrected the secondary potential derivative.
- Accounted for libration when positioning spots in eccentric orbits.
- Conserved volumes in eccentric systems with spots.

## 0.3 - 2020-06-17

### Added

- Single-star light curves with spots and pulsations.
- A revised analytics API with fitting summaries and improved loading and saving
  of fitting data.
- Support for the TESS passband, including limb-darkening tables.

### Fixed

- Fixed crashes reporting a missing `param` when backend constraints were
  configured incorrectly for over-contact binary fitting.
- Fixed fitting normalization to scale each light curve by its own maximum
  rather than the maximum across all curves.
- Returned a large negative MCMC penalty for invalid binary systems instead of
  a value close to zero.
- Raised `elisa.base.error.AtmosphereError` rather than `FileNotFoundError` for
  missing atmosphere files.
- Allowed light curve observations to use the selected atmosphere instead of
  always using `ck04`.
- Prevented crashes caused by small spots.
- Prevented MCMC chain evaluation crashes near Roche-lobe filling by snapping
  surface potentials to critical values when they fall within the fitted errors.

## 0.2.3 - 2020-05-27

### Fixed

- Fixed crashes reporting a missing `param` when backend constraints were
  configured incorrectly for over-contact binary fitting.
- Fixed fitting normalization to scale each light curve by its own maximum
  rather than the maximum across all curves.
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

- Binary radial velocity curves from center-of-mass motion, parameterized by
  mass ratio `q` and projected semi-major axis `asini`.
- Parallel light curve calculation by distributing batches of orbital phases
  across the configured number of processes, limited by available cores.
- MCMC and nonlinear least squares fitting of binary light curves using standard
  masses `M_1`, `M_2` or community parameters `q` (mass ratio) and `a` (semi-major axis).
- MCMC and nonlinear least squares fitting of radial velocities using the
  parameters `M_1`, `M_2`, `e`, `i`, `omega`, and `gamma`.
- More specific exception types in `elisa.base.errors`.

### Fixed

- Accepted NumPy arrays for `phases` in `Observer.observe.lc()` and `Observer.observe.rv()`.
- Fixed adaptive binary meshing to respect discretization limits, preventing
  triangulation failures caused by too few surface points.
- Prevented `phase_interval_reduce` crashes when `BinarySystem` was not imported directly.
- Removed a constant PI multiplier from the output flux; further investigation
  was still required.
- Prevented crashes when using the `bolometric` passband.
- Accepted NumPy integer and floating-point scalar types during binary system initialization.

## 0.1 - 2019-11-06

### Added

- Binary star surface meshes generated from generalized surface potentials
  and triangulated into surface elements.
- Gravity and temperature distributions over component surfaces.
- Surface spots and temperature pulsations.
- Light curves for circular synchronous and asynchronous binary orbits, with
  or without spots.
- Radial velocity curves based on center-of-mass motion.
- Binary visualization through surface point plots, wire meshes, surface maps of
  physical quantities, orbit plots, and orbital animations.
