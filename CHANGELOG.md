# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/).


## [Unreleased]

## [0.7.0dev0] - 2026-03-02

### Added
- Support for Python 3.13 and 3.14.
- `pyproject.toml` with PEP 517 build isolation using setuptools backend.
- Structured development extras including build and publishing tools.

### Changed
- Raised minimum supported Python version to 3.10.
- Modernized packaging configuration and dependency declarations.
- Replaced deprecated NumPy APIs (`np.row_stack`, `np.chararray`, `np.NaN`, ...) with NumPy 2.x compatible implementations.
- Improved compatibility with NumPy 2.x across geometry and intersection utilities.
- Standardized changelog format to follow *Keep a Changelog* specification.

### Removed
- Support for Python versions below 3.10.

### Fixed
- Packaging now correctly includes required data files in source distributions.


## [0.6.2] - 2026-03-01

### Fixed

- Build packs all required `data` from data directory


## [0.6.1] - 2026-02-04

### Fixed

- corrected INI configuration parsing to align ConfigParser getters with documented setting types
- fixed multiple settings previously read via incorrect getters (getfloat/getint/getboolean)
- added strict and safe tuple parser for ``rv_lambda_interval`` with type preservation (int/float)
- improved robustness of settings loading against valid configurations that previously failed to load

### Changed

- added tests enforcing correct INI typing and value integrity during settings import


## [0.6.0] - 2026-02-04

### Added

- first run configuration manager - minimal required configuration wizzard on first start when not configured
- download manager - download limb darkening and atmospheres via download manager instead of manual copying it
- extended support for Python 3.9 up to 3.12
- Passbands (notation is adopted from Van Hammes Limb Darkening tables 2019 release)
    - Gaia.2010.G (equivalent to previous GaiaDR2)
    - Gaia.2010.BP
    - Gaia.2010.RP

### Changed

- ability to set default_discretization_factor that governs fidelity of the surface mesh
- ability to automatically fill the semi-major axis (SMA) in fit parameters while fitting the LC data.
  In such cases, the SMA cannot be derived and it has to be fixed at some sensible value which will put component
  surface gravity accelerations within the the range supported by atmospheric models.
  The `LCBinaryAnalyticsTask.set_result()` and `load_result()` functions have default argument
  `autofill_sma`=True that will try to generate a sensible value of SMA if `semi_major_axis` fitting parameter is
  missing in initial fitting parameters JSON/dictionary.
- setting custom atmosphere models and limb-darkening coefficients for components of modelled system. In case of
  `SingleSystem` and `Binary system`, custom atmosphere model is set with `atmosphere` argument of the `Star`
  instance and custom limb-darkening coefficients can be passed in `limb_darkening_coefficients` for each
  passband filter. Custom limb-darkening coefficients are, however, set constant across the whole surface. In case
  of fitting tasks, custom atmospheres and limb-darkening coefficients are passed as arguments of `AnalyticsTask'
  instance.
- `BinarySystem` and `SingleSystem` have a new parameter `distance` which defines a distance between observer on
  the system's centre of mass. If not supplied, default value of 10 pc is used.
- Observer module is now capable of producing light curves ni magnitudes by setting `Observer.flux_unit = u.mag`
  or by keyword argument `flux_unit` in Observer.observe.lc() function.
- New configuration parameter `MAGNITUDE_SYSTEM` was introduced to define sets of zero points used to
  calculate magnitudes. Available magnitude system are `vega`(default), `ab`, `st`.
- Updated package dependecies and basecode to run ELISa on Python versions from 3.6 up to 3.12.

### Fixed

- configuration parser will not crash when `general.home` is set in config file
- prior probability in case of normal distribution now clips the edges of the
  distributions correctly according to `min` and `max` fit parameter configuration arguments.


## [0.5.1] - 2021-11-04

### Fixed

- fixed requirements to avoid installation error::

    ERROR: packaging 21.2 has requirement pyparsing<3,>=2.0.2, but you'll have pyparsing 3.0.4 which is incompatible.


## [0.5] - 2021-10-20

### Added

- added `black_body` as one of possibilities for atmospheres
- support different atmospheres for celestial objects
- `velocity` and `radial_velocity` option for `colormap` argument added to BinarySystem.plot.surface() and
  SingleSystem.plot.surface()
- ability to select priors from uniform or normal distribution, standard deviation of the normal distribution is
  defined with the `sigma` fit parameter attribute

### Changed

- solar constant conserved with different levels of surface discretization
- improvements to trapezoidal discretization
- additional constraints for approximations used during integration of eccentric light curves,
  relative change in irradiation is checked when similar orbital positions are evaluated, improves precision
- pre-build logging schemas added, that are accessible via LOG_CONFIG parameter with options 'default' or 'fit' or
  path to custom configuration file. 'fit' schema will suppress all logging messages except for messages from
  analytics class.
- utilizing numba for computationally heavy tasks such as reflection effect (preparation for GPU ready version of
  ELISa)
- function elisa.analytics.tasks.load_results() returns results in form of dict
- command set_up_logging() not needed anymore while changing logging schemas
- adaptive and custom sampling during fitting accessed by 'samples' argument
- ability for surface plot to return figure instance with boolean argument `return_figure_instance`
- correction of surface underestimation is separately tuned for each discretization method (single star, detached,
  over-contact)
- ablility to filter flat chain to be within specific interval of parameters using AnalyticsTask.filter_chain
  function. This method is suitable for examining multiple solutions.
- ability to evaluate R^2 for set of model parameters using function AnalyticsTask.coefficient_of_determination.
- adding ability to load from json in "radius"-based format that describes the size of the star with
  `equivalent_radius` instead of polar gravity `polar_log_g` in "standard" format
- BinarySystem and SingleSystem now contain a function build_container that builds a complete model of a system at
  given photometric `phase` or observational `time`.
- in the default mode (when user did not specified the discretization factor), sizes of surface elements of both
  components are scaled in a way, that the surface elements of both components roughly output the same amount of
  flux, if the (min, max) range of discretization factors can be maintained. This prevents from unnecessary surface
  oversampling of smaller and dimmer binary components.


### Fixed

- <binary_system>.init() reinitialize parameters corretly (require fix for pulsations)
- inclination rotation is provided in positive direction instead of negative
- line-of-sight vector is switched from [1, 0, 0] to [-1, 0, 0] to make model consistent with radial velocity
  observations where negative value describes velocity of body moving towards the observer. Azimuth of the body is
  now measured with respect to y-axis. Observer is now located at [-inf, 0, 0]
- atmosphere models are interpolated using flux-based weights instead of temperature based weights
- calculation of surface element visibility was fixed in cases of eclipses caused by stars smaller than surface
  elements on eclipsed components
- starting value for implicit solver adjusted in case of near-side parts of overcontact stars generated in
  cylindrical symmetry from polar_radius to 0.25 * polar radius. This prevents a crash of solver for points near
  the neck.


## [0.4] - 2020-10-01


### Added

- radial velocity curves modelled based on radiometric quantities capable of modelling
  Rossitter effect and effect of spots

### Changed

- dependencies updates
- support Python 3.6|3.7|3.8
- configuration module uses singleton instead of global variables
  > from elisa import settings
- ability to display observation stored in DataSet class using DataSet.plot.display_observation()

### Fixed

- removed faulty curve points produced by multiprocessing curve integration methods
- component's volume conserved for eccentric spotty orbits
- surface areas produced by numeric noise when total eclipse is occuring are mitigated
- renormalization of temperature (temperatures powered to exponent of 4)


## [0.3.1] - 2020-08-19

### Changed

- fit_summary (result_summary) function now enables full propagation of errors using `propagate_errors` argument

### Fixed

- on-demand normalization of light curves
- mcmc chain evaluator often crashed when fitting system with component filling its roche lobe, fixed by snapping
  surface potential to critical potentials if they are within errors from fitted potential
- wrong intervals used in corner and trace plot, now fitting confidence intervals instead of fit intervals
- more suitable form of cost function for least squares fitting method
- correcting secondary potential derivative component
- libration motion accounted for in spot position in case of eccetric orbits
- fix: volume conserved in eccentric spotty systems


## [0.3] - 2020-06-17

### Added

* **single system**
  - light curve calculation of single stars with spots and pulsations

* **analytics api** *
    - more user frendly analytics api
    - summary outputs of fitting
    - extended i/o of fitting

* **computaional** *
    - TESS passband (limb darkening tables included)

### Fixed

- fitting light curves of over-contact binaries won't crash with missing `param` error due to invalid constraint setting on backend
- normalize lightcurves (during fitting procedure) each on its max values instead of normalization on global maximum
- MCMC penalisation in case of invalid binary system return big negative number instead of value near to 0.0
- raise `elisa.base.error.AtmosphereError` when atmosphere file not founf instead `FileNotFoundError`
- lc observation atmosphere is not hardcode to `ck04` anymore
- small spots do not cause crashes
- mcmc chain evaluator often crashed when fitting system with component filling its roche lobe, fixed by snapping
  surface potential to critical potentials if they are within errors from fitted potential


## [0.2.3] - 2020-05-27

### Fixed

- fitting light curves of over-contact binaries won't crash with missing `param` error due to invalid constraint setting on backend
- normalize lightcurves (during fitting procedure) each on its max values instead of normalization on global maximum
- MCMC penalisation in case of invalid binary system return big negative number instead of value near to 0.0
- raise `elisa.base.error.AtmosphereError` when atmosphere file not founf instead `FileNotFoundError`


## [0.2.2] - 2020-01-29

### Fixed

- radial velocity curves orientation
- fixed requirements in setupy.py
- fixed requirements in docs


## [0.2.1] - 2020 2020-01-17

### Fixed

- spots discretization managed by parent object if not specified otherwise
- valid detection of spots on over-contact neck


## [0.2] - 2019-12-29

### Added

* **binary system radial velocities curves modeling**

    - radial velocity curves based on movement of center of mass computed upon astro-community quantities (:math:`q`, :math:`asini`)

* **capability to compute lightcurves on several processor's cores (multiprocessing)**

    - split supplied phases to `N` smaller batches (N is equal to desired processes but up to number of available cores) and computed all at once

* **fitting parameters of binary system**

    - light curve fitting using ``Markov Chain Monte Carlo`` (capability to fit using standard physical parameters :math:`M_1`, :math:`M_2` or parameters used by community :math:`q` (mass ratio) and :math:`a` (semi major axis))
    - light curve fitting using ``non-linear least squares`` method (capability to fit using standard physica; parameters :math:`M_1`, :math:`M_2` or parameters used by community :math:`q` (mass ratio) and :math:`a` (semi major axis))
    - radial velocity fitting based on ``Markov Chain Monte Carlo`` method (standard physical parameters, :math:`M_1`, :math:`M_2`, :math:`e`, :math:`i`, :math:`{\omega}`, :math:`{\gamma}`
    - radial velocity fitting based on ``non-linear least squares`` method (standard physical parameters, :math:`M_1`, :math:`M_2`, :math:`e`, :math:`i`, :math:`{\omega}`, :math:`{\gamma}`

* **more specific errors raised**

    - created several different type of errors (see ``elisa.base.errors`` for more information)

### Fixed

- `elisa.observer.Observer.observe.lc` and `elisa.observer.Observer.observe.rv` will not raise an error in case
  when parameter `phases` is `numpy.array` type
- adaptive discretization of binaries do not allow to change distretization factor out of prescribed boundaries
  (it used to lead to small amount of surface points and then triangulation crashed)
- app does not crash on `phase_interval_reduce` in observer during light curve computation
  if BinarySystem is not used from direct import of `BinarySystem`
- const PI multiplicator removed from output flux (still require investigation)
- app does not crash if `bolometric` passband is used
- np.int32/64 and np.float32/64 are considered as valid values on binary system initialization


## [0.1] - 2019-11-06

### Added


* **binary system modeling**

    - points surface generation from generalized surface potential
    - triangulation (faces creation) of component`s surface points
    - physical quantities (gravity, temperatures) distribution over component surface (faces)
    - surface spots
    - temperature pulsations effect
    - light curves modeling for circular synchronous/asynchronous orbits with spotty/no-spotty components
    - radial velocity curves based on movement of center of mass

* **binary system visualization**

    - surface points plot
    - surface wire mesh plot
    - surface faces plot with visualization of distribution of physical quantities
    - orbit plot
    - animations of orbital motions


# Future plans

## v1.0
    - web GUI and API