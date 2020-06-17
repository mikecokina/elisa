Change Log
==========
|


v0.1_
-----
.. v0.1_: https://github.com/mikecokina/elisa/commits/release/0.1

**Release date:** 2019-11-06

**Features**


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


v0.2_
-----
.. v0.2_: https://github.com/mikecokina/elisa/commits/release/0.2

**Release date:** 2019-12-29

**Features**

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

**Fixes**

- `elisa.observer.Observer.observe.lc` and `elisa.observer.Observer.observe.rv` will not raise an error in case
  when parameter `phases` is `numpy.array` type
- adaptive discretization of binaries do not allow to change distretization factor out of prescribed boundaries
  (it used to lead to small amount of surface points and then triangulation crashed)
- app does not crash on `phase_interval_reduce` in observer during light curve computation
  if BinarySystem is not used from direct import of `BinarySystem`
- const PI multiplicator removed from output flux (still require investigation)
- app does not crash if `bolometric` passband is used
- np.int32/64 and np.float32/64 are considered as valid values on binary system initialization


v0.2.1_
-------
.. v0.2.1_: https://github.com/mikecokina/elisa/commits/release/0.2.1

**Release date:** 2020 2020-01-17

**Fixes**

- spots discretization managed by parent object if not specified otherwise
- valid detection of spots on over-contact neck

v0.2.2_
-------
.. v0.2.2_: https://github.com/mikecokina/elisa/commits/release/0.2.2

**Release date:** 2020-01-29

**Fixes**

- radial velocity curves orientation
- fixed requirements in setupy.py
- fixed requirements in docs

v0.2.3_
-------
.. v0.2.3_: https://github.com/mikecokina/elisa/commits/release/0.2.3

**Release date:** 2020-05-27

**Fixes**

- fitting light curves of over-contact binaries won't crash with missing `param` error due to invalid constraint setting on backend
- normalize lightcurves (during fitting procedure) each on its max values instead of normalization on global maximum
- MCMC penalisation in case of invalid binary system return big negative number instead of value near to 0.0
- raise `elisa.base.error.AtmosphereError` when atmosphere file not founf instead `FileNotFoundError`

v0.3_
-----

.. v0.3_: https://github.com/mikecokina/elisa/commits/release/0.3

**Release date:** ? - current (hopefully somewhat stable) version available in dev branch

**Features**

* **single system**
    - light curve calculation of single stars with spots and pulsations

* **analytics api** *
    - more user frendly analytics api
    - summary outputs of fitting
    - extended i/o of fitting

* **computaional** *
    - TESS passband (limb darkening tables included)

**Fixes**

    - fitting light curves of over-contact binaries won't crash with missing `param` error due to invalid constraint setting on backend
    - normalize lightcurves (during fitting procedure) each on its max values instead of normalization on global maximum
    - MCMC penalisation in case of invalid binary system return big negative number instead of value near to 0.0
    - raise `elisa.base.error.AtmosphereError` when atmosphere file not founf instead `FileNotFoundError`
    - lc observation atmosphere is not hardcode to `ck04` anymore
    - small spots do not cause crashes


Future plans
============

v0.4
----
    - pulsations

v0.5
----
    - genetic algorithm
    - extended fitting methods


v1.0
----
    - web GUI and API
