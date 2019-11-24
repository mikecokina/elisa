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


v0.2.dev0_
----------
.. v0.2.dev0_: https://github.com/mikecokina/elisa

**Release date:** In progress

**Features**

* **binary system radial velocities curves modeling**

    - radial velocity curves based on movement of center of mass computed upon astro-community quantities (:math:`q`, :math:`asini`)

* **capability to compute lightcurves on several processor's cores (multiprocessing)**

    - split supplied phases to `N` smaller batches (N is equal to desired processes but up to number of available cores) and computed all at once
* **fitting parameters of binary system**

    - light curve fitting using ``Markov Chain Monte Carlo`` (capability to fit using standard physical parameters :math:`M_1`, :math:`M_2` or parameters used by community :math:`q` (mass ratio) and :math:`a` (semi major axis))
    - light curve fitting using ``non-linear least squares`` method (capability to fit using standard physica; parameters :math:`M_1`, :math:`M_2` or parameters used by community :math:`q` (mass ratio) and :math:`a` (semi major axis))
    - radial velocity fitting based on ``non-linear least squares`` method (standard physical parameters, :math:`M_1`, :math:`M_2`, :math:`e`, :math:`i`, :math:`{\omega}`, :math:`{\gamma}`

* **more specific errors raised**

    - created several different type of errors (see ``elisa.base.errors`` for more information)

**Fixes**

- `elisa.observer.Observer.observe.lc` and `elisa.observer.Observer.observe.rv` will not raise an error in case
  when parameter `phases` is `numpy.array` type
- adaptive discretization of binaries do not allow to change distretization factor out of prescribed boundaries
  (it used to lead to small amount of surface points and triangulation crashed)



Future plans
============

v0.3
----
    - pulsations

v0.4
----
    - genetic algorithm
    - single system
    - extended fitting methods
