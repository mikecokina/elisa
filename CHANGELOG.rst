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

* **capability to compute lightcurves on several processor's cores (multiprocessing)**

    - split supplied phases to `N` smaller batches (N is equal to desired processes but up to number of available cores) and computed all at once
* **fitting parameters of binary system** *

    - radial velocity fitting based on ``least squares`` method

**Fixes**

- `elisa.observer.Observer.observe.lc` and `elisa.observer.Observer.observe.rv` will not raise an error in case
  when parameter `phases` is `numpy.array` type



Future plans
============

v0.2
----

    - basic fitting methods

v0.3
----
    - single system

v0.4
----
    - genetic algorithm
    - pulsations
    - extended fitting methods
