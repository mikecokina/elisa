|Travis build|  |GitHub version|  |Licence GPLv2|

.. |Travis build| image:: https://travis-ci.org/mikecokina/elisa.svg?branch=dev
    :target: https://travis-ci.org/mikecokina/elisa

.. |GitHub version| image:: https://img.shields.io/badge/version-0.2-yellow.svg
   :target: https://github.com/Naereen/StrapDown.js

.. |Licence GPLv2| image:: https://img.shields.io/badge/License-GNU/GPLv2-blue.svg
   :target: https://github.com/Naereen/StrapDown.js


Eclipsing binaries Learning Interactive System
==============================================

ELISa
-----

**ELISa** is crossplatform python package dedicated to light curves modelling of close eclipsing binaries including
surface features such as spots and pulsation (soon). Current capabilities include:

    - ``BinarySystem:`` class for modelling surfaces of detached, semi-detached and over-contact binaries
    - ``Observer:`` class for generating light curves (and in future other observables)
    - ``Spots:`` class for generating stellar spot with given longitude, latitude, radius and temperature factor
    - ``Fitting methods`` provide capabilities to fit radial velocities curves and light curves via implementaion of
      ``non-linear least squares`` method and also via ``Markov Chain Monte Carlo`` method.

**ELISa** is currently still under development. Following features are in progress:

    - ``SingleSystem:`` class for modelling surfaces of single star light curves with full implementation of spots and
      pulsations
    - ``Pulsations:`` class for modelling low amplitude pulsations based on spherical harmonics solution

We also plan to implement following features:

    - addition of radial velocity curves to Observer class with ``Rossiter-McLaughlin`` effect
    - some extended ``LC`` and ``RV`` fitting using various additional methods and features like classification via ``neural-networks``
    - addition of synthetic spectral line modelling

Requirements
------------

**ELISa** is a python package which requires ``python v3.6+`` and has following dependencies::

    astropy==2.0.2
    cycler==0.10.0
    corner==2.0.1
    emcee==3.0.1
    jsonschema==3.2.0
    matplotlib==2.1.0
    numpy==1.16.2
    pandas==0.24.0
    py==1.4.34
    pyparsing==2.2.0
    pypex==0.1.0
    pytest==3.2.3
    python-dateutil==2.6.1
    pytz==2017.2
    scipy==1.0.0
    six==1.11.0


and potentially also **python-tk** package or equivalent for matplotlib package to display the figures correctly.

:note: although python distribution and package versions are specified precisely, that does not mean that the package will not work with higher versions, only that it was not tested with higher versions of packages. However we highly recommend to stick with python distribution and package versions listed above.


Install
-------

In case of ``ELISa`` the easiest and the safest way to install is to create python virtual
environment and install all requirements into it. Bellow is a simple guide, how to od it. Details of installation differ
in dependence on the selected operating system.

Ubuntu [or similar]
~~~~~~~~~~~~~~~~~~~

First, you have to install Python 3.6 or higher. In latest stable version ``Ubuntu 18.04`` there is already preinstalled
python `3.6.x`. In older versions, you will have to add repository and install it manually. There is several quides
on the internet that will help you with installation, e.g. Python_3.6_

.. _Python_3.6: http://ubuntuhandbook.org/index.php/2017/07/install-python-3-6-1-in-ubuntu-16-04-lts/

Install ``pip3`` python package manager if is not already installed on your system, usually by execution of command::

    apt install -y python3-pip

or you can also use `raw` python script which provide installation via ``python``::

    curl https://bootstrap.pypa.io/get-pip.py | python3.6

Install virtual environment by command::

    pip3 install virtualenv


To create virtual environment, create directory where python virtual environment will be stored,
e.g. ``/<any>/<path>/elisa/venv``
and run following command::

    virtualenv /<any>/<path>/elisa/venv --python=python3.6

After few moments you virtual environment is created and ready for use. In terminal window, activate virtual
environment::

    . /<any>/<path>/elisa/venv/bin/activate

When virtual environment is activated, install ``elisa`` package in `dev` version. Execute following command::

    pip install git+https://github.com/mikecokina/elisa.git@dev

You will probably also need to install::

    apt install -y python3-tk


Windows
~~~~~~~

To install python in windows, download ``python 3.6.x`` installation package from official python web site.
Installation package will create all necessary dependencies except of virtual environment package.
Install virtual environment by execution of following command in command line::

    pip3 install virtualenv

Make sure a proper version of  python and pip is used. When done, create directory where virtual environment will be
stored and run::

    virtualenv /<any>/<path>/elisa --python<path>/<to>/python3.6/python.exe

It is common to specify full path to ``python.exe`` file under Windows, otherwise It might not work.

Now, when virtual environment is prepared, run::

    . /<any>/<path>/elisa/Scripts/activate

And finally install ``ELISa``::

    pip install elisa


How to build API docs
=====================

Use virtual environment created in previous steps and add following dependencies::

    Sphinx==2.2.0
    sphinx-rtd-theme==0.4.3

.. _docs: https://github.com/mikecokina/elisa/tree/master/docs

Change directory to cloned docs_ and use following command::

    sphinx-build -W -b html -c .\source -d .\build\doctrees .\source\ .\build\


Minimal configuration
---------------------

``ELISa`` requires minimal configuration before the first run provided by the config file. Initially, it is necessary to
download Atmospheres_ models and Limb-Darkening_ tables.

.. _Atmospheres: https://github.com/mikecokina/elisa/tree/dev/atmosphere
.. _Limb-Darkening: https://github.com/mikecokina/elisa/tree/dev/limbdarkening

Models can be stored on your machine in directory of your choosing. Lets say you want ot use ``Castelli-Kurucz 2004``
models stored in directory ``/home/user/castelli_kurucz/ck04`` and Van Hamme limb darkening models in directory
``/home/user/van_hamme_ld/vh93``. You have to create configuration ``ini`` file where
model and directories will be specified. Now assume that name of our configuration file is ``elisa_config.ini`` located
in path ``/home/user/.elisa/``. Then content of your configuration file should be at least like following::

    [support]
    van_hamme_ld_tables = /home/user/van_hamme_ld/vh93
    castelli_kurucz_04_atm_tables = /home/user/castelli_kurucz/ck04
    atlas = ck04

Full content of configuration file with description might be found here, Elisa-Configuration-File_

.. _Elisa-Configuration-File: https://github.com/mikecokina/elisa/blob/master/src/elisa/conf/elisa_conf_docs.ini

:warning: atmospheric models and limb darkening tables for this package are stored in industry standard ''.csv'' files.
          Therefore, their native format as usually provided on web sites is not suitable for Elisa and require
          conversion to standard format. Models have been altered to form required for Elisa.

Now, you have to tell ELISa, where to find configuration file. In environment you are using setup environment variable
`ELISA_CONFIG` to full path to config file. In UNIX like operation systems it is doable by following command::

    export ELISA_CONFIG=/home/user/.elisa/elisa_config.ini

There is plenty ways how to setup environment variable which vary on operation system and also on tool (IDE)
that you have in use. Optionally, you can use ``config.ini`` file located in ``ELISa_folder/src/elisa/conf/`` without
any need for setting the enviromental variable.

Now you are all setup and ready to code.


Usage
-------
For in depth tutorials, see directory ``elisa/jupyter_tutorials``


Available passbands
-------------------

::

    bolometric
    Generic.Bessell.U
    Generic.Bessell.B
    Generic.Bessell.V
    Generic.Bessell.R
    Generic.Bessell.I
    SLOAN.SDSS.u
    SLOAN.SDSS.g
    SLOAN.SDSS.r
    SLOAN.SDSS.i
    SLOAN.SDSS.z
    Generic.Stromgren.u
    Generic.Stromgren.v
    Generic.Stromgren.b
    Generic.Stromgren.y
    Kepler
    GaiaDR2


Multiprocessing
---------------

To speedup computation of light curves, paralellization of computations has been implemented. Computation
of light curve points is separated to smaller batches and each batch is evaluated on separated CPU core. Paralellization
necessarily brings some overhead to process and in some cases might cause even slower behavior of application.
It is important to choose wisely when use it espeically in case of circular synchronous orbits which consist of
spot-free components where multiprocessing is usually not as effective.

Down below are shown some result of multiprocessor approach for different binary system types. Absolute time necessary
for calculation of the light curve is highly dependent on the type of the system and hardaware. Therefore we have
normalized the time axis according to maximum value in our datasets.

.. figure:: ./docs/source/_static/readme/detached.circ.sync.svg
  :width: 70%
  :alt: detached.circ.sync.svg
  :align: center

  Paralellization benchmark for ``detached circular synchronous`` star system.

.. figure:: ./docs/source/_static/readme/detached.circ.async.svg
  :width: 70%
  :alt: detached.circ.async.svg
  :align: center

  Paralellization benchmark for ``detached circular asynchronous`` star system.


.. figure:: ./docs/source/_static/readme/detached.ecc.sync.svg
  :width: 70%
  :alt: detached.ecc.sync.svg
  :align: center

  Paralellization benchmark for ``eccentric synchronous`` star system.

:note: outliers in charts are caused by curve symetrization process


Binary Stars Radial Curves Fitting
----------------------------------

`ELISa` is capable to fit radial velocity curves to observed radial velocities.
In current version of `ELISa`, the radial velocity curves are calculated from radial velocities of centre of mass of
primary and secondary component. An example of synthetic radial velocity curve is shown below.

.. image:: ./docs/source/_static/readme/rv_example.svg
  :width: 70%
  :alt: rv_example.svg
  :align: center

This radial velocity curve was obtained on system with following relevant parameters::

    primary mass: 2.0 [Solar mass]
    secondary mass: 1.0 [Solar mass]
    inclination: 85 [degree]
    argument of periastron: 0.0 [degree]
    eccentricity: 0.0 [-]
    period: 4.5 [day]
    gamma: 20000.0 [m/s]

Each fitted parameter has an input form as follows::

    initial = [
        {
            'value': <float>,
            'param': <str>,
            'fixed': <bool>,
            'min': <float>,
            'max': <float>,
            'constraint': <str>
        }, ...
    ]

and require all params from the following list if you would like to try absolute parameters fitting:

    * ``p__mass`` - mass of primary component (in Solar masses)
    * ``s__mass`` - mass of secondary component (in Solar masses)
    * ``eccentricity`` - eccentricity of binary system, (0, 1)
    * ``inclination`` - inclination of binary system in `degrees`
    * ``argument_of_periastron`` - argument of periastron in `degrees`
    * ``gamma`` - radial velocity of system center of mass in `m/s`
    * ``period`` - period of binary system (in days), usually fixed parameters
    * ``primary_minimum_time`` - numeric time of primary minimum (ny time units); used when exact period is unknown and fitting is required

or otherwise, in "community approach", you can use instead of ``p__mass``, ``s__mass`` and ``inclination`` parameters:

    * ``asini`` - in Solar radii
    * ``mass_ratio`` - mass ratio (M_2/M_1), also known as `q`

There are already specified global minimal and maximal values for parameters, but user is highly encouraged to adjust
parameter boundaries which might work better for the particular case.

Parameter set to be `fixed` will not be fitted and its value will stay fixed during the fitting procedure. User can
also setup `constraint` for any parameter, e.g.::

    {
        'value': 16.515,
        'param': 'semi_major_axis',
        'constraint': '16.515 / sin(radians({inclination}))'
    },

It is allowed to put bounds (constraints) only on parameter using other free parameters, otherwise the parameter should stay fixed.
For example, it makes no sense to set bound like this::

    {
        'value': 5000.0,
        'param': 'p__temperature',
        'fixed': True
    },
    {
        'value': 10000.0,
        'param': 's__temperature',
        'constraint': '{p__temperature * 0.5}'
    }


In this part you can see minimal example of code providing fitting. Sample radial velocity curve was obtained
by parameters::

    {
        'eccentricity': '0.0',
        'asini': 16.48026197,
        'mass_ratio': 0.5,
        'argument_of_periastron': 0.0,
        'gamma': 20000.0,
        "period": 4.5,

        "inclination": 85.0,
        "semi_major_axis": 16.54321389
    }

.. code:: python

    import numpy as np
    from elisa.analytics.binary.least_squares import central_rv

    def main():
        phases = np.arange(-0.6, 0.62, 0.02)
        xs = {comp: phases for comp in BINARY_COUNTERPARTS}
        rv = {'primary': [59290.08594439, 54914.25751111, 42736.77725629, 37525.38500226,..., -15569.43109441],
              'secondary': [-52146.12757077, -42053.17971052, -18724.62240468,..., 90020.23738585]}

        rv_initial = [
            {
                'value': 0.0,
                'param': 'eccentricity',
                'fixed': True
            },
            {
                'value': 15.0,
                'param': 'asini',
                'fixed': False,
                'min': 10.0,
                'max': 20.0

            },
            {
                'value': 3,
                'param': 'mass_ratio',
                'fixed': False,
                'min': 0,
                'max': 10
            },
            {
                'value': 0.0,
                'param': 'argument_of_periastron',
                'fixed': True
            },
            {
                'value': 30000.0,
                'param': 'gamma',
                'fixed': False,
                'min': 10000.0,
                'max': 50000.0
            },
            {
                'value': 4.5,
                'param': 'period',
                'fixed': True
            }
        ]

        result = central_rv.fit(xs=xs, ys=rv, x0=rv_initial, xtol=1e-10, yerrs=None)

    if __name__ == '__main__':
        main()


Result of fitting procedure is displayed in the following format:

.. code:: python

    [
        {
            "param": "asini",
            "value": 16.515011290521596,
            "unit": "solRad"
        },
        {
            "param": "mass_ratio",
            "value": 0.49156922351202637,
            "unit": "dimensionless"
        },
        {
            "param": "gamma",
            "value": 19711.784379242825,
            "unit": "m/s"
        },
        {
            "param": "eccentricity",
            "value": 0.0,
            "unit": "dimensionless"
        },
        {
            "param": "argument_of_periastron",
            "value": 0.0,
            "unit": "degrees"
        },
        {
            "param": "period",
            "value": 4.5,
            "unit": "days"
        },
        {
            "r_squared": 0.998351027628904
        }
    ]


.. image:: ./docs/source/_static/readme/rv_fit.svg
  :width: 70%
  :alt: rv_fit.svg
  :align: center

Another approach is to use implemented fitting method based on `Markov Chain Monte Carlo`. Reading data output requires
an experience with MCMC since output is not simple dictionary but a descriptive set of parameters progress during
evaluation of method.

Following represents minimalistic code which should explain how to use mcmc method and how to read outputs.

.. code:: python

    import numpy as np
    from elisa.analytics.binary.mcmc import central_rv


    def main():
        phases = np.arange(-0.6, 0.62, 0.02)
        xs = {comp: phases for comp in BINARY_COUNTERPARTS}

        rv = {'primary': [59290.08594439, 54914.25751111, 42736.77725629, 37525.38500226,..., -15569.43109441]),
              'secondary': [-52146.12757077, -42053.17971052, -18724.62240468,..., 90020.23738585]}

        rv_initial = [
            {
                'value': 0.2,
                'param': 'eccentricity',
                'fixed': False,
                'max': 0.0,
                'min': 0.5
            },
            {
                'value': 15.0,
                'param': 'asini',
                'fixed': False,
                'min': 10.0,
                'max': 20.0

            },
            {
                'value': 3,
                'param': 'mass_ratio',
                'fixed': False,
                'min': 0,
                'max': 10
            },
            {
                'value': 0.0,
                'param': 'argument_of_periastron',
                'fixed': True
            },
            {
                'value': 30000.0,
                'param': 'gamma',
                'fixed': False,
                'min': 10000.0,
                'max': 50000.0
            },
            {
                'value': 4.5,
                'param': 'period',
                'fixed': True
            }
        ]

        central_rv.fit(xs=xs, ys=rv, x0=rv_initial, nwalkers=20, nsteps=10000, nsteps_burn_in=1000, yerrs=None)

        result = central_rv.restore_flat_chain(central_rv.last_fname)
        central_rv.plot.corner(result['flat_chain'], result['labels'], renorm=result['normalization'])

    if __name__ == '__main__':
        main()

Result of code above is corner plot which might looks like this one

.. image:: ./docs/source/_static/readme/mcmc_rv_corner.svg
  :width: 95%
  :alt: mcmc_rv_corner.svg
  :align: center

Object `central_rv` keep track of last executed mcmc "simulation" so you can work with output. It stores::

    last_sampler: emcee.EnsembleSampler; last instance of `sampler`
    last_normalization: Dict; normalization map used during fitting
    last_fname: str; filename of last stored flatten emcee `sampler` with metadata

The same information is stored in "elisa home" in json file, so you are able to access each
previous run.


Binary Stars Radial Curves Fitting - No Ephemeris
-------------------------------------------------

In case we do not have enough information / measurements and we are not able determine ephemeris with
desired accuracy, analytics modules of elisa are capable to handle such situation and gives you tools to fit
period and primary minimum time as unknown parameters. In such case, `xs` values has to be supplied in form::

    {
        "primary": [jd0, jd1, ..., jdn],
        "secondary": [jd0, jd1, ..., jdn],
    }

Based on primiary minimum time and period adjusted in fitting proces, JD times are transformed to phases within process
itself.

:warning: make sure you have reasonable boundaries set for `primary_minimum_time` and `period`

Initial parameters for ``primary_minimum_time`` and ``period`` fitting might looks like following::

    [
        {
            'value': 0.0,
            'param': 'eccentricity',
            'fixed': True
        },
        {
            'value': 15.0,
            'param': 'asini',
            'fixed': False,
            'min': 10.0,
            'max': 20.0

        },
        {
            'value': 3,
            'param': 'mass_ratio',
            'fixed': False,
            'min': 0,
            'max': 10
        },
        {
            'value': 0.0,
            'param': 'argument_of_periastron',
            'fixed': True
        },
        {
            'value': 30000.0,
            'param': 'gamma',
            'fixed': False,
            'min': 10000.0,
            'max': 50000.0
        },
        {
            'value': 4.4,
            'param': 'period',
            'fixed': False,
            'min': 4.4,
            'max': 4.6
        },
        {
            'value': 11.1,
            'param': 'primary_minimum_time',
            'fixed': False,
            'min': 11.1,
            'max': 12.1
        }
    ]


:note: values of *primary_minimum_time* are cut off to smaller numbers (toto vysvetli lepsie)

Corner plot of `mcmc` result for such approach is in figure bellow

.. image:: ./docs/source/_static/readme/ mcmc_rv_corner_noperiod.svg
  :width: 95%
  :alt: mcmc_rv_corner_noperiod.svg
  :align: center


Binary Stars Light Curves Fitting
---------------------------------

Package `elisa` currently implements two approaches that provides very basic capability to fit light curves to
observed photometric data. First method is standard approach which use `non-linear least squares` method algorithm and
second approach uses Markov Chain Monte Carlo (`MCMC`) method.

Following chapter is supposed to give you brief information about capabilities provided by `elisa`.
Lets assume that we have a given light curve like shown below generated on parameters::

    {
        'mass_ratio': 0.5,
        'semi_major_axis': 16.54321389,
        'p__t_eff': 8000.0,
        'p__surface_potential': 4.0,
        's__t_eff': 6000.0,
        's__surface_potential': 6.0,
        'inclination': 85.0,
        'eccentricity': 0.0,
        'p__beta': 0.32,
        's__beta': 0.32,
        'p_albedo': 0.6,
        's__albedo': 0.6,
        'period': 4.5
    }



.. image:: ./docs/source/_static/readme/lc_example.svg
  :width: 70%
  :alt: lc_example.svg
  :align: center


Lets apply some fitting algorithm to demonstrate the software capabilities. Fitting modules are stored in module path
``elisa.analytics.binary.least_squares`` and ``elisa.analytics.binary.mcmc``. It is up to the user what methods
choose to use. In both cases, there is prepared instances for fitting, called ``binary_detached`` and
``binary_overcontact``. Difference is that ``binary_overcontact`` fitting module keeps surface potential of both binary
components constrained to the same value.

First, we describe the algorithm based on `non-linear least squares` method. Binary system which can generate light
curve shown above is the most probably the detached system, therefore we will use module ``binary_detached``.

:warning: Non-linear least squares method used in such complex problem as fitting light
          curves of eclipsing binaries, might be insuficient in case of initial parametres being
          too far from real values and also too broad fitting boundaries.

Following minimalistic python snippet will show you, how to use ``binary_detached`` fitting module. System parameter
definitions are the same as in case of radial velocities fiting. Following guide is appropriate sequence of steps for
solving the binary system with available radial and photometric data.

First, you should solve radial velocities if available and fix parametres in light curve fitting. Since we were able
to obtain some basic information about our system, we should fix or efficiently truncate boundaries
for following parameters::

    {
        "asini": 16.515,
        "mass_ratio": "0.5",
        "eccentricity": "0.0",
        "argument_of_periastron": 0.0
    }


.. _Ballesteros: https://arxiv.org/pdf/1201.1809.pdf

We can also estimate surface temperature of primary component via formula implemented in `elisa` package.

.. code:: python

    from elisa.analytics import bvi
    b_v = bvi.pogsons_formula(lc['Generic.Bessell.B'][55], lc['Generic.Bessell.V'][55])
    bvi.elisa_bv_temperature(b_v)


This approach give us value ~ 8307K.

:note: index `55` is used because we know that such index will give as flux on photometric phase :math:`\Phi=0.5`,
       where we eliminte impact of secondary component to result of primary component temperature.

:note: we recommend you to set boundaries for temperature obtained from `bvi` module at least in range +/-500K.

Lets create an example for code which demonstrates least squares fitting method.

.. code:: python

    import numpy as np
    from elisa.analytics.binary.least_squares import binary_detached

    phases = {band: np.arange(-0.6, 0.62, 0.02) for band in lc}
    lc = {
            'Generic.Bessell.B': np.array([0.9790975 , 0.97725314, 0.97137167, ..., 0.97783875]),
            'Generic.Bessell.V': np.array([0.84067043, 0.8366796 , ..., 0.8389709 ]),
            'Generic.Bessell.R': np.array([0.64415833, 0.64173746, 0.63749762, ..., 0.64368843])
         }

    lc_initial = [
        {
            'value': 16.515,
            'param': 'semi_major_axis',
            'constraint': '16.515 / sin(radians({inclination}))'
        },
        {
            'value': 8307.0,
            'param': 'p__t_eff',
            'fixed': False,
            'min': 7800.0,
            'max': 8800.0
        },
        {
            'value': 3.0,
            'param': 'p__surface_potential',
            'fixed': False,
            'min': 3,
            'max': 5
        },
        {
            'value': 4000.0,
            'param': 's__t_eff',
            'fixed': False,
            'min': 4000.0,
            'max': 7000.0
        },
        {
            'value': 5.0,
            'param': 's__surface_potential',
            'fixed': False,
            'min': 5.0,
            'max': 7.0
        },
        {
            'value': 85.0,
            'param': 'inclination',
            'fixed': False,
            'min': 80,
            'max': 90
        },
        {
            'value': 1.0,
            'param': 'p__gravity_darkening',
            'fixed': True
        },
        {
            'value': 0.32,
            'param': 's__gravity_darkening',
            'fixed': True
        },
        {
            'value': 1.0,
            'param': 'p__albedo',
            'fixed': True
        },
        {
            'value': 0.6,
            'param': 's__albedo',
            'fixed': True
        },
        {
            'value': 0.0,
            'param': 'argument_of_periastron',
            'fixed': True
        },
        {
            'value': 0.5,
            'param': 'mass_ratio',
            'fixed': True
        },
        {
            'value': 0.0,
            'param': 'eccentricity',
            'fixed': True
        }
    ]

    result = binary_detached.fit(xs=phases, ys=lc, period=4.5, discretization=5.0, x0=lc_initial,
                                 yerrs=None, xtol=1e-10, diff_step=0.001)

    if __name__ == '__main__':
        main()


Such approach leads to solution shown bellow::

    [
        {
            "param": "p__t_eff",
            "value": 7998.79728848134,
            "unit": "K"
        },
        {
            "param": "p__surface_potential",
            "value": 3.967385781004351,
            "unit": "dimensionless"
        },
        {
            "param": "s__t_eff",
            "value": 5914.666436423595,
            "unit": "K"
        },
        {
            "param": "s__surface_potential",
            "value": 6.066890977326584,
            "unit": "dimensionless"
        },
        {
            "param": "inclination",
            "value": 85.84258474614543,
            "unit": "degrees"
        },
        {
            "param": "p__gravity_darkening",
            "value": 0.32,
            "unit": "dimensionless"
        },
        {
            "param": "s__gravity_darkening",
            "value": 0.32,
            "unit": "dimensionless"
        },
        {
            "param": "p__albedo",
            "value": 0.6,
            "unit": "dimensionless"
        },
        {
            "param": "s__albedo",
            "value": 0.6,
            "unit": "dimensionless"
        },
        {
            "param": "argument_of_periastron",
            "value": 0.0,
            "unit": "degrees"
        },
        {
            "param": "mass_ratio",
            "value": 0.5,
            "unit": "dimensionless"
        },
        {
            "param": "eccentricity",
            "value": 0.0,
            "unit": "dimensionless"
        },
        {
            "param": "semi_major_axis",
            "value": 16.558571635780567,
            "unit": "solRad"
        },
        {
            "r_squared": 0.9999500530482149
        }
    ]

:warning: make sure all your light curve values are normalized using the highest value from whole set of
          curves supplied to algorithm

Visualization of fit is

.. image:: ./docs/source/_static/readme/lc_fit.svg
  :width: 70%
  :alt: lc_fit.svg
  :align: center

``Elisa`` also provides lightcurve fitting method based on `Markov Chain Monte Carlo`. Read data output requires
the same level of knowledge as in case of radial velocities fitting.

Bellow you can see minimalistic base code which should demonstrate how to use MCMC method and how to read outputs.


.. code:: python

    import numpy as np
    from elisa.analytics.binary.mcmc import binary_detached

    phases = {band: np.arange(-0.6, 0.62, 0.02) for band in lc}
    lc = {
            'Generic.Bessell.B': np.array([0.9790975 , 0.97725314, 0.97137167, ..., 0.97783875]),
            'Generic.Bessell.V': np.array([0.84067043, 0.8366796 , ..., 0.8389709 ]),
            'Generic.Bessell.R': np.array([0.64415833, 0.64173746, 0.63749762, ..., 0.64368843])
         }

    lc_initial = [
        {
            'value': 16.515,
            'param': 'semi_major_axis',
            'constraint': '16.515 / sin(radians({inclination}))'
        },
        {
            'value': 8307.0,
            'param': 'p__t_eff',
            'fixed': False,
            'min': 7800.0,
            'max': 8800.0
        },
        ...
    ]

    binary_detached.fit(xs=phases, ys=lc, x0=lc_initial, period=4.5, discretization=5.0,
                        nwalkers=20, nsteps=10000, nsteps_burn_in=1000, yerrs=None)
    result = binary_detached.restore_flat_chain(binary_detached.last_fname)
    binary_detached.plot.corner(result['flat_chain'], result['labels'], renorm=result['normalization'])

    if __name__ == '__main__':
        main()

:note: initial value are same as in case of least squares method base code demonstration

Corner plot of `mcmc` result for such approach is in figure bellow

.. image:: ./docs/source/_static/readme/mcmc_lc_corner.svg
  :width: 95%
  :alt: mcmc_lc_corner.svg
  :align: center

.. _example_scripts: https://github.com/mikecokina/elisa/tree/master/scripts/analytics

All example scripts can be found in example_scripts_
