|Travis build|  |GitHub version|  |Licence GPLv2|

.. |Travis build| image:: https://travis-ci.org/mikecokina/elisa.svg?branch=dev
    :target: https://travis-ci.org/mikecokina/elisa

.. |GitHub version| image:: https://img.shields.io/badge/version-0.2.dev0-yellow.svg
   :target: https://github.com/Naereen/StrapDown.js

.. |Licence GPLv2| image:: https://img.shields.io/badge/License-GNU/GPLv2-blue.svg
   :target: https://github.com/Naereen/StrapDown.js


Eclipsing binaries Learning Interactive System
==============================================

ELISa
-----

**ELISa** is crossplatform python package dedicated to light curves modelling of close eclipsing binaries including
surface features. Current capabilities include:

    - ``BinarySystem:`` class for modelling surfaces of detached, semi-detached and over-contact binaries
    - ``Observer:`` class for generating light curves (and in future other observables)
    - ``Spots:`` class for generating stellar spot with given longitude, latitude, radius and temperature factor
    - ``Fitting methods`` provide capability to fit radial velocities curves and light curves via implementaion of
      ``non-linear least squares`` method and it also implements lightcurve fitting via ``Markov Chain Monte Carlo``
      method.

**ELISa** is currently still under development. Following features are in progress:

    - ``SingleSystem:`` class for modelling surfaces of single star light curves with full implementation of spots and
      pulsations
    - ``Pulsations:`` class for modelling low amplitude pulsations based on spherical harmonics

We also plan to implement following features:

    - addition of radial velocity curves to Observer class with ``Rossiter-McLaughlin`` effect
    - ``LC`` and ``RV`` fitting using various methods
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

    pip install git+https://github.com/mikecokina/elisa.git@dev

Minimal configuration
---------------------

``ELISa`` require before first run minimal configuration provided by config file. Basically it is necessary to download
atmospheres models, limbdarkening tables and configure path to directories where files will be stored.

Where to find atmospheres and also atmospheres structure is explained in Atmospheres_
as well as limb darkening in Limb-Darkening_.

.. _Atmospheres: https://github.com/mikecokina/elisa/tree/dev/atmosphere
.. _Limb-Darkening: https://github.com/mikecokina/elisa/tree/dev/limbdarkening

Models might be stored on your machine in directory wherever you desire. For purpose of following guide, lets say you
want ot use ``Castelli-Kurucz 2004`` models stored in directory ``/home/user/castelli_kurucz/ck04`` and Van Hamme
limb darkening models in directory ``/home/user/van_hamme_ld/vh93``. You have to create configuration ``ini`` file where
model and directories will be specified. Just assume, name of our configuration file is ``elisa_config.ini`` located in
path ``/home/user/.elisa/``. Then content of your configuration file should be at least like following::

    [support]
    van_hamme_ld_tables = /home/user/van_hamme_ld/vh93
    castelli_kurucz_04_atm_tables = /home/user/castelli_kurucz/ck04
    atlas = ck04

Full content of configuration file with description might be found here, Elisa-Configuration-File_

.. _Elisa-Configuration-File: https://github.com/mikecokina/elisa/blob/dev/src/elisa/conf/elisa_conf_docs.ini

:warning: atmospheric models and limb darkening tables are not in native format as usually provided on web sites. Models have been altered to form required for elisa.

Now, you have to tell ELISa, where to find configuration file. In environment you are using setup environment variable
`ELISA_CONFIG` to full path to config file. In UNIX like operation systems it is doable by following command::

    export ELISA_CONFIG=/home/user/.elisa/elisa_config.ini

There is plenty ways how to setup environment variable which vary on operation system and also on tool (IDE)
that you have in use.

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

To speedup computaion of light curves, paralellization of processes has been implemented. Practically, computation
of light curve points is separated to smaller batches and each batch is evaluated on separated CPU core. Paralelliation
necessarily bring some overhead to process and in some cases might cause even slower behavior of application.
It is important to choose wisely when use it espeially in case of circular synchronous orbits which consist of spot-free
components.

Down below are shown some result of multiprocessor approach for different binary system type.


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

  Paralellization benchmark for ``detached eccentric synchronous`` star system.

:note: outliers in charts are caused by symetrization hitted during benchmark process


Binary Stars Radial Curves Fitting
----------------------------------

In current version of `ELISa`, you can use capability to fit curves of radial velocities obtained as velocities
of center of mass from primary and secondary component. An example of synthetic radial vecolity curve is shown below.

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

Each fitting initial input (initial parameters list) has form like::

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

and require all params from following list if you would like to try absolute parameters fitting:

    * ``p__mass`` - mass of primary component in units of Solar mass
    * ``s__mass`` - mass of secondary component in units of Solar mass
    * ``eccentricity`` - eccentricity of binary system, (0, 1)
    * ``inclination`` - inclination of binary system in `degrees`
    * ``argument_of_periastron`` - argument of periastron in `degrees`
    * ``gamma`` - radial velocity of system center of mass in `m/s`

or otherwise, in community approach, you can use instead of ``p__mass``, ``s__mass`` and ``inclination`` parameters:

    * ``asini`` - product of sinus of inclination and semi major axis in units of Solar radii
    * ``mass_ratio`` - mass ratio, known as `q`

There are already specified global minimal an maximal values for parameters, but user is free to adjust parameters
which might work better for him.

Parameter set to be `fixed` is naturaly not fitted and its value is fixed during procedure. User can also
setup `constraint` for any parameter. It is allowed to put bound only parameter with other parameter which is not fixed.
It makes sence, since if you have fixed parameter, you are free to use its numerica value in constraint directly withou
other onstructions.

In this part you can see minimal example of base code providing fitting. Sample radial velocity curve was obtained
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
        rv = {'primary': [59290.08594439, 54914.25751111, 42736.77725629, 37525.38500226,..., -15569.43109441]),
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
            }
        ]

        result = central_rv.fit(xs=phases, ys=rv, period=4.5, x0=rv_initial, xtol=1e-10, yerrs=None)

    if __name__ == '__main__':
        main()



Result of fitting procedure was estimated as

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
            "r_squared": 0.998351027628904
        }
    ]


.. image:: ./docs/source/_static/readme/rv_fit.svg
  :width: 70%
  :alt: rv_fit.svg
  :align: center

Another approach is to use implemented fitting method based on `Markov Chain Monte Carlo`. Read data output requires
more analytics skills, some minimal expirience with MCMC since output is not simple dictionary of values but
it is basically descriptive set of parameters progress during evaluation of method.

Following represents minimalistic base code which should explain how to use mcmc method and how to read outputs.

.. code:: python

    import numpy as np
    from elisa.analytics.binary.mcmc import central_rv


    def main():
        phases = np.arange(-0.6, 0.62, 0.02)
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
            }
        ]

        central_rv.fit(xs=phases, ys=rv, period=0.6, x0=rv_initial, nwalkers=20,
                       nsteps=10000, nsteps_burn_in=1000, yerrs=None)

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

There are also such informations stored in "elisa home" in json file, so you are able to parse and work with each
previous run.

Binary Stars Light Curves Fitting
---------------------------------

Packgae `elisa` currently implements two approaches to be able provide very basic fitting of light curves.
First method is standard approach which use `non-linear least squares` method algorithm and second rule
Markov Chain Monte Carlo (`MCMC`) method.

Following chapter is supposed to give you brief information about capabilities provided by `elisa`.
Lets assume that we have a given light curve like shown below generated on parameters::

    {
        'mass_ratio': 0.5,
        'semi_major_axis': 16.54321389,
        'p__t_eff': 6000.0,
        'p__surface_potential': 5.0,
        's__t_eff': 8000.0,
        's__surface_potential': 6.0,
        'inclination': 85.0,
        'eccentricity': 0.0,
        'p__beta': 1.0,
        's__beta': 1.0,
        'p_albedo': 1.0,
        's__albedo': 1.0
    }



.. image:: ./docs/source/_static/readme/lc_example.svg
  :width: 70%
  :alt: lc_example.svg
  :align: center

