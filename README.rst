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

**ELISa** is currently under development. Following features are in progress:

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
    matplotlib==2.1.0
    numpy==1.16.2
    pandas==0.24.0
    pyparsing==2.2.0
    pypex==0.1.0
    pytest==3.2.3
    python-dateutil==2.6.1
    pytz==2017.2
    py==1.4.34
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


Radial Curves Fitting of Binary Stars
-------------------------------------

In current version of `ELISa`, you can use capability to fit curves of radial velocities obtained as velocities
of center of mass from primary and secondary component. An example of synthetic radial vecolity curve is shown below.

.. image:: ./docs/source/_static/readme/rv_example.png
  :width: 70%
  :alt: rv_example.png
  :align: center

This radial velocity curve was obtained on system with following relevant parameters::

    primary mass: 2.0 [Solar mass]
    secondary mass: 1.0 [Solar mass]
    inclination: 90 [degree]
    argument of periastron: 0.0 [degree]
    eccentricity: 0.2 [-]
    period: 4.5 [day]


Each fitting initial input has form like::

    initial = [
        {
            'value': <float>,
            'param': <str>,
            'fixed': <bool>,
            'min': <float>,
            'max': <float>
        }, ...
    ]

and require all params from list:

    * ``p__mass`` - mass of primary component in units of Solar mass
    * ``s__mass`` - mass of secondary component in units of Solar mass
    * ``eccentricity`` - eccentricity of binary system, (0, 1)
    * ``inclination`` - inclination of binary system in `degrees`
    * ``argument_of_periastron`` - argument of periastron in `degrees`
    * ``gamma`` - radial velocity of system center of mass in `m/s`

There are already specified global minimal an maximal values for parameters, but user is free to adjust parameters
which might work better.

Parameter set to be `fixed` is naturaly not fitted and its value is fixed during procedure.

In this part you can see minimal example of base code providing fitting

.. code:: python

    import numpy as np
    from elisa.analytics.binary.fit import central_rv

    def main():
        phases = np.arange(-0.6, 0.62, 0.02)
        rv = {
            'primary': np.array([79218.00737957, 76916.16835599, 74104.73384787, 70765.71345562, ...]),
            'secondary': np.array([-59436.01475914, -54832.33671198, -49209.46769573, -42531.42691124, ...])
        }

        rv_initial_parameters = [
            {
                'value': 0.1,
                'param': 'eccentricity',
                'fixed': False,
                'min': 0,
                'max': 1

            },
            {
                'value': 90.0,
                'param': 'inclination',
                'fixed': True,

            },
            {
                'value': 1.8,
                'param': 'p__mass',
                'fixed': False,
                'min': 1,
                'max': 3
            },
            {
                'value': 1.0,
                'param': 's__mass',
                'fixed': True,
            },
            {
                'value': 0.0,
                'param': 'argument_of_periastron',
                'fixed': True
            },
            {
                'value': 20000.0,
                'param': 'gamma',
                'fixed': False,
                'min': 20000,
                'max': 40000
            }
        ]


        result = central_rv.fit(xs=phases, ys=rv, period=4.5, x0=rv_initial_parameters)

    if __name__ == '__main__':
        main()

Result of fitting procedure was estimated as

.. code:: python

    [
        {
            'value': 0.19999999738789395,
            'param': 'eccentricity',

        },
        {
            'value': 1.9999999157860147,
            'param': 'p__mass',

        },
        {
            'value': 32999.99919352919,
            'param': 'gamma',

        },
        {
            'value': 90,
            'param': 'inclination',
        },
        {
            'value': 1.0,
            'param': 's__mass',
        },
        {
            'value': 0.0,
            'param': 'argument_of_periastron',
        },
        "r_squared": 0.9999999999999997,
    ]

.. image:: ./docs/source/_static/readme/rv_fit.png
  :width: 70%
  :alt: rv_fit.png
  :align: center


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
