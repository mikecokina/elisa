Eclipsing binaries Learning Interactive System
==============================================

ELISa
-----

**ELISa** is a python package dedicated to light curves modelling of close eclipsing binaries including surface features
Current capabilities include:

    - BinarySystem - class for modelling surfaces of detached, semi-detached and over-contact binaries
    - Observer - class for generating light curves (and in future other observables)
    - Spots - class for generating stellar spot with given longitude, latitude, radius and temperature factor
    - Pulsations - class for modelling low amplitude pulsations based on spherical harmonics

**ELISa** is currently under development. Following features are currently under development:

    - SingleSystem - class for modelling surfaces of single star light curves with full implementation of spots and
      pulsations

We also plan to implement following features:

    - addition of radial velocity curves to Observer class with Rossiter–McLaughlin effect
    - LC and RV fitting using various methods
    - addition of synthetic spectral line modelling

Requirements
------------

**ELISa** is a python package which requires ``python v3.6+`` and has following dependencies::

    cycler==0.10.0
    matplotlib==2.1.0
    numpy==1.13.3
    pandas==0.23.0
    pyparsing==2.2.0
    pytest==3.2.3
    python-dateutil==2.6.1
    pytz==2017.2
    py==1.4.34
    astropy==2.0.2
    scipy==1.0.0
    six==1.11.0
    pypex==0.1.0

and potentially also **python-tk** package or equivalent for matplotlib package to display the figures correctly.

Although python distribution and package versions are specified precisely, that does not mean that the package will not
work with higher versions, only that it was not tested with higher versions of packages. However we
highly recommend to stick with python distribution and package versions listed above.

Install
-------

In case of ``ELISa`` the easiest and the safest way to install is to create python virtual
environment and install all requirements into it. Here is a simple guide, how to od it. Details of installation differ
in dependence on the selected operating system.

Ubuntu [or similar]
~~~~~~~~~~~~~~~~~~~

First, you have to install Python 3.6 or higher. In newest stable version ``Ubuntu 18.04`` there is already preinstalled
python `3.6.x`. In older versions, you will have to add repository and install it manually.

Install ``pip3`` python package manager if is not already installed on your system, usually by execution of command::

    apt install -y python3-pip

Install virtual environment by command::

    pip3 install virtualenv


To create virtual environment, create directory where python virtual environment will be stored,
e.g. ``/<any>/<path>/elisa/venv``
and run following command::

    virtualenv /<any>/<path>/elisa/venv --python=python3.6

After few moments you virtual environment is created and ready for use. In terminal window, activate virtual
environment::

    . /<any>/<path>/elisa/venv/bin/activate

When virtual environment is activated, inside `<any>/<path>/elisa` directory clone ``ELISa`` from github repository
with::

    git clone https://github.com/mikecokina/elisa.git ???

This will create subdirectory ``/<any>/<path>/elisa/elisa``. Required packages can be installed with::

    pip install -r elisa/requirements.txt

You will probably also need to install::

    apt install -y python3-tk


Windows
~~~~~~~

To install python in windows, download ``python 3.6.x`` installation package from official python web site.
Installation package will create all necessary dependencies with exception of virtual environment.
Install virtual environment by execution of following command in command line::

    pip3 install virtualenv

Make sure a proper version of  python and pip is used. When done, create directory where virtual environment will be
stored and run::

    virtualenv /<any>/<path>/elisa --python=python3.6

Now, when virtual environment is prepared, run::

    . /<any>/<path>/elisa/Scripts/activate

And finally install ``ELISa``::

    placeholder

Usage
-------
For in depth tutorials, see directory ``elisa/jupyter_tutorials``