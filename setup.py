"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject

Pre-release steps
-----------------

1.  make sure all newly added data files are listed in `setup.py` in variable
    `package_data`as well as in `MANIFEST.in` file

2.  make sure version information in following destinations is up-to-date:

    - `README.md` in yellow version badge https://img.shields.io/badge/version-<VERSION>-yellow.svg
    - in `src/elisa/__init__.py`, variable `__version__`

3.  make sure that `CHANGELOG.md` is up-to-date; content as well as release date and valid version

4.  make sure all newly added dependencies are listed in `requirements.txt` as well as in
    `setup.py` in variable `install_requires`

5.  make sure a latest docstsring documentation is generated and there is no error during Sphinx HTML build
    (for more comprehensive information take a look into `docs/README.rst`)

6.  make sure setup.cfg contains all supported Python versions

7.  commit changes

8.  create release branch with name `release/<VERSION>` and push changes

9.  make sure all unittests are running

10. create a tag::

    >> git tag -a v<VERSION> -m "version <VERSION>"
    >> git push origin --tags

11. build package with command::

    >> python setup.py sdist bdist_wheel

12. configure pypi repositories if necessary (use following configuration template with valid credentials)::

        # ~/.pypirc
        [distutils] # this tells distutils what package indexes you can push to
        index-servers =
            pypi

        [pypi]
        repository = https://upload.pypi.org/legacy/
        username = <username>
        password = <password>


13. release packages with following command (install twine if necessary with `pip install twine`)::

    >> twine upload dist/* -r pypi
"""

from __future__ import annotations

# Always prefer setuptools over distutils
# Use pathlib for path operations and Path.open for file IO
from pathlib import Path

from setuptools import find_packages, setup

here = Path(__file__).parent


def read_version() -> str:
    """Read project version from `src/elisa/__init__.py`.

    This helper returns the value of the ``__version__`` string defined in
    the package's ``__init__.py``.  It uses :class:`pathlib.Path` for safe
    cross-platform path handling and :meth:`Path.open` for file access.

    :return: Version string, e.g. ``"0.1.0"``.
    :rtype: str
    :raises RuntimeError: When the ``__version__`` variable cannot be found.
    """
    version_file = here / "src" / "elisa" / "__init__.py"
    with version_file.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
    msg = "Unable to find __version__ string."
    raise RuntimeError(msg)


# Get the long description from the README file

long_description = (
    "ELISA is a scientific software package for modeling eclipsing binary star systems. "
    "It provides tools for computing light curves, radial velocity curves, and related "
    "observables based on physically consistent stellar and orbital models. "
    "The library is designed with numerical accuracy and extensibility in mind, "
    "making it suitable for research, experimentation, and advanced analysis in "
    "stellar astrophysics.\n\n"
    "ELISA supports configurable system geometries, surface discretization, "
    "radiative modeling, and observational simulation pipelines. "
    "It enables detailed investigation of photometric and spectroscopic behavior "
    "of interacting and detached binary systems.\n\n"
    "The project is built on top of NumPy and SciPy and follows a modular architecture "
    "that allows users to extend physical models, observational methods, and numerical "
    "strategies. It is intended for developers and researchers working in computational "
    "astronomy and binary star modeling."
)

setup(
    name="elisa",
    src_root="src",
    version=read_version(),

    python_requires=">=3.10",

    description="Eclipsing Binary Modeling Software",
    long_description=long_description,

    # The project's main homepage.
    url="https://github.com/mikecokina/elisa",

    # Author details
    author="Michal Cokina, Miroslav Fedurco",
    author_email="mikecokina@gmail.com, mirofedurco@gmail.com",

    # Choose your license
    license="GPLv3",

    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 4 - Beta",

        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Astronomy",

        # Pick your license as you wish (should match "license" above)
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        # 'Programming Language :: Python :: 2',
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    ],

    # What does your project relate to?
    keywords="eclipsing binaries astronomy analysis analytics physic",

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    # packages=find_packages(where='src', exclude=["single_system"]),
    packages=find_packages(where="src"),

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html

    install_requires=[
        # Python 3.10 compatible floors
        'astropy>=6.1.7; python_version<"3.11"',
        'numpy>=2.2.0; python_version<"3.11"',
        'pandas>=2.2.3; python_version<"3.11"',
        'scipy>=1.15.0; python_version<"3.11"',

        # Python 3.11+ floors (aligns with newer environments)
        'astropy>=7.2.0; python_version>="3.11"',
        'numpy>=2.4.2; python_version>="3.11"',
        'pandas>=3.0.1; python_version>="3.11"',
        'scipy>=1.17.1; python_version>="3.11"',

        # Common deps
        "corner>=2.2.1",
        "emcee>=3.0.1",
        "jsonschema>=3.2.0",
        "matplotlib>=3.3.2",
        "packaging>=20.0",
        "python-dateutil>=2.6.1",
        "tqdm>=4.43.0",
        "parameterized>=0.7.4",
        "numba>=0.51.2",
        "requests>=2.26.0",
    ],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        "dev": [
            "setuptools>=82.0.0",
            "wheel>=0.46.3",
            "build",
            "twine",
            "ruff",
        ],
        "test": [
            "coverage",
            "pytest>=9.0.2",
            "parameterized>=0.7.4",
        ],
        "ui": [
            "gradio~=6.10.0",
            # UI runtime dependencies used by the Gradio desktop app
            # pandas and matplotlib are already listed in install_requires
            'pandas>=2.2.3; python_version < "3.11"',
            'pandas>=3.0.1; python_version >= "3.11"',
            "matplotlib>=3.3.2",
            "pillow>=9.0.0",
        ],
    },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={
        "elisa": [
            "passband/**",
            "conf/**",
            "schema_registry/**",
            "data/**",
        ],
    },

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    data_files=[],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        "console_scripts": [
        ],
    },
)
