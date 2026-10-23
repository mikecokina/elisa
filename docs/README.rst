Build API docs on Linux
~~~~~~~~~~~~~~~~~~~~~~~

Run the commands below in a Bash terminal. Replace ``/path/to/elisa`` with
the path to your local clone, for example ``/home/mike/Work/projects/elisa``.
Linux paths use forward slashes (``/``) and are case-sensitive.

Python 3.12 or newer is required by the pinned Sphinx version. Check your
interpreter with ``python3 --version``. On Debian/Ubuntu, install the matching
``python3-venv`` package if creating a virtual environment fails because
``ensurepip`` is unavailable.

Sphinx uses ``docs/elisa-docs.ini`` when importing ELISa for API documentation,
so builds do not depend on your personal ELISa configuration or launch the
first-run setup wizard.

1. Change to the repository root and create a virtual environment::

       cd /path/to/elisa
       python3 -m venv .venv
       source .venv/bin/activate

2. Install the documentation dependencies::

       python -m pip install -r docs/requirements.txt

3. Change to the documentation directory and build the HTML documentation::

       cd docs
       mkdir -p build
       sphinx-build -W -b html -c ./source -d ./build/doctrees ./source ./build

4. Open the generated documentation in your default browser::

       xdg-open build/index.html

For subsequent builds, activate the existing virtual environment and run the
build command from ``docs``::

    cd /path/to/elisa
    source .venv/bin/activate
    cd docs
    sphinx-build -W -b html -c ./source -d ./build/doctrees ./source ./build


Generate apidoc - developers only
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Activate the virtual environment and change to ``docs``::

       cd /path/to/elisa
       source .venv/bin/activate
       cd docs

2. Regenerate the API documentation. The ``-f`` option overwrites existing
   generated files::

       sphinx-apidoc ../src/elisa -o ./source/api -f

3. Build the documentation with verbose output::

       sphinx-build -W -b html -c ./source -d ./build/doctrees ./source ./build -v

