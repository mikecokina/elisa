Build API docs
~~~~~~~~~~~~~~

1. Clone elisa
2. Change directory to ``elisa/docs``
3. Install requirements
4. Use terminal and change directory to docs
5. Create directory `build`
6. Run command::

    sphinx-build -W -b html -c ./source -d ./build/doctrees ./source ./build

:warning: make sure that path separator is valid for your OS


Generate apidoc - developers only
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. ``cd /path/to/docs``
2. activate virtual environment
3. remove old api docs folder
4. ``sphinx-apidoc  ..\src\elisa  -o .\source\api -f``
5. make sure ``source/api/elisa.rst`` contain at the end of file

::

    .. automodule:: elisa
       :members:
       :undoc-members:
       :show-inheritance:
       :noindex:

6. build (``sphinx-build -W -b html -c ./source -d ./build/doctrees ./source ./build -v``)
