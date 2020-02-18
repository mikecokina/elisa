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
