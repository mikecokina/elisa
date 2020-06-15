Limbdarkening models
====================

    VanHamme 2016 **url:** https://mega.nz/#!PAs1xCqZ!JIMLIdEjK48OEu1kkPPm_uw1GjxtbD0gUP5loKYKpac

    Claret 2017 (TESS) **url:** https://mega.nz/#!PAs1xCqZ!JIMLIdEjK48OEu1kkPPm_uw1GjxtbD0gUP5loKYKpac

    Contain limb darkening models for given limbdarkening models, passbands and metalicities
    Extract content of ``limbdarkening.tar`` archive to any directory and setup ``support.van_hamme_ld_tables``
    in ``elisa_config.ini``.

Metallicity sign
~~~~~~~~~~~~~~~~

    Metallicity is encoded in filename in following way. Sign of number is denoted by alphabetical representation.
    It is mean, minus is denoted as ``m`` and plus as ``p``. Digit value of metallicity is encoded
    in two digits representation, where in between any of such two digits, a decimal point has to be added.


    An example::

        m05 = -0.5

Filename naming convention
~~~~~~~~~~~~~~~~~~~~~~~~~~

    ``{model}.{passband}.{sign}.{metallicity}.csv``, e.g. ``lin.Generic.Bessell.B.m01.csv``

File content
~~~~~~~~~~~~

    Each file contain a header and values. In general, header composition of::

        temperature, gravity, coefficients, error

    Temperature is stored in units of Kelvin

    Gravity is stored in logarithmic form

List of models included
-----------------------

::

    linear or cosine
    logarithmic
    square_root

List of passbands included
--------------------------
    
::

    bolometric,
    Generic.Bessell.U
    Generic.Bessell.B
    Generic.Bessell.V
    Generic.Bessell.R
    Generic.Bessell.I
    Kepler
    GaiaDR2
    SLOAN.SDSS.u
    SLOAN.SDSS.g
    SLOAN.SDSS.r
    SLOAN.SDSS.i
    SLOAN.SDSS.z
    Generic.Stromgren.u
    Generic.Stromgren.b
    Generic.Stromgren.v
    Generic.Stromgren.y