Atmospheres
===========

Contain model of stellar atmospheres

Castelli-Kurucz 2004
~~~~~~~~~~~~~~~~~~~~

    **url:** https://mega.nz/#!rQc0FQKL!IiygAjuAA-Ea3u2lRHJNpkiY3y-RqHohkUC9g2FOGiA

    Extract content of ``ck04.tar`` archive to any directory and setup ``support.castelli_kurucz_04_atm_tables``
    in ``elisa_config.ini``.

    Directory contain structure like::

        ck<metallicity>
                        `-ck<metallicity>_<temperature>_g<logg>.csv

Metallicity form and sign
`````````````````````````

Metallicity is encoded in filename in following way. Sign of number is denoted by alphabetical representation.
It is mean, minus is denoted as ``m`` and plus as ``p``. Digit value of metallicity is encoded
in two digits representation, where in between any of such two digits, a decimal point has to be added.


An example::

    m05 = -0.5

Gravity form
````````````

Gravity is encoded in filename as two digits representation of ``logg`` without any prefix for sign as in case
of metallicity (all values are positive).


An example::

    g35 = 3.5

Files content
`````````````

Each ``.csv`` file contain two columns like::

	flux,wave
	0.0,90.9
	0.0,93.5
	0.0,96.1
	0.0,97.7
	0.0,99.6
	0.0,102.0
	0.0,103.8
	0.0,105.6
	0.0,107.7
	0.0,110.4
	0.0,114.0
	0.0,117.8
	0.0,121.3

Flux is stored in ``flam`` units and ``wave`` in Angstromes.


Kurucz 1993
~~~~~~~~~~~

    **url:** https://mega.nz/#!vQFi2YbD!0k8uZ_44HJGqlZ9XkDk2JuOTrgZ8JkY_4mu9ygDcUA0
