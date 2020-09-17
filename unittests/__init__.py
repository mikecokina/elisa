import sys

if 'astropy.units' not in sys.modules:
    from astropy import physical_constants
    physical_constants.set('astropyconst20')
