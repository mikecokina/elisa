import os
import sys
import tempfile

if 'astropy.units' not in sys.modules:
    from astropy import physical_constants
    physical_constants.set('astropyconst20')

# create default elisa confing file for unittests
CONFIG_FILE = os.path.join(tempfile.gettempdir(), "elisa.ini")

os.environ["ELISA_CONFIG"] = CONFIG_FILE
with open(CONFIG_FILE, "w") as f:
    f.write("")
