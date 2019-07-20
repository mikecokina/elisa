import json
import logging
import os
import warnings

from configparser import ConfigParser
from logging import config as log_conf
from os.path import dirname, isdir


def level_up(path, n=0):
    for i in range(n):
        path = dirname(path)
    return path


c_parse = ConfigParser()

env_variable_config = os.environ.get('ELISA_CONFIG', '')
venv_config = os.path.join(os.environ.get('VIRTUAL_ENV', ''), 'conf', 'elisa_conf.ini')
default_config = os.path.join(os.path.dirname(__file__), "elisa_conf.ini")

# read configuration file
if os.path.isfile(env_variable_config):
    config_file = env_variable_config
elif os.path.isfile(venv_config):
    config_file = venv_config
elif os.path.isfile(default_config):
    config_file = default_config
else:
    raise LookupError("Couldn't resolve configuration file. To define it \n "
                      "  - Set the environment variable ELISA_CONFIG, or \n "
                      "  - Add conf/elisa_conf.ini under your virtualenv root \n ")

# basic app configuration
CONFIG_FILE = config_file
LOG_CONFIG = os.path.join(dirname(__file__), 'logging.json')
SUPPRESS_WARNINGS = False

# physics
REFLECTION_EFFECT = True
REFLECTION_EFFECT_ITERATIONS = 2
LIMB_DARKENING_LAW = 'cosine'

# computational
DISCRETIZATION_FACTOR = 5
MAX_DISCRETIZATION_FACTOR = 20
NUMBER_OF_THREADS = int(os.cpu_count())
D_RADIUS = 0.001
POINTS_ON_ECC_ORBIT = 99999
MAX_D_DISTANCE = 0.0

# support data
PASSBAND_TABLES = os.path.join(level_up(__file__, 3), "passband")
VAN_HAMME_LD_TABLES = os.path.join(level_up(__file__, 3), "limbdarkening", "vh")
CK04_ATM_TABLES = os.path.join(level_up(__file__, 3), "atmosphere", "ck04")
K93_ATM_TABLES = os.path.join(level_up(__file__, 3), "atmosphere", "k93")
ATM_ATLAS = "ck04"


def set_up_logging():
    if os.path.isfile(LOG_CONFIG):
        with open(LOG_CONFIG) as f:
            conf_dict = json.loads(f.read())
        log_conf.dictConfig(conf_dict)
    else:
        logging.basicConfig(level=logging.INFO)


def read_and_update_config(conf_path=None):
    if not conf_path:
        conf_path = CONFIG_FILE

    if not os.path.isfile(conf_path):
        msg = (
            "Couldn't find configuration file. Using default settings.\n"
            "   To customize configuration using file either\n"
            "    - specify config with environment variable ELISA_CONFIG\n"
            "  -   add conf/elisa_conf.ini under your virtualenv root \n")
        warnings.warn(msg, Warning)
        return

    c_parse.read(conf_path)
    update_config()


def update_config():
    if c_parse.has_section('general'):
        global SUPPRESS_WARNINGS
        SUPPRESS_WARNINGS = c_parse.getboolean('general', 'suppress_warnings', fallback=SUPPRESS_WARNINGS)

        global LOG_CONFIG
        LOG_CONFIG = c_parse.get('general', 'log_config') if c_parse.get('general', 'log_config') else LOG_CONFIG
    # ******************************************************************************************************************

    if c_parse.has_section('physics'):
        global REFLECTION_EFFECT
        REFLECTION_EFFECT = c_parse.getboolean('physics', 'reflection_effect', fallback=REFLECTION_EFFECT)

        global REFLECTION_EFFECT_ITERATIONS
        REFLECTION_EFFECT_ITERATIONS = c_parse.getint('physics', 'reflection_effect_iterations',
                                                      fallback=REFLECTION_EFFECT_ITERATIONS)

        global LIMB_DARKENING_LAW
        LIMB_DARKENING_LAW = c_parse.get('physics', 'limb_darkening_law', fallback=LIMB_DARKENING_LAW)
        if LIMB_DARKENING_LAW not in ['linear', 'cosine', 'logarithmic', 'square_root']:
            raise ValueError(f'{LIMB_DARKENING_LAW} is not valid name of limb darkening law. '
                             f'Available limb darkening laws are: `linear` or `cosine`, `logarithmic`, `square_root`')
    # ******************************************************************************************************************

    if c_parse.has_section('computational'):
        global DISCRETIZATION_FACTOR
        DISCRETIZATION_FACTOR = c_parse.getfloat('computational', 'discretization_factor',
                                                 fallback=DISCRETIZATION_FACTOR)

        global MAX_DISCRETIZATION_FACTOR
        MAX_DISCRETIZATION_FACTOR = c_parse.getfloat('computational', 'max_discretization_factor',
                                                     fallback=MAX_DISCRETIZATION_FACTOR)

        global NUMBER_OF_THREADS
        NUMBER_OF_THREADS = c_parse.getint('computational', 'number_of_threads', fallback=NUMBER_OF_THREADS)

        global D_RADIUS
        D_RADIUS = c_parse.getfloat('computational', 'd_radius', fallback=D_RADIUS)

        global POINTS_ON_ECC_ORBIT
        POINTS_ON_ECC_ORBIT = c_parse.getint('computational', 'points_on_ecc_orbit', fallback=POINTS_ON_ECC_ORBIT)

        global MAX_D_DISTANCE
        MAX_D_DISTANCE = c_parse.getfloat('computational', 'max_d_distance', fallback=MAX_D_DISTANCE)
    # ******************************************************************************************************************

    if c_parse.has_section('support'):
        global VAN_HAMME_LD_TABLES
        VAN_HAMME_LD_TABLES = c_parse.get('support', 'van_hamme_ld_tables', fallback=VAN_HAMME_LD_TABLES)

        if not isdir(VAN_HAMME_LD_TABLES) and not SUPPRESS_WARNINGS:
            warnings.warn(f"path {VAN_HAMME_LD_TABLES} to van hamme ld tables doesn't exists\n"
                          f"Specifiy it in elisa_conf.ini file")

        global CK04_ATM_TABLES
        CK04_ATM_TABLES = c_parse.get('support', 'castelli_kurucz_04_atm_tables', fallback=CK04_ATM_TABLES)

        if not os.path.isdir(CK04_ATM_TABLES) and not SUPPRESS_WARNINGS:
            warnings.warn("path {}\n"
                          "to castelli-kurucz 2004 atmosphere atlas doesn't exists\n"
                          "Specifiy it in elisa_conf.ini file".format(CK04_ATM_TABLES))

        global K93_ATM_TABLES
        K93_ATM_TABLES = c_parse.get('support', 'kurucz_93_atm_tables', fallback=K93_ATM_TABLES)

        if not os.path.isdir(K93_ATM_TABLES):
            warnings.warn("path {}\n"
                          "to kurucz 1993 atmosphere atlas doesn't exists\n"
                          "Specifiy it in elisa_conf.ini file".format(K93_ATM_TABLES))

        global ATM_ATLAS
        ATM_ATLAS = c_parse.get('support', 'atlas', fallback=ATM_ATLAS)

        global PASSBAND_TABLES
        PASSBAND_TABLES = c_parse.get('support', 'passband_tables', fallback=PASSBAND_TABLES)

        if not isdir(PASSBAND_TABLES) and not SUPPRESS_WARNINGS:
            warnings.warn(f"path {PASSBAND_TABLES} to passband tables doesn't exists\n"
                          f"Specifiy it in elisa_conf.ini file")
    # ******************************************************************************************************************


# PASSBAND RELATED *****************************************************************************************************

PASSBANDS = [
    'bolometric',
    'Generic.Bessell.U',
    'Generic.Bessell.B',
    'Generic.Bessell.V',
    'Generic.Bessell.R',
    'Generic.Bessell.I',
    'SLOAN.SDSS.u',
    'SLOAN.SDSS.g',
    'SLOAN.SDSS.r',
    'SLOAN.SDSS.i',
    'SLOAN.SDSS.z',
    'Generic.Stromgren.u',
    'Generic.Stromgren.v',
    'Generic.Stromgren.b',
    'Generic.Stromgren.y'
]

PASSBAND_DATAFRAME_THROUGHPUT = "throughput"
PASSBAND_DATAFRAME_WAVE = "wavelength"
# **********************************************************************************************************************

# ATM RELATED **********************************************************************************************************
ATM_MODEL_DATAFRAME_FLUX = "flux"
ATM_MODEL_DATAFRAME_WAVE = "wave"

ATM_MODEL_DATAFARME_DTYPES = {
    ATM_MODEL_DATAFRAME_FLUX: float,
    ATM_MODEL_DATAFRAME_WAVE: float
}
# **********************************************************************************************************************


# LIMB DARKENING RELATED ***********************************************************************************************
LD_LAW_TO_FILE_PREFIX = {
    "linear": "lin",
    "cosine": "lin",
    "logarithmic": "log",
    "square_root": "sqrt",
}

LD_LAW_CFS_COLUMNS = {
    "linear": ["xlin"],
    "cosine": ["xlin"],
    "logarithmic": ["xlog", "ylog"],
    "square_root": ["xsqrt", "ysqrt"],
}


BASIC_COLS = ["temperature", "gravity"]
LD_LAW_COLS_ORDER = {
    "linear": BASIC_COLS + LD_LAW_CFS_COLUMNS["linear"],
    "cosine": BASIC_COLS + LD_LAW_CFS_COLUMNS["cosine"],
    "logarithmic": BASIC_COLS + LD_LAW_CFS_COLUMNS["logarithmic"],
    "square_root": BASIC_COLS + LD_LAW_CFS_COLUMNS["square_root"]
}

LD_DOMAIN_COLS = ["temperature", "gravity"]
# **********************************************************************************************************************


BINARY_COUNTERPARTS = {"primary": "secondary", "secondary": "primary"}


read_and_update_config()
