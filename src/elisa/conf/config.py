import json
import logging
import os
import warnings
import numpy as np

from configparser import ConfigParser
from logging import config as log_conf
from os.path import dirname, isdir, pardir
from .. schema_registry import registry


c_parse = ConfigParser()

env_variable_config = os.environ.get('ELISA_CONFIG', '')
venv_config = os.path.join(os.environ.get('VIRTUAL_ENV', ''), 'conf', 'elisa_conf.ini')
default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini")

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

# schema registry
SCHEMA_REGISTRY = registry.Registry()

# basic app configuration
CONFIG_FILE = config_file
LOG_CONFIG = os.path.join(dirname(os.path.abspath(__file__)), 'logging.json')
SUPPRESS_WARNINGS = False
SUPPRESS_LOGGER = None
HOME = os.path.expanduser(os.path.join("~", '.elisa'))

# physics
REFLECTION_EFFECT = True
REFLECTION_EFFECT_ITERATIONS = 2
LIMB_DARKENING_LAW = 'cosine'
DEFAULT_TEMPERATURE_PERTURBATION_PHASE_SHIFT = np.pi / 2.0
SURFACE_DISPLACEMENT_TOL = 1e-2
RV_METHOD = 'point_mass'
RV_LAMBDA_INTERVAL = (5500, 5600)

# computational
MAX_DISCRETIZATION_FACTOR = 20
MIN_DISCRETIZATION_FACTOR = 3
NUMBER_OF_THREADS = 1
NUMBER_OF_PROCESSES = -1  # int(os.cpu_count())
NUMBER_OF_MCMC_PROCESSES = -1
POINTS_ON_ECC_ORBIT = 118
MAX_RELATIVE_D_R_POINT = 3e-3
MAX_SUPPLEMENTAR_D_DISTANCE = 1e-1
MAX_SPOT_D_LONGITUDE = np.pi / 180.0  # in radians
MAX_SOLVER_ITERS = 100
MAX_CURVE_DATA_POINTS = 300

TIMER = 0.0

# support data
PASSBAND_TABLES = os.path.join(dirname(os.path.abspath(__file__)), pardir, "passband")
LD_TABLES = os.path.join(HOME, "limbdarkening", "ld")
CK04_ATM_TABLES = os.path.join(HOME, "atmosphere", "ck04")
K93_ATM_TABLES = os.path.join(HOME, "atmosphere", "k93")
ATM_ATLAS = "ck04"
ATLAS_TO_BASE_DIR = {
    "castelli": CK04_ATM_TABLES,
    "castelli-kurucz": CK04_ATM_TABLES,
    "ck": CK04_ATM_TABLES,
    "ck04": CK04_ATM_TABLES,
    "kurucz": K93_ATM_TABLES,
    "k": K93_ATM_TABLES,
    "k93": K93_ATM_TABLES
}


def _create_home():
    os.makedirs(HOME, exist_ok=True)


def _update_atlas_to_base_dir():
    ATLAS_TO_BASE_DIR.update({
        "castelli": CK04_ATM_TABLES,
        "castelli-kurucz": CK04_ATM_TABLES,
        "ck": CK04_ATM_TABLES,
        "ck04": CK04_ATM_TABLES,
        "kurucz": K93_ATM_TABLES,
        "k": K93_ATM_TABLES,
        "k93": K93_ATM_TABLES
    })


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
            "    - add conf/elisa_conf.ini under your virtualenv root \n")
        warnings.warn(msg, Warning)
        return

    c_parse.read(conf_path)
    update_config()
    _create_home()


def update_config():
    if c_parse.has_section('general'):
        global SUPPRESS_WARNINGS
        SUPPRESS_WARNINGS = c_parse.getboolean('general', 'suppress_warnings', fallback=SUPPRESS_WARNINGS)

        global LOG_CONFIG
        LOG_CONFIG = c_parse.get('general', 'log_config', fallback=LOG_CONFIG)
        if not os.path.isfile(LOG_CONFIG):
            if not SUPPRESS_WARNINGS:
                warnings.warn(f"log config `{LOG_CONFIG}` doesn't exist, rollback to default")
            LOG_CONFIG = os.path.join(dirname(os.path.abspath(__file__)), 'logging.json')

        global SUPPRESS_LOGGER
        SUPPRESS_LOGGER = c_parse.getboolean('general', 'suppress_logger', fallback=SUPPRESS_LOGGER)

        global HOME
        HOME = c_parse.getboolean('general', 'home', fallback=HOME)
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

        global DEFAULT_TEMPERATURE_PERTURBATION_PHASE_SHIFT
        DEFAULT_TEMPERATURE_PERTURBATION_PHASE_SHIFT = \
            c_parse.getfloat('physics', 'default_temperature_perturbation_phase_shift',
                             fallback=DEFAULT_TEMPERATURE_PERTURBATION_PHASE_SHIFT)

        global SURFACE_DISPLACEMENT_TOL
        SURFACE_DISPLACEMENT_TOL = \
            c_parse.getfloat('physics', 'surface_displacement_tol', fallback=SURFACE_DISPLACEMENT_TOL)

        global RV_METHOD
        RV_METHOD = c_parse.getfloat('physics', 'rv_method', fallback=RV_METHOD)

        global RV_LAMBDA_INTERVAL
        RV_LAMBDA_INTERVAL = c_parse.getfloat('physics', 'rv_lambda_interval', fallback=RV_LAMBDA_INTERVAL)

    # ******************************************************************************************************************

    if c_parse.has_section('computational'):
        global MAX_DISCRETIZATION_FACTOR
        MAX_DISCRETIZATION_FACTOR = c_parse.getfloat('computational', 'max_discretization_factor',
                                                     fallback=MAX_DISCRETIZATION_FACTOR)
        global MIN_DISCRETIZATION_FACTOR
        MIN_DISCRETIZATION_FACTOR = c_parse.getfloat('computational', 'min_discretization_factor',
                                                     fallback=MIN_DISCRETIZATION_FACTOR)

        global NUMBER_OF_THREADS
        NUMBER_OF_THREADS = c_parse.getint('computational', 'number_of_threads', fallback=NUMBER_OF_THREADS)
        if NUMBER_OF_THREADS <= 0:
            raise ValueError("Invalid value for `number_of_threads`, allowed >= 1")

        global NUMBER_OF_PROCESSES
        NUMBER_OF_PROCESSES = c_parse.getint('computational', 'number_of_processes', fallback=NUMBER_OF_PROCESSES)
        if NUMBER_OF_PROCESSES > os.cpu_count():
            warnings.warn("argument number_of_processes is too big, fallback to number of machine cores")
            NUMBER_OF_PROCESSES = int(os.cpu_count())

        global NUMBER_OF_MCMC_PROCESSES
        NUMBER_OF_MCMC_PROCESSES = c_parse.getint('computational', 'number_of_mcmc_processes',
                                                  fallback=NUMBER_OF_MCMC_PROCESSES)
        if NUMBER_OF_MCMC_PROCESSES > os.cpu_count():
            warnings.warn("argument number_of_mcmc_processes is too big, fallback to number of machine cores")
            NUMBER_OF_MCMC_PROCESSES = int(os.cpu_count())

        global POINTS_ON_ECC_ORBIT
        POINTS_ON_ECC_ORBIT = c_parse.getint('computational', 'points_on_ecc_orbit', fallback=POINTS_ON_ECC_ORBIT)

        global MAX_RELATIVE_D_R_POINT
        MAX_RELATIVE_D_R_POINT = c_parse.getfloat('computational', 'max_relative_d_r_point',
                                                  fallback=MAX_RELATIVE_D_R_POINT)

        global MAX_SUPPLEMENTAR_D_DISTANCE
        MAX_SUPPLEMENTAR_D_DISTANCE = c_parse.getfloat('computational', 'max_supplementar_d_distance',
                                                       fallback=MAX_SUPPLEMENTAR_D_DISTANCE)

        global MAX_SPOT_D_LONGITUDE
        MAX_SPOT_D_LONGITUDE = c_parse.getfloat('computational', 'max_spot_d_longitude', fallback=MAX_SPOT_D_LONGITUDE)

        global MAX_SOLVER_ITERS
        MAX_SOLVER_ITERS = c_parse.getfloat('computational', 'max_solver_iters', fallback=MAX_SOLVER_ITERS)

        global MAX_CURVE_DATA_POINTS
        MAX_CURVE_DATA_POINTS = c_parse.getfloat('computational', 'max_curve_datapoints',
                                                 fallback=MAX_CURVE_DATA_POINTS)
    # ******************************************************************************************************************

    if c_parse.has_section('support'):
        global LD_TABLES

        LD_TABLES = c_parse.get('support', 'ld_tables', fallback=LD_TABLES)

        if not isdir(LD_TABLES) and not SUPPRESS_WARNINGS:
            warnings.warn(f"path {LD_TABLES} to limb darkening tables doesn't exists\n"
                          f"Specifiy it in elisa_conf.ini file")

        global CK04_ATM_TABLES
        CK04_ATM_TABLES = c_parse.get('support', 'castelli_kurucz_04_atm_tables', fallback=CK04_ATM_TABLES)

        if not os.path.isdir(CK04_ATM_TABLES) and not SUPPRESS_WARNINGS:
            warnings.warn(f"path {CK04_ATM_TABLES}\n"
                          f"to castelli-kurucz 2004 atmosphere atlas doesn't exists\n"
                          f"Specifiy it in elisa_conf.ini file")

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
    'Generic.Stromgren.y',
    'Kepler',
    'GaiaDR2',
    'TESS',
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


ATLAS_TO_ATM_FILE_PREFIX = {
    "castelli": "ck",
    "castelli-kurucz": "ck",
    "ck": "ck",
    "ck04": "ck",
    "kurucz": "k",
    "k": "k",
    "k93": "k"
}

ATM_DOMAIN_QUANTITY_TO_VARIABLE_SUFFIX = {
    "temperature": "TEMPERATURE_LIST_ATM",
    "gravity": "GRAVITY_LIST_ATM",
    "metallicity": "METALLICITY_LIST_ATM"
}

DATETIME_MASK = '%Y-%m-%dT%H.%M.%S'
DATE_MASK = '%Y-%m-%d'

DATASET_MANDATORY_KWARGS = ['x_data', 'y_data', 'x_unit', 'y_unit']
DATASET_OPTIONAL_KWARGS = ['y_err']

DELIM_WHITESPACE = r'\s+|\t+|\s+\t+|\t+\s+'

read_and_update_config()
_update_atlas_to_base_dir()
_create_home()


def config_serializer():
    return {
        "CONFIG_FILE": CONFIG_FILE,
        "LOG_CONFIG": LOG_CONFIG,
        "SUPPRESS_WARNINGS": SUPPRESS_WARNINGS,
        "SUPPRESS_LOGGER": SUPPRESS_LOGGER,
        "HOME": HOME,
        "REFLECTION_EFFECT": REFLECTION_EFFECT,
        "REFLECTION_EFFECT_ITERATIONS": REFLECTION_EFFECT_ITERATIONS,
        "LIMB_DARKENING_LAW": LIMB_DARKENING_LAW,
        "DEFAULT_TEMPERATURE_PERTURBATION_PHASE_SHIFT": DEFAULT_TEMPERATURE_PERTURBATION_PHASE_SHIFT,
        "SURFACE_DISPLACEMENT_TOL": SURFACE_DISPLACEMENT_TOL,
        "RV_METHOD": RV_METHOD,
        "RV_LAMBDA_INTERVAL": RV_LAMBDA_INTERVAL,
        "MAX_DISCRETIZATION_FACTOR": MAX_DISCRETIZATION_FACTOR,
        "MIN_DISCRETIZATION_FACTOR": MIN_DISCRETIZATION_FACTOR,
        "NUMBER_OF_THREADS": NUMBER_OF_THREADS,
        "NUMBER_OF_PROCESSES": NUMBER_OF_PROCESSES,
        "NUMBER_OF_MCMC_PROCESSES": NUMBER_OF_MCMC_PROCESSES,
        "POINTS_ON_ECC_ORBIT": POINTS_ON_ECC_ORBIT,
        "MAX_RELATIVE_D_R_POINT": MAX_RELATIVE_D_R_POINT,
        "MAX_SUPPLEMENTAR_D_DISTANCE": MAX_SUPPLEMENTAR_D_DISTANCE,
        "MAX_SPOT_D_LONGITUDE": MAX_SPOT_D_LONGITUDE,
        "MAX_SOLVER_ITERS": MAX_SOLVER_ITERS,
        "MAX_CURVE_DATA_POINTS": MAX_CURVE_DATA_POINTS,
        "TIMER": TIMER,
        "PASSBAND_TABLES": PASSBAND_TABLES,
        "LD_TABLES": LD_TABLES,
        "CK04_ATM_TABLES": CK04_ATM_TABLES,
        "K93_ATM_TABLES": K93_ATM_TABLES,
        "ATM_ATLAS": ATM_ATLAS
    }


configuration_serializer = config_serializer
