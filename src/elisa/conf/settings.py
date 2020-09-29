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


class _Const(object):
    # PASSBAND RELATED *************************************************************************************************
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
    # ******************************************************************************************************************

    # ATM RELATED ******************************************************************************************************
    ATM_MODEL_DATAFRAME_FLUX = "flux"
    ATM_MODEL_DATAFRAME_WAVE = "wave"

    ATM_MODEL_DATAFARME_DTYPES = {
        ATM_MODEL_DATAFRAME_FLUX: float,
        ATM_MODEL_DATAFRAME_WAVE: float
    }
    # ******************************************************************************************************************

    # LIMB DARKENING RELATED *******************************************************************************************
    LD_LAW_TO_FILE_PREFIX = {
        "linear": "lin",
        "cosine": "lin",
        "logarithmic": "log",
        "square_root": "sqrt",
    }

    AVAILABLE_LD_LAWS = list(LD_LAW_TO_FILE_PREFIX.keys())

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
    # ******************************************************************************************************************

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


class Settings(_Const):
    _instance = None

    # defaults #########################################################################################################
    DEFAULT_SETTINGS = {}

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

    ####################################################################################################################

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Settings, cls).__new__(cls)
            cls.DEFAULT_SETTINGS = cls.settings_serializer()
            # Put any initialization here.
            cls.read_and_update_config()
            cls._update_atlas_to_base_dir()
            cls._create_home()

        return cls._instance

    @classmethod
    def settings_serializer(cls):
        return {
            "CONFIG_FILE": cls.CONFIG_FILE,
            "LOG_CONFIG": cls.LOG_CONFIG,
            "SUPPRESS_WARNINGS": cls.SUPPRESS_WARNINGS,
            "SUPPRESS_LOGGER": cls.SUPPRESS_LOGGER,
            "HOME": cls.HOME,
            "REFLECTION_EFFECT": cls.REFLECTION_EFFECT,
            "REFLECTION_EFFECT_ITERATIONS": cls.REFLECTION_EFFECT_ITERATIONS,
            "LIMB_DARKENING_LAW": cls.LIMB_DARKENING_LAW,
            "DEFAULT_TEMPERATURE_PERTURBATION_PHASE_SHIFT": cls.DEFAULT_TEMPERATURE_PERTURBATION_PHASE_SHIFT,
            "SURFACE_DISPLACEMENT_TOL": cls.SURFACE_DISPLACEMENT_TOL,
            "RV_METHOD": cls.RV_METHOD,
            "RV_LAMBDA_INTERVAL": cls.RV_LAMBDA_INTERVAL,
            "MAX_DISCRETIZATION_FACTOR": cls.MAX_DISCRETIZATION_FACTOR,
            "MIN_DISCRETIZATION_FACTOR": cls.MIN_DISCRETIZATION_FACTOR,
            "NUMBER_OF_THREADS": cls.NUMBER_OF_THREADS,
            "NUMBER_OF_PROCESSES": cls.NUMBER_OF_PROCESSES,
            "NUMBER_OF_MCMC_PROCESSES": cls.NUMBER_OF_MCMC_PROCESSES,
            "POINTS_ON_ECC_ORBIT": cls.POINTS_ON_ECC_ORBIT,
            "MAX_RELATIVE_D_R_POINT": cls.MAX_RELATIVE_D_R_POINT,
            "MAX_SUPPLEMENTAR_D_DISTANCE": cls.MAX_SUPPLEMENTAR_D_DISTANCE,
            "MAX_SPOT_D_LONGITUDE": cls.MAX_SPOT_D_LONGITUDE,
            "MAX_SOLVER_ITERS": cls.MAX_SOLVER_ITERS,
            "MAX_CURVE_DATA_POINTS": cls.MAX_CURVE_DATA_POINTS,
            "TIMER": cls.TIMER,
            "PASSBAND_TABLES": cls.PASSBAND_TABLES,
            "LD_TABLES": cls.LD_TABLES,
            "CK04_ATM_TABLES": cls.CK04_ATM_TABLES,
            "K93_ATM_TABLES": cls.K93_ATM_TABLES,
            "ATM_ATLAS": cls.ATM_ATLAS
        }

    @classmethod
    def set_up_logging(cls):
        if os.path.isfile(cls.LOG_CONFIG):
            with open(cls.LOG_CONFIG) as f:
                conf_dict = json.loads(f.read())
            log_conf.dictConfig(conf_dict)
        else:
            logging.basicConfig(level=logging.INFO)

    @classmethod
    def read_and_update_config(cls, conf_path=None):
        if not conf_path:
            conf_path = cls.CONFIG_FILE

        if not os.path.isfile(conf_path):
            msg = (
                "Couldn't find configuration file. Using default settings.\n"
                "   To customize configuration using file either\n"
                "    - specify config with environment variable ELISA_CONFIG\n"
                "    - add conf/elisa_conf.ini under your virtualenv root \n")
            warnings.warn(msg, Warning)
            return

        c_parse.read(conf_path)
        cls.update_config()
        cls._update_atlas_to_base_dir()
        cls._create_home()

    @classmethod
    def configure(cls, **kwargs):
        for key, value in kwargs.items():
            if not hasattr(cls, key):
                raise ValueError("You are about to set configuration which doesn't exist")
            setattr(cls, key, value)
        cls._update_atlas_to_base_dir()

    @classmethod
    def _create_home(cls):
        os.makedirs(cls.HOME, exist_ok=True)

    @classmethod
    def update_config(cls):
        # **************************************************************************************************************
        if c_parse.has_section('general'):
            cls.SUPPRESS_WARNINGS = c_parse.getboolean('general', 'suppress_warnings', fallback=cls.SUPPRESS_WARNINGS)
            cls.LOG_CONFIG = c_parse.get('general', 'log_config', fallback=cls.LOG_CONFIG)

            if not os.path.isfile(cls.LOG_CONFIG):
                if not cls.SUPPRESS_WARNINGS:
                    warnings.warn(f"log config `{cls.LOG_CONFIG}` doesn't exist, rollback to default")
                cls.LOG_CONFIG = os.path.join(dirname(os.path.abspath(__file__)), 'logging.json')

            cls.SUPPRESS_LOGGER = c_parse.getboolean('general', 'suppress_logger', fallback=cls.SUPPRESS_LOGGER)
            cls.HOME = c_parse.getboolean('general', 'home', fallback=cls.HOME)
        # **************************************************************************************************************
        if c_parse.has_section('physics'):
            cls.REFLECTION_EFFECT = c_parse.getboolean('physics', 'reflection_effect', fallback=cls.REFLECTION_EFFECT)
            cls.REFLECTION_EFFECT_ITERATIONS = c_parse.getint('physics', 'reflection_effect_iterations',
                                                              fallback=cls.REFLECTION_EFFECT_ITERATIONS)
            cls.LIMB_DARKENING_LAW = c_parse.get('physics', 'limb_darkening_law', fallback=cls.LIMB_DARKENING_LAW)
            if cls.LIMB_DARKENING_LAW not in ['linear', 'cosine', 'logarithmic', 'square_root']:
                raise ValueError(f'{cls.LIMB_DARKENING_LAW} is not valid name of limb darkening law. '
                                 f'Available limb darkening laws are: `linear` '
                                 f'or `cosine`, `logarithmic`, `square_root`')

            cls.DEFAULT_TEMPERATURE_PERTURBATION_PHASE_SHIFT = \
                c_parse.getfloat('physics', 'default_temperature_perturbation_phase_shift',
                                 fallback=cls.DEFAULT_TEMPERATURE_PERTURBATION_PHASE_SHIFT)
            cls.SURFACE_DISPLACEMENT_TOL = \
                c_parse.getfloat('physics', 'surface_displacement_tol', fallback=cls.SURFACE_DISPLACEMENT_TOL)
            cls.RV_METHOD = c_parse.getfloat('physics', 'rv_method', fallback=cls.RV_METHOD)
            cls.RV_LAMBDA_INTERVAL = c_parse.getfloat('physics', 'rv_lambda_interval', fallback=cls.RV_LAMBDA_INTERVAL)
        # ******************************************************************************************************************
        if c_parse.has_section('computational'):
            cls.MAX_DISCRETIZATION_FACTOR = c_parse.getfloat('computational', 'max_discretization_factor',
                                                             fallback=cls.MAX_DISCRETIZATION_FACTOR)
            cls.MIN_DISCRETIZATION_FACTOR = c_parse.getfloat('computational', 'min_discretization_factor',
                                                             fallback=cls.MIN_DISCRETIZATION_FACTOR)
            cls.NUMBER_OF_THREADS = c_parse.getint('computational', 'number_of_threads', fallback=cls.NUMBER_OF_THREADS)
            if cls.NUMBER_OF_THREADS <= 0:
                raise ValueError("Invalid value for `number_of_threads`, allowed >= 1")

            cls.NUMBER_OF_PROCESSES = c_parse.getint('computational', 'number_of_processes',
                                                     fallback=cls.NUMBER_OF_PROCESSES)
            if cls.NUMBER_OF_PROCESSES > os.cpu_count():
                warnings.warn("argument number_of_processes is too big, fallback to number of machine cores")
                cls.NUMBER_OF_PROCESSES = int(os.cpu_count())

            cls.NUMBER_OF_MCMC_PROCESSES = c_parse.getint('computational', 'number_of_mcmc_processes',
                                                          fallback=cls.NUMBER_OF_MCMC_PROCESSES)
            if cls.NUMBER_OF_MCMC_PROCESSES > os.cpu_count():
                warnings.warn("argument number_of_mcmc_processes is too big, fallback to number of machine cores")
                cls.NUMBER_OF_MCMC_PROCESSES = int(os.cpu_count())

            cls.POINTS_ON_ECC_ORBIT = c_parse.getint('computational', 'points_on_ecc_orbit',
                                                     fallback=cls.POINTS_ON_ECC_ORBIT)
            cls.MAX_RELATIVE_D_R_POINT = c_parse.getfloat('computational', 'max_relative_d_r_point',
                                                          fallback=cls.MAX_RELATIVE_D_R_POINT)
            cls.MAX_SUPPLEMENTAR_D_DISTANCE = c_parse.getfloat('computational', 'max_supplementar_d_distance',
                                                               fallback=cls.MAX_SUPPLEMENTAR_D_DISTANCE)
            cls.MAX_SPOT_D_LONGITUDE = c_parse.getfloat('computational', 'max_spot_d_longitude',
                                                        fallback=cls.MAX_SPOT_D_LONGITUDE)
            cls.MAX_SOLVER_ITERS = c_parse.getfloat('computational', 'max_solver_iters', fallback=cls.MAX_SOLVER_ITERS)
            cls.MAX_CURVE_DATA_POINTS = c_parse.getfloat('computational', 'max_curve_datapoints',
                                                         fallback=cls.MAX_CURVE_DATA_POINTS)
        # ******************************************************************************************************************
        if c_parse.has_section('support'):
            cls.LD_TABLES = c_parse.get('support', 'ld_tables', fallback=cls.LD_TABLES)

            if not isdir(cls.LD_TABLES) and not cls.SUPPRESS_WARNINGS:
                warnings.warn(f"path {cls.LD_TABLES} to limb darkening tables doesn't exists\n"
                              f"Specifiy it in elisa_conf.ini file")

            cls.CK04_ATM_TABLES = c_parse.get('support', 'castelli_kurucz_04_atm_tables', fallback=cls.CK04_ATM_TABLES)

            if not os.path.isdir(cls.CK04_ATM_TABLES) and not cls.SUPPRESS_WARNINGS:
                warnings.warn(f"path {cls.CK04_ATM_TABLES}\n"
                              f"to castelli-kurucz 2004 atmosphere atlas doesn't exists\n"
                              f"Specifiy it in elisa_conf.ini file")

            cls.K93_ATM_TABLES = c_parse.get('support', 'kurucz_93_atm_tables', fallback=cls.K93_ATM_TABLES)

            if not os.path.isdir(cls.K93_ATM_TABLES):
                warnings.warn(f"path {cls.K93_ATM_TABLES}\n"
                              "to kurucz 1993 atmosphere atlas doesn't exists\n"
                              "Specifiy it in elisa_conf.ini file")

            cls.ATM_ATLAS = c_parse.get('support', 'atlas', fallback=cls.ATM_ATLAS)
            cls.PASSBAND_TABLES = c_parse.get('support', 'passband_tables', fallback=cls.PASSBAND_TABLES)

            if not isdir(cls.PASSBAND_TABLES) and not cls.SUPPRESS_WARNINGS:
                warnings.warn(f"path {cls.PASSBAND_TABLES} to passband tables doesn't exists\n"
                              f"Specifiy it in elisa_conf.ini file")
        # ******************************************************************************************************************

    @classmethod
    def _update_atlas_to_base_dir(cls):
        cls.ATLAS_TO_BASE_DIR.update({
            "castelli": cls.CK04_ATM_TABLES,
            "castelli-kurucz": cls.CK04_ATM_TABLES,
            "ck": cls.CK04_ATM_TABLES,
            "ck04": cls.CK04_ATM_TABLES,
            "kurucz": cls.K93_ATM_TABLES,
            "k": cls.K93_ATM_TABLES,
            "k93": cls.K93_ATM_TABLES
        })


settings = Settings()
