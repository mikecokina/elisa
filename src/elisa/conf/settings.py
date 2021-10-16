import json
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

    ATM_ATLAS_NORMALIZER = {
        **ATLAS_TO_ATM_FILE_PREFIX,
        **{"bb": "bb", "black_body": "bb"}
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
    DATA_PATH = os.path.join(dirname(dirname(__file__)), "data")

    # paths to mesh correction factors
    PATH_TO_SINGLE_CORRECTIONS = os.path.join(DATA_PATH, 'mesh_corrections', 'correction_factors_single.npy')
    PATH_TO_DETACHED_CORRECTIONS = os.path.join(DATA_PATH, 'mesh_corrections', 'correction_factors_detached.npy')
    PATH_TO_OVER_CONTACT_CORRECTIONS = os.path.join(
        DATA_PATH, 'mesh_corrections', 'correction_factors_over-contact.npy'
    )
    PATH_TO_ALBEDOS = os.path.join(DATA_PATH, 'albedos.json')
    PATH_TO_BETA = os.path.join(DATA_PATH, 'gravity_darkening.json')


class Settings(_Const):
    _instance = None

    # defaults #########################################################################################################
    DEFAULT_SETTINGS = {}

    # schema registry
    SCHEMA_REGISTRY = registry.Registry()

    # basic app configuration
    CONFIG_FILE = config_file
    LOG_CONFIG = os.path.join(dirname(os.path.abspath(__file__)), 'logging_schemas/default.json')
    SUPPRESS_WARNINGS = False
    SUPPRESS_LOGGER = None
    HOME = os.path.expanduser(os.path.join("~", '.elisa'))

    # physics
    REFLECTION_EFFECT = True
    REFLECTION_EFFECT_ITERATIONS = 2
    LIMB_DARKENING_LAW = 'cosine'
    PULSATION_MODEL = 'uniform'
    DEFAULT_TEMPERATURE_PERTURBATION_PHASE_SHIFT = np.pi / 3.0
    SURFACE_DISPLACEMENT_TOL = 1e-2
    RV_METHOD = 'kinematic'
    RV_LAMBDA_INTERVAL = (5500, 5600)

    # computational
    MAX_DISCRETIZATION_FACTOR = 8
    MIN_DISCRETIZATION_FACTOR = 3
    NUMBER_OF_THREADS = 1
    NUMBER_OF_PROCESSES = -1  # int(os.cpu_count())
    NUMBER_OF_MCMC_PROCESSES = -1
    MAX_NU_SEPARATION = 0.08
    MAX_D_FLUX = 2e-4
    MAX_SPOT_D_LONGITUDE = np.pi / 180.0  # in radians
    MIN_POINTS_IN_ECLIPSE = 35
    MAX_SOLVER_ITERS = 100
    MAX_CURVE_DATA_POINTS = 300
    MESH_GENERATOR = 'auto'
    DEFORMATION_TOL = 0.05
    MCMC_SAVE_INTERVAL = 1800
    USE_SINGLE_LD_COEFFICIENTS = False
    USE_INTERPOLATION_APPROXIMATION = True
    USE_SYMMETRICAL_COUNTERPARTS_APPROXIMATION = True
    USE_SIMILAR_NEIGHBOURS_APPROXIMATION = True

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
    CUDA = False

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
            "MAX_NU_SEPARATION": cls.MAX_NU_SEPARATION,
            "MAX_D_FLUX": cls.MAX_D_FLUX,
            "MAX_SPOT_D_LONGITUDE": cls.MAX_SPOT_D_LONGITUDE,
            "MAX_SOLVER_ITERS": cls.MAX_SOLVER_ITERS,
            "MAX_CURVE_DATA_POINTS": cls.MAX_CURVE_DATA_POINTS,
            "MIN_POINTS_IN_ECLIPSE": cls.MIN_POINTS_IN_ECLIPSE,
            "TIMER": cls.TIMER,
            "PASSBAND_TABLES": cls.PASSBAND_TABLES,
            "LD_TABLES": cls.LD_TABLES,
            "CK04_ATM_TABLES": cls.CK04_ATM_TABLES,
            "K93_ATM_TABLES": cls.K93_ATM_TABLES,
            "ATM_ATLAS": cls.ATM_ATLAS,
            "MESH_GENERATOR": cls.MESH_GENERATOR,
            "DEFORMATION_TOL": cls.DEFORMATION_TOL,
            "PULSATION_MODEL": cls.PULSATION_MODEL,
            "MCMC_SAVE_INTERVAL": cls.MCMC_SAVE_INTERVAL,
            "CUDA": cls.CUDA,
            "USE_SINGLE_LD_COEFFICIENTS": cls.USE_SINGLE_LD_COEFFICIENTS,
            "USE_INTERPOLATION_APPROXIMATION": cls.USE_INTERPOLATION_APPROXIMATION,
            "USE_SYMMETRICAL_COUNTERPARTS_APPROXIMATION": cls.USE_SYMMETRICAL_COUNTERPARTS_APPROXIMATION,
            "USE_SIMILAR_NEIGHBOURS_APPROXIMATION": cls.USE_SIMILAR_NEIGHBOURS_APPROXIMATION,
        }

    @staticmethod
    def load_conf(path):
        with open(path) as f:
            conf_dict = json.loads(f.read())
        log_conf.dictConfig(conf_dict)

    @classmethod
    def set_up_logging(cls):
        if os.path.isfile(cls.LOG_CONFIG):
            cls.load_conf(cls.LOG_CONFIG)
            return
        elif cls.LOG_CONFIG == 'default':
            cls.LOG_CONFIG = os.path.join(dirname(os.path.abspath(__file__)),
                                          'logging_schemas/default.json')
        elif cls.LOG_CONFIG == 'fit':
            cls.LOG_CONFIG = os.path.join(dirname(os.path.abspath(__file__)),
                                          'logging_schemas/fit.json')
        else:
            cls.LOG_CONFIG = os.path.join(dirname(os.path.abspath(__file__)),
                                          'logging_schemas/default.json')

        cls.load_conf(cls.LOG_CONFIG)

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
            if key == 'LOG_CONFIG':
                cls.set_up_logging()
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

            cls.set_up_logging()

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
            cls.PULSATION_MODEL = c_parse.getfloat('physics', 'pulsation_model', fallback=cls.PULSATION_MODEL)
        # **************************************************************************************************************
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
                if not cls.SUPPRESS_WARNINGS:
                    warnings.warn("argument number_of_processes is too big, fallback to number of machine cores")
                cls.NUMBER_OF_PROCESSES = int(os.cpu_count())

            cls.NUMBER_OF_MCMC_PROCESSES = c_parse.getint('computational', 'number_of_mcmc_processes',
                                                          fallback=cls.NUMBER_OF_MCMC_PROCESSES)
            if cls.NUMBER_OF_MCMC_PROCESSES > os.cpu_count():
                if not cls.SUPPRESS_WARNINGS:
                    warnings.warn("argument number_of_mcmc_processes is too big, fallback to number of machine cores")
                cls.NUMBER_OF_MCMC_PROCESSES = int(os.cpu_count())

            cls.MAX_NU_SEPARATION = c_parse.getfloat('computational', 'max_nu_separation',
                                                     fallback=cls.MAX_NU_SEPARATION)
            cls.MAX_D_FLUX = c_parse.getfloat('computational', 'max_d_flux', fallback=cls.MAX_D_FLUX)
            cls.MAX_SPOT_D_LONGITUDE = c_parse.getfloat('computational', 'max_spot_d_longitude',
                                                        fallback=cls.MAX_SPOT_D_LONGITUDE)
            cls.MAX_SOLVER_ITERS = c_parse.getfloat('computational', 'max_solver_iters', fallback=cls.MAX_SOLVER_ITERS)
            cls.MAX_CURVE_DATA_POINTS = c_parse.getfloat('computational', 'max_curve_datapoints',
                                                         fallback=cls.MAX_CURVE_DATA_POINTS)
            cls.MIN_POINTS_IN_ECLIPSE = c_parse.getint('computational', 'min_points_in_eclipse',
                                                       fallback=cls.MIN_POINTS_IN_ECLIPSE)
            cls.MESH_GENERATOR = c_parse.getfloat('computational', 'mesh_generator', fallback=cls.MESH_GENERATOR)
            cls.DEFORMATION_TOL = c_parse.getfloat('computational', 'deformation_tol', fallback=cls.DEFORMATION_TOL)
            cls.MCMC_SAVE_INTERVAL = c_parse.getfloat('computational', 'mcmc_save_interval',
                                                      fallback=cls.MCMC_SAVE_INTERVAL)

            cls.CUDA = c_parse.getboolean('computational', 'cuda', fallback=cls.CUDA)
            if cls.CUDA:
                try:
                    import torch
                    if not torch.cuda.is_available():
                        cls.CUDA = False
                        warnings.warn("You have no CUDA enabled/available on your device. "
                                      "Runtime continue with CPU.", UserWarning)

                except ImportError:
                    warnings.warn("You need to install `pytorch` with cuda to be "
                                  "able to use CUDA features. Fallback to CPU.", UserWarning)
                    cls.CUDA = False

            cls.USE_SINGLE_LD_COEFFICIENTS = c_parse.getboolean('computational', 'use_single_ld_coefficients',
                                                                fallback=cls.USE_SINGLE_LD_COEFFICIENTS)

            cls.USE_INTERPOLATION_APPROXIMATION = c_parse.getboolean(
                'computational', 'use_interpolation_approximation', fallback=cls.USE_INTERPOLATION_APPROXIMATION
            )

            cls.USE_SYMMETRICAL_COUNTERPARTS_APPROXIMATION = c_parse.getboolean(
                'computational', 'use_symmetrical_counterparts_approximation',
                fallback=cls.USE_SYMMETRICAL_COUNTERPARTS_APPROXIMATION
            )

            cls.USE_SIMILAR_NEIGHBOURS_APPROXIMATION = c_parse.getboolean(
                'computational', 'use_similar_neighbours_approximation',
                fallback=cls.USE_SIMILAR_NEIGHBOURS_APPROXIMATION
            )

        # **************************************************************************************************************
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

            if not os.path.isdir(cls.K93_ATM_TABLES) and not cls.SUPPRESS_WARNINGS:
                warnings.warn(f"path {cls.K93_ATM_TABLES}\n"
                              "to kurucz 1993 atmosphere atlas doesn't exists\n"
                              "Specifiy it in elisa_conf.ini file", UserWarning)

            if c_parse.get('support', 'atlas', fallback=None):
                with np.errstate(divide='ignore', invalid='ignore'):
                    if not cls.SUPPRESS_WARNINGS:
                        warnings.simplefilter("always", DeprecationWarning)
                        warnings.warn("Variable `atlas` in configuration section `support` is not "
                                      "longer supported and will be removed in future version.\n"
                                      "Use atmosphere definition as initial parameter "
                                      "for given celestial object", DeprecationWarning)
                        warnings.simplefilter("ignore", DeprecationWarning)
            cls.ATM_ATLAS = c_parse.get('support', 'atlas', fallback=cls.ATM_ATLAS)
            cls.PASSBAND_TABLES = c_parse.get('support', 'passband_tables', fallback=cls.PASSBAND_TABLES)

            if not isdir(cls.PASSBAND_TABLES) and not cls.SUPPRESS_WARNINGS:
                warnings.warn(f"path {cls.PASSBAND_TABLES} to passband tables doesn't exists\n"
                              f"Specifiy it in elisa_conf.ini file", UserWarning)
        # **************************************************************************************************************

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
