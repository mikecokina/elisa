import os
import sys
import configparser
import os.path as op
import numpy as np

from pathlib import Path


class DefaultSettings(object):
    LOG_CONFIG = op.join(op.dirname(op.abspath(__file__)), 'logging_schemas/default.json')
    SUPPRESS_WARNINGS = False
    SUPPRESS_LOGGER = None
    HOME = op.expanduser(op.join("~", '.elisa'))

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
    DEFAULT_DISCRETIZATION_FACTOR = 5
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
    PASSBAND_TABLES = op.join(op.dirname(op.abspath(__file__)), op.pardir, "passband")
    LD_TABLES = op.join(HOME, "limbdarkening", "ld")
    CK04_ATM_TABLES = op.join(HOME, "atmosphere", "ck04")
    K93_ATM_TABLES = op.join(HOME, "atmosphere", "k93")
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
    MAGNITUDE_SYSTEM = 'vega'


class SettingsManager(object):
    @staticmethod
    def run():
        def _create_configuration_path_input(_directory):
            if not op.isdir(_directory):
                os.makedirs(_directory, exist_ok=True)

        def _write_ini_file(_config, _path):
            with open(_path, "w") as _f:
                _config.write(_f)

        def _system_exit(exit_code):
            sys.exit(exit_code)

        print("========================== ELISa - first time run ==========================")
        print("It seems that you are using ELISa for the first time.\n"
              "ELISa requires your attention to do a minimal necessary configuration.\n"
              "Please type your configuration preferenceies in following wizzard.\n"
              "If you are power user and you would like to configure it manually, \n"
              "than abort this wizzard and put your configuration in following posible locations: \n"
              "\t- set the environment variable ELISA_CONFIG with absolute path to you config file, or \n "
              "\t- add conf/elisa_conf.ini under your virtualenv root, or \n"
              "\t- add ~/.elisa/config.ini file.\n"
              "For more cofigurable options take a look at: \n"
              "\t\thttps://github.com/mikecokina/elisa/blob/dev/src/elisa/conf/elisa_conf_docs.ini")

        proceed = input("Do you want to continue [y/N]: ")
        if not proceed.lower() == "y":
            _system_exit(0)

        # sections
        section_general = "general"
        section_support = "support"

        # create default ini and add sections
        config = configparser.ConfigParser()
        config.add_section(section_general)
        config.add_section(section_support)

        # configuration file
        default_config_path = op.join(DefaultSettings.HOME, 'config.ini')
        file_prompt = f"Absolute path to configuration file\n" \
                      f"[ hit enter for default {default_config_path} ]: "
        configuration_file = input(file_prompt) or default_config_path
        _create_configuration_path_input(str(Path(configuration_file).parent))
        config.set(section_general, "home", str(Path(configuration_file).parent))

        # ld tables path
        ld_prompt = f"Absolute path to limb darkening tables (will be downloaded)\n" \
                    f"[ default {DefaultSettings.LD_TABLES} ]: "
        ld_tables = input(ld_prompt) or DefaultSettings.LD_TABLES
        _create_configuration_path_input(ld_tables)
        config.set(section_support, "ld_tables", ld_tables)

        # atlas
        atlas_prompt = f"Choose default atmosphere model to use \n" \
                       f"\t type ck04 for catelli-kurucz 2004\n" \
                       f"\t type k93 for kurucz 1993\n" \
                       f"[ default  {DefaultSettings.ATM_ATLAS} ]: "
        atm_atlas = input(atlas_prompt) or DefaultSettings.ATM_ATLAS
        if atm_atlas not in ["ck04", "k93"]:
            raise ValueError("Invalid input for atmosphere atlas. Type option ck04 or k93.")
        config.set(section_support, "atlas", atm_atlas)

        # castelli_kurucz_04_atm_tables
        ck04_prompt = f"Absolute path to store castelli-kurucz 2004 atmosphere models (will be downloaded)\n" \
                      f"[ default {DefaultSettings.CK04_ATM_TABLES} ]: "
        ck04_atm_tables = input(ck04_prompt) or DefaultSettings.CK04_ATM_TABLES
        _create_configuration_path_input(ck04_atm_tables)
        config.set(section_support, "castelli_kurucz_04_atm_tables", ck04_atm_tables)

        # kurucz_93_atm_tables
        k93_prompt = f"Absolute path to store kurucz 1993 atmosphere models (will be downloaded)\n" \
                     f"[ default {DefaultSettings.K93_ATM_TABLES} ]: "
        k93_atm_tables = input(k93_prompt) or DefaultSettings.K93_ATM_TABLES
        _create_configuration_path_input(k93_atm_tables)
        config.set(section_support, "kurucz_93_atm_tables", k93_atm_tables)

        # write file
        _write_ini_file(config, default_config_path)

