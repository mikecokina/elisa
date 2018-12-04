import json
import logging
import os
import warnings
from configparser import ConfigParser
from logging import config as log_conf

config_parser = ConfigParser()

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
                      "  - Add conf/elisa_conf.ini under your virtualenv root, or \n ")

CONFIG_FILE = config_file
LOG_CONFIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'conf', 'logging.json')
# physics
REFLECTION_EFFECT = True
REFLECTION_EFFECT_ITERATIONS = 2
LIMB_DARKENING_LAW = 'cosine'
# computational
DISCRETIZATION_FACTOR = 5
MAX_DISCRETIZATION_FACTOR = 20

VAN_HAMME_LD_TABLES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "limbdarkening")


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
            "    - place your config to default location ~/elisa.ini"
        )
        warnings.warn(msg, Warning)
        return

    config_parser.read(conf_path)
    update_config()


def update_config():
    if config_parser.has_section('general'):
        global LOG_CONFIG
        LOG_CONFIG = config_parser.get('general', 'log_config', fallback=LOG_CONFIG)

    if config_parser.has_section('physics'):
        global REFLECTION_EFFECT
        REFLECTION_EFFECT = config_parser.getboolean('physics', 'reflection_effect', fallback=REFLECTION_EFFECT)

        global REFLECTION_EFFECT_ITERATIONS
        REFLECTION_EFFECT_ITERATIONS = config_parser.getint('physics', 'reflection_effect_iterations',
                                                            fallback=REFLECTION_EFFECT_ITERATIONS)

        global LIMB_DARKENING_LAW
        LIMB_DARKENING_LAW = config_parser.get('physics', 'limb_darkening_law',
                                               fallback=LIMB_DARKENING_LAW)
        if LIMB_DARKENING_LAW not in ['linear', 'cosine', 'logarithmic', 'square_root']:
            raise ValueError('{0} is not valid name of limb darkening law. Available limb darkening laws are: `linear` '
                             'or `cosine`, `logarithmic`, `square_root`'.format(LIMB_DARKENING_LAW))

    if config_parser.has_section('computational'):
        global DISCRETIZATION_FACTOR
        DISCRETIZATION_FACTOR = config_parser.getfloat('computational', 'discretization_factor',
                                                       fallback=DISCRETIZATION_FACTOR)

        global MAX_DISCRETIZATION_FACTOR
        MAX_DISCRETIZATION_FACTOR = config_parser.getfloat('computational', 'max_discretization_factor',
                                                           fallback=MAX_DISCRETIZATION_FACTOR)

    if config_parser.has_section('support'):
        global VAN_HAMME_LD_TABLES
        VAN_HAMME_LD_TABLES = config_parser.get('support', 'van_hamme_ld_tables', fallback=VAN_HAMME_LD_TABLES)

        if not os.path.isdir(VAN_HAMME_LD_TABLES):
            warnings.warn("path {}\n"
                          "to van hamme ld tables doesn't exists\n"
                          "Specifiy it in elisa_conf.ini file".format(VAN_HAMME_LD_TABLES))


# fixme: import ths file somewher in code instead of previous engine.conf
PASSBAND = [
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

read_and_update_config()
