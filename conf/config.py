import os
import logging
import json
import warnings

from configparser import ConfigParser
from logging import config as log_conf

config_parser = ConfigParser()


CONFIG_FILE = os.environ.get('ELISA_CONFIG', os.path.expanduser('~/elisa.ini'))
LOG_CONFIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'conf', 'logging.json')
EXAMPLE_PARAM_1 = "ex1"
EXAMPLE_PARAM_2 = "ex2"


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

        global EXAMPLE_PARAM_1
        EXAMPLE_PARAM_1 = config_parser.get('general', 'example_param_1', fallback=EXAMPLE_PARAM_1)

    if config_parser.has_section('any_group'):
        global EXAMPLE_PARAM_2
        EXAMPLE_PARAM_2 = config_parser.get('any_group', 'example_param_2', fallback=EXAMPLE_PARAM_2)


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
