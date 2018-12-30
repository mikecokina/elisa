import logging
import os

import numpy as np
import pandas as pd

from conf import config
from engine import utils

config.set_up_logging()
logger = logging.getLogger("atm")


ATLAS_TO_FILE_PREFIX = {
    "castelli": "ck",
    "castelli-kurucz": "ck",
    "ck": "ck",
    "ck04": "ck"
    # implement kurucz 93
}

ATLAS_TO_BASE_DIR = {
    "castelli": config.CK04_ATM_TABLES,
    "castelli-kurucz": config.CK04_ATM_TABLES,
    "ck": config.CK04_ATM_TABLES,
    "ck04": config.CK04_ATM_TABLES
    # implement kurucz 93
}

def validated_atlas(atlas):
    try:
        return ATLAS_TO_FILE_PREFIX[atlas]
    except KeyError:
        raise KeyError("Incorrect atlas. Following are allowed: {}"
                       "".format(", ".join(ATLAS_TO_FILE_PREFIX.keys())))


def get_metallicity_from_atm_table_filename(filename):
    """
    get metallicity as number from filename / directory

    :param filename: str
    :return: float
    """
    m = str(filename).split("_")[0][-3:]
    sign = 1 if str(m).startswith("p") else -1
    value = float(m[1:]) / 10.0
    return value * sign


def get_atm_table_filename(temperature, logg, metallicity, atlas):
    """
    get filename based on given descriptive values

    :param temperature: float
    :param logg: float
    :param metallicity: float
    :param atlas: str
    :return: str
    """
    prefix = validated_atlas(atlas)
    return "{prefix}{metallicity}_{temperature}_{logg}.csv".format(
        prefix=prefix, metallicity=utils.numeric_metallicity_to_string(metallicity),
        temperature=int(temperature),
        logg=utils.numeric_logg_to_string(logg)
    )


def get_atm_directory(metallicity, atlas):
    """
    get table directory name based on given descriptive  evalues

    :param metallicity: float
    :param atlas: str
    :return: str
    """
    prefix = validated_atlas(atlas)
    return "{prefix}{metallicity}".format(
        prefix=prefix, metallicity=utils.numeric_metallicity_to_string(metallicity),
    )


def get_van_hamme_ld_table(temperature, logg, metallicity, atlas):
    """
    get dataframe for flux and wavelengths for given values

    :param temperature: float
    :param logg: float
    :param metallicity: float
    :param atlas: str
    :return: pandas.DataFrame
    """
    directory = get_atm_directory(metallicity, atlas)
    filename = get_atm_table_filename(temperature, logg, metallicity, atlas)
    path = os.path.join(ATLAS_TO_BASE_DIR[atlas], directory, filename) if directory is not None else \
        os.path.join(ATLAS_TO_BASE_DIR[atlas], filename)

    if not os.path.isfile(path):
        raise FileNotFoundError("there is no file like {}".format(path))
    return pd.read_csv(path)


if __name__ == "__main__":
    _temperature = [
        5551.36,
        5552.25,
        6531.81,
        7825.66,
        9874.85
    ]

    _metallicity = 0.11

    _logg = [
        4.12,
        3.92,
        2.85,
        2.99,
        3.11
    ]

    print(get_van_hamme_ld_table(3500.0, 3.0, 0.0, "ck04"))
