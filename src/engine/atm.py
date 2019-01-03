import logging
import os

import numpy as np
import pandas as pd

from conf import config
from engine import utils, const

from queue import Queue
from threading import Thread

config.set_up_logging()
logger = logging.getLogger("atm")


# * 1e-7 * 1e4 * 1e10 * (1.0/np.pi)

ATLAS_TO_ATM_FILE_PREFIX = {
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


ATM_DOMAIN_QUANTITY_TO_VARIABLE_SUFFIX = {
    "temperature": "TEMPERATURE_LIST_ATM",
    "gravity": "GRAVITY_LIST_ATM",
    "metallicity": "METALLICITY_LIST_ATM"
}

class AtmDataContainer(object):
    def __init__(self, model, temperature, logg, metallicity):
        self.model = model
        self.temperature = temperature
        self.logg = logg
        self.metallicity = metallicity
        self.flux_unit = "flam"
        self.wave_unit = "angstrom"
        self.flux_to_si_mult = 1e-7 * 1e4 * 1e10 * (1.0/np.pi)
        self.wave_to_si_mult = 1e-10


def atm_file_prefix_to_quantity_list(qname, atlas):
    atlas = validated_atlas(atlas)
    return getattr(
        const,
        "{}_{}".format(
            str(atlas).upper(),
            str(ATM_DOMAIN_QUANTITY_TO_VARIABLE_SUFFIX[qname])
        )
    )

def validated_atlas(atlas):
    try:
        return ATLAS_TO_ATM_FILE_PREFIX[atlas]
    except KeyError:
        raise KeyError("Incorrect atlas. Following are allowed: {}"
                       "".format(", ".join(ATLAS_TO_ATM_FILE_PREFIX.keys())))


def parse_domain_quantities_from_atm_table_filename(filename):
    return get_temperature_from_atm_table_filename(filename), \
           get_logg_from_atm_table_filename(filename), \
           get_metallicity_from_atm_table_filename(filename)


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

def get_temperature_from_atm_table_filename(filename):
    return float(str(filename).split("_")[1])


def get_logg_from_atm_table_filename(filename):
    filename = filename if not str(filename).endswith(".csv") else str(filename).replace('.csv', '')
    g = str(filename).split("_")[2][1:]
    return int(g) / 10.0


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


def get_atm_table(temperature, logg, metallicity, atlas):
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


def get_list_of_all_atm_tables(atlas):
    source = ATLAS_TO_BASE_DIR[validated_atlas(atlas)]
    matches = []
    for root, dirnames, filenames in os.walk(source):
        for filename in filenames:
            if filename.endswith(('.csv', )):
                matches.append(os.path.join(root, filename))
    return matches


def get_relevant_atm_tables(temperature, logg, metallicity, atlas):
    pass


def nearest_atm_tables_list(temperature, logg, metallicity, atlas):
    atlas = validated_atlas(atlas)

    t_array = atm_file_prefix_to_quantity_list("temperature", atlas)
    g_array = atm_file_prefix_to_quantity_list("gravity", atlas)
    m_array = atm_file_prefix_to_quantity_list("metallicity", atlas)

    t = [utils.find_nearest_value(t_array, _t)[0] for _t in temperature]
    g = [utils.find_nearest_value(g_array, _g)[0] for _g in logg]
    m = utils.find_nearest_value(m_array, metallicity)[0]

    domain_df = pd.DataFrame({
        "temp": t,
        "logg": g,
        "mh": [m] * len(t)
    })

    directory = get_atm_directory(m, atlas)
    fnames = str(atlas) + \
        domain_df["mh"].apply(lambda x: utils.numeric_metallicity_to_string(x)) + "_" + \
        domain_df["temp"].apply(lambda x: str(int(x))) + "_" + \
        domain_df["logg"].apply(lambda x: utils.numeric_logg_to_string(x))

    return list(os.path.join(str(ATLAS_TO_BASE_DIR[atlas]), str(directory)) + "/" + fnames + ".csv")


def nearest_atm_tables(temperature, logg, metallicity, atlas):
    # todo: make configurable
    n_threads = 4

    fpaths = nearest_atm_tables_list(temperature, logg, metallicity, atlas)

    path_queue = Queue(maxsize=len(fpaths) + n_threads)
    result_queue = Queue()
    error_queue = Queue()

    threads = list()
    try:
        for fpath in fpaths:
            if not os.path.isfile(fpath):
                raise FileNotFoundError(
                    "file {} doesn't exist. it seems your model could be not physical".format(fpath))
            path_queue.put(fpath)

        for _ in range(n_threads):
            path_queue.put("TERMINATOR")

        logger.debug("initialising multithread atm table reader")
        for _ in range(n_threads):
            t = Thread(target=multithread_atm_tables_reader, args=(path_queue, error_queue, result_queue))
            threads.append(t)
            t.daemon = True
            t.start()

        for t in threads:
            t.join()
        logger.debug("atm multithread reader finished all jobs")
    except KeyboardInterrupt:
        raise
    finally:
        if not error_queue.empty():
            raise error_queue.get()

    models = [qval for qval in utils.IterableQueue(result_queue)]
    return models


def multithread_atm_tables_reader(path_queue: Queue, error_queue: Queue, result_queue: Queue):
    while True:
        file_path = path_queue.get(timeout=1)

        if file_path == "TERMINATOR":
            break
        if not error_queue.empty():
            break
        try:
            t, l, m = parse_domain_quantities_from_atm_table_filename(os.path.basename(file_path))
            atm_container = AtmDataContainer(pd.read_csv(file_path), t, l, m)
            result_queue.put(atm_container)
        except Exception as we:
            error_queue.put(we)
            break


def get_nearest_atm_data():
    pass


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

    print(nearest_atm_tables(_temperature, _logg, _metallicity, "ck")[4])
