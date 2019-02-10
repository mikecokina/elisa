import logging
import os
from collections import Iterable
from queue import Queue
from threading import Thread

import itertools
import numpy as np
import pandas as pd
from scipy import integrate

from elisa.conf import config
from elisa.engine import utils, const

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
    def __init__(self, model: pd.DataFrame, temperature: float, logg: float, metallicity: float):
        self.model = model
        self.temperature = temperature
        self.logg = logg
        self.metallicity = metallicity
        self.flux_unit = "flam"
        self.wave_unit = "angstrom"
        # in case this np.pi will stay here, there will be rendundant multiplication in intensity integration
        self.flux_to_si_mult = 1e-7 * 1e4 * 1e10  # * (1.0/np.pi)
        self.wave_to_si_mult = 1e-10


class IntensityContainer(object):
    def __init__(self, intensity, temperature, logg, metallicity):
        self.intensity = intensity
        self.temperature = temperature
        self.logg = logg
        self.metallicity = metallicity


def atm_file_prefix_to_quantity_list(qname, atlas):
    """
    get list of available values for given atm domain quantity, e.g. list of temperatures available in atlas CK04

    :param qname: str - e.g. `temperature`, `metallicity`, `gravity`
    :param atlas: str - e.g. `castelli` or `ck04`
    :return: list
    """
    atlas = validated_atlas(atlas)
    return getattr(
        const,
        "{}_{}".format(
            str(atlas).upper(),
            str(ATM_DOMAIN_QUANTITY_TO_VARIABLE_SUFFIX[qname])
        )
    )


def validated_atlas(atlas):
    """
    get validated atm atlas, e.g. `castelli` or `ck04` transform to `ck`, it matches folder
    and file prefix for given atlas

    :param atlas: str - e.g. `castelli` or `ck04`
    :return: str
    """
    try:
        return ATLAS_TO_ATM_FILE_PREFIX[atlas]
    except KeyError:
        raise KeyError("Incorrect atlas. Following are allowed: {}"
                       "".format(", ".join(ATLAS_TO_ATM_FILE_PREFIX.keys())))


def parse_domain_quantities_from_atm_table_filename(filename):
    """
    parse filename to given quantities, e.g. ckm05_3500_g15.csv parse to tuple (-0.5, 3500, 1.5)

    :param filename: str
    :return: tuple
    """
    return get_temperature_from_atm_table_filename(filename), get_logg_from_atm_table_filename(
        filename), get_metallicity_from_atm_table_filename(filename)


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
    :param atlas: str- e.g. `castelli` or `ck04`
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
    :param atlas: str - e.g. `castelli` or `ck04`
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
    :param atlas: str - e.g. `castelli` or `ck04`
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
    """
    get list of all available atm table files stored in configured location

    :param atlas: str - e.g. `castelli` or `ck04`
    :return: list
    """
    source = ATLAS_TO_BASE_DIR[validated_atlas(atlas)]
    matches = []
    for root, dirnames, filenames in os.walk(source):
        for filename in filenames:
            if filename.endswith(('.csv',)):
                matches.append(os.path.join(root, filename))
    return matches


def get_relevant_atm_tables(temperature, logg, metallicity, atlas, method):
    pass


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


def compute_integral_si_intensity_from_atm_data_containers(atm_data_containers: list):
    """
    returns bolometric intensity
    :param atm_data_containers:
    :return:
    """
    return [
        np.pi * integrate.simps(adc.model["flux"] * adc.flux_to_si_mult,
                                adc.model["wave"] * adc.wave_to_si_mult)
        for adc in atm_data_containers
    ]


class NearestAtm(object):
    @staticmethod
    def nearest_atm_files_list(temperature, logg, metallicity, atlas):
        """
        returns files that contains atmospheric model for parameters closest to the given atmospheric parameters
        `temperature`, `logg` and `metallicity`

        :param temperature: list
        :param logg: list
        :param metallicity: list
        :param atlas: str - e.g. `castelli` or `ck04`
        :return:
        """
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

    @staticmethod
    def nearest_atm_tables(temperature, logg, metallicity, atlas):
        """
        returns spectrum profile for the atmospheric model that is the closest to the given parameters `temperature`, `logg`
        and `metallicity`

        :param temperature: list
        :param logg: list
        :param metallicity: list
        :param atlas: str - e.g. `castelli` or `ck04`
        :return:
        """
        n_threads = config.NUMBER_OF_THREADS

        fpaths = NearestAtm.nearest_atm_files_list(temperature, logg, metallicity, atlas)

        path_queue = Queue(maxsize=len(fpaths) + n_threads)
        result_queue = Queue()
        error_queue = Queue()

        threads = list()
        try:
            for fpath in fpaths:
                if not os.path.isfile(fpath):
                    raise FileNotFoundError("file {} doesn't exist. it seems your model "
                                            "could be not physical".format(fpath))
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


def build_atm_validation_hypertable(atlas):
    atlas = validated_atlas(atlas)
    all_files = get_list_of_all_atm_tables(atlas)
    filenames = (os.path.basename(f) for f in all_files)
    quantities = sorted([parse_domain_quantities_from_atm_table_filename(f) for f in filenames], key=lambda x: x[0])
    temp_qroups = itertools.groupby(quantities, key=lambda x: x[0])
    hypertable = {
        str(int(temp_qroup[0])):
            {
                "gravity": sorted(set(np.array(list(temp_qroup[1])).T[1])),
                # mettalicity is stored in this dict just because of consitency
                "metallicity": atm_file_prefix_to_quantity_list("metallicity", atlas)
            }
        for temp_qroup in temp_qroups
    }
    return hypertable


def is_out_of_bound(in_arr: Iterable, values: Iterable, tolerance: float):
    values = [values] if not isinstance(values, Iterable) else values
    top, bottom = max(in_arr) + tolerance, min(in_arr) - tolerance
    return [False if bottom <= val <= top else True for val in values]


# pay attention to those methods bellow
# in the future for different atm model might happen that function won't be valid anymore
def validate_temperature(temperature: Iterable, atlas: str):
    atlas = validated_atlas(atlas)
    allowed = sorted(atm_file_prefix_to_quantity_list("temperature", atlas))
    invalid = any([True if (allowed[-1] < t or allowed[-1] < t) else False for t in temperature])
    if invalid:
        raise ValueError("any temperature in system atmosphere is out of bound; "
                         "it is ussualy caused by invalid physical model")
    return True


def validate_metallicity(metallicity: Iterable, atlas: str):
    out_of_bound_tol = 0.1  # how far `out of bound` can any value of metallicity runs
    atlas = validated_atlas(atlas)
    allowed = sorted(atm_file_prefix_to_quantity_list("metallicity", atlas))
    out_of_bound = is_out_of_bound(allowed, metallicity, out_of_bound_tol)
    if any(out_of_bound):
        raise ValueError("any metallicity in system atmosphere is out of bound, allowed values "
                         "are in range <{}, {}>; it is ussualy caused by invalid physical model"
                         "".format(min(allowed) - out_of_bound_tol, max(allowed) + out_of_bound_tol))
    return True


def validate_logg(logg, atlas: str):
    # not implemented, naive implementation is uselles
    # proper `like` implementaion is _validate_logg
    pass


def _validate_logg(temperature, logg, atlas):
    # it has a different name beacuse there is a different interface
    validation_hypertable = build_atm_validation_hypertable(atlas)
    allowed = sorted(atm_file_prefix_to_quantity_list("temperature", atlas))

    invalid = [
        is_out_of_bound(validation_hypertable[
                            str(int(utils.find_nearest_value(allowed, t)[0]))
                        ]["gravity"], [g], 0.1)[0] for t, g in zip(temperature, logg)]
    if any(invalid):
        raise ValueError("any gravity (logg) in system atmosphere is out of bound; "
                         "it is ussualy caused by invalid physical model")
    return True


def validate_atm(temperature, logg, metallicity, atlas):
    metallicity = [metallicity] * len(temperature) if not isinstance(metallicity, Iterable) else metallicity
    validate_temperature(temperature, atlas)
    validate_metallicity(metallicity, atlas)
    _validate_logg(temperature, logg, atlas)
    return True


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

    validate_atm(_temperature, _logg, _metallicity, "ck04")








    # atm_containers = NearestAtm.nearest_atm_tables(_temperature, _logg, _metallicity, "ck")
    # i = compute_integral_si_intensity_from_atm_data_containers(atm_containers)

    # print(nearest_atm_files_list(temperature=[10005], logg=[0.1], metallicity=[0.1], atlas="ck"))

    # find nearest loggs
    # find nearest [M/H]
    # find surounded Ts
    # validity check
    # get relevant atm tables
    # interpolate T - passbadn restriction or not (switch)
    # return interpolated containers
