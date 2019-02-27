import itertools
import logging
import os
from collections import Iterable
from queue import Queue
from threading import Thread

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


def multithread_atm_tables_reader(path_queue: Queue, error_queue: Queue, result_queue: Queue):
    while True:
        args = path_queue.get(timeout=1)

        if args == "TERMINATOR":
            break
        if not error_queue.empty():
            break
        index, file_path = args
        if file_path is None:
            # consider put here an empty container
            result_queue.put((index, None))
            continue
        try:
            t, l, m = parse_domain_quantities_from_atm_table_filename(os.path.basename(file_path))
            atm_container = AtmDataContainer(pd.read_csv(file_path), t, l, m)
            result_queue.put((index, atm_container))
        except Exception as we:
            error_queue.put(we)
            break


def multithread_atm_tables_reader_runner(fpaths):
    n_threads = config.NUMBER_OF_THREADS

    path_queue = Queue(maxsize=len(fpaths) + n_threads)
    result_queue = Queue()
    error_queue = Queue()

    threads = list()
    try:
        for index, fpath in enumerate(fpaths):
            if isinstance(fpath, str):
                if not os.path.isfile(fpath):
                    raise FileNotFoundError("file {} doesn't exist. it seems your model "
                                            "could be not physical".format(fpath))
            path_queue.put((index, fpath))

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
    return result_queue


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
    def atm_files(temperature, logg, metallicity, atlas):
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
    def atm_tables(fpaths):
        """
        returns spectrum profile for the atmospheric model that is the closest to the given parameters `temperature`, `logg`
        and `metallicity`

        :return:
        """
        result_queue = multithread_atm_tables_reader_runner(fpaths)
        models = [qval for qval in utils.IterableQueue(result_queue)]
        return models

    @staticmethod
    def radiance(temperature, logg, metallicity, atlas, constraint):
        atlas = validated_atlas(atlas)
        # todo: need full implementaion
        atm_files = NearestAtm.atm_files(temperature, logg, metallicity, atlas)
        atm_tables = NearestAtm.atm_tables(atm_files)
        return compute_integral_si_intensity_from_atm_data_containers(atm_tables)


class NaiveInterpolatedAtm(object):
    @staticmethod
    def radiance(temperature: Iterable, logg: Iterable, metallicity: float, atlas: str, **kwargs):
        # validate_atm(temperature, logg, metallicity, atlas)
        atm_files = NaiveInterpolatedAtm.atm_files(temperature, logg, metallicity, atlas)
        atm_tables = NaiveInterpolatedAtm.atm_tables(atm_files)
        localized_atm = NaiveInterpolatedAtm.interpolate(atm_tables,
                                                         **dict(left_bandwidth=kwargs['left_bandwidth'],
                                                                right_bandwidth=kwargs['right_bandwidth'],
                                                                temperature=temperature))
        return compute_integral_si_intensity_from_atm_data_containers(localized_atm)

    @staticmethod
    def strip_atm_container_by_bandwidth(atm_container, left_bandwidth, right_bandwidth):
        """

        :param atm_container:
        :param left_bandwidth:
        :param right_bandwidth:
        :return:
        """
        if atm_container is not None:
            # todo: it assumes that wave unit is angstrom and this is reason, why there is bandwidth multiplied by 10
            # todo: fix it to general values based on atm container units info
            valid_indices = list(
                atm_container.model.index[
                    atm_container.model["wave"].between(left_bandwidth * 10, right_bandwidth * 10, inclusive=True)
                ])
            left_extention_index = valid_indices[0] - 1 if valid_indices[0] > 1 else 0
            right_extention_index = valid_indices[-1] + 1 if valid_indices[-1] > 1 else valid_indices[-1]
            atm_container.model = atm_container.model.iloc[
                sorted(valid_indices + [left_extention_index] + [right_extention_index])
            ]

    @staticmethod
    def compute_interpolation_weights(temperatures, top_atm_containers, bottom_atm_containers):
        top_temperatures = np.array([a.temperature for a in top_atm_containers])
        bottom_temperatures = np.array([a.temperature if a is not None else 0.0 for a in bottom_atm_containers])
        return (temperatures - bottom_temperatures) / (top_temperatures - bottom_temperatures)

    @staticmethod
    def interpolate(atm_tables, **kwargs):
        temperature = kwargs.pop("temperature")
        bottom_atm, top_atm = atm_tables[:len(atm_tables) // 2], atm_tables[len(atm_tables) // 2:]
        left_bandwidth, right_bandwidth = kwargs.pop("left_bandwidth"), kwargs.pop("right_bandwidth")

        # no set needed, values are mutable, and all are modified in ``strip_atm_container_by_bandwidth`` method
        [NaiveInterpolatedAtm.strip_atm_container_by_bandwidth(a, left_bandwidth, right_bandwidth) for a in bottom_atm]
        [NaiveInterpolatedAtm.strip_atm_container_by_bandwidth(a, left_bandwidth, right_bandwidth) for a in top_atm]
        interpolation_weights = NaiveInterpolatedAtm.compute_interpolation_weights(temperature, top_atm, bottom_atm)

        # todo: continue here

        print(interpolation_weights)

        return top_atm

    @staticmethod
    def atm_tables(fpaths):
        result_queue = multithread_atm_tables_reader_runner(fpaths)
        models = [qval for qval in utils.IterableQueue(result_queue)]
        models = [val[1] for val in sorted(models, key=lambda x: x[0])]
        return models

    @staticmethod
    def atm_files(temperature: Iterable, logg: Iterable, metallicity: float, atlas: str):
        atlas = validated_atlas(atlas)

        g_array = atm_file_prefix_to_quantity_list("gravity", atlas)
        m_array = atm_file_prefix_to_quantity_list("metallicity", atlas)
        t_array = atm_file_prefix_to_quantity_list("temperature", atlas)

        g = [utils.find_nearest_value(g_array, _logg)[0] for _logg in logg]
        m = utils.find_nearest_value(m_array, metallicity)[0]
        t = [_t if len(_t) == 2 else [np.nan] + _t
             for _t in [utils.find_surrounded(t_array, _temp) for _temp in temperature]]
        t = np.array(t).T

        domain_df = pd.DataFrame({
            "temp": list(t[0]) + list(t[1]),
            "logg": list(g) + list(g),
            "mh": [m] * len(g) * 2
        })

        directory = get_atm_directory(m, atlas)
        # in case when temperature is same as one of temperatures on grid, sourrounded value is only one number
        # and what we have to do is just read a atm table and do not any interpolation
        fnames = str(atlas) + \
                 domain_df["mh"].apply(lambda x: utils.numeric_metallicity_to_string(x)) + "_" + \
                 domain_df["temp"].apply(lambda x: str(int(x) if not np.isnan(x) else '__NaN__')) + "_" + \
                 domain_df["logg"].apply(lambda x: utils.numeric_logg_to_string(x))
        return [
            path if '__NaN__' not in path
            else None
            for path in
            list(os.path.join(str(ATLAS_TO_BASE_DIR[atlas]), str(directory)) + os.path.sep + fnames + ".csv")
        ]


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
                         "it is usually caused by invalid physical model")
    return True


def validate_metallicity(metallicity: Iterable, atlas: str):
    out_of_bound_tol = 0.1  # how far `out of bound` can any value of metallicity runs
    atlas = validated_atlas(atlas)
    allowed = sorted(atm_file_prefix_to_quantity_list("metallicity", atlas))
    out_of_bound = is_out_of_bound(allowed, metallicity, out_of_bound_tol)
    if any(out_of_bound):
        raise ValueError("any metallicity in system atmosphere is out of bound, allowed values "
                         "are in range <{}, {}>; it is usually caused by invalid physical model"
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
                         "it is usually caused by invalid physical model")
    return True


def validate_atm(temperature, logg, metallicity, atlas):
    metallicity = [metallicity] * len(temperature) if not isinstance(metallicity, Iterable) else metallicity
    validate_temperature(temperature, atlas)
    validate_metallicity(metallicity, atlas)
    _validate_logg(temperature, logg, atlas)
    return True


if __name__ == "__main__":
    pass
