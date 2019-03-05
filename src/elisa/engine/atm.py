import itertools
import logging
import os
from collections import Iterable
from queue import Queue
from threading import Thread

import numpy as np
import pandas as pd
from copy import copy
from scipy import integrate, interpolate

from elisa.conf import config
from elisa.conf.config import ATM_MODEL_DATAFRAME_FLUX, ATM_MODEL_DATAFRAME_WAVE
from elisa.conf.config import PASSBAND_DATAFRAME_WAVE, PASSBAND_DATAFRAME_THROUGHPUT
from elisa.engine import utils, const

config.set_up_logging()
logger = logging.getLogger("atm")

# * 1e-7 * 1e4 * 1e10 * (1.0/np.pi)

ATLAS_TO_ATM_FILE_PREFIX = {
    "castelli": "ck",
    "castelli-kurucz": "ck",
    "ck": "ck",
    "ck04": "ck",
    "kurucz": "k",
    "k": "k",
    "k93": "k"
}

ATLAS_TO_BASE_DIR = {
    "castelli": config.CK04_ATM_TABLES,
    "castelli-kurucz": config.CK04_ATM_TABLES,
    "ck": config.CK04_ATM_TABLES,
    "ck04": config.CK04_ATM_TABLES,
    "kurucz": config.K93_ATM_TABLES,
    "k": config.K93_ATM_TABLES,
    "k93": config.K93_ATM_TABLES
}

ATM_DOMAIN_QUANTITY_TO_VARIABLE_SUFFIX = {
    "temperature": "TEMPERATURE_LIST_ATM",
    "gravity": "GRAVITY_LIST_ATM",
    "metallicity": "METALLICITY_LIST_ATM"
}


class AtmDataContainer(object):
    def __init__(self, model: pd.DataFrame, temperature: float, logg: float, metallicity: float):
        self._model = None
        self.temperature = temperature
        self.logg = logg
        self.metallicity = metallicity
        self.flux_unit = "flam"
        self.wave_unit = "angstrom"
        # in case this np.pi will stay here, there will be rendundant multiplication in intensity integration
        self.flux_to_si_mult = 1e-7 * 1e4 * 1e10  # * (1.0/np.pi)
        self.wave_to_si_mult = 1e-10
        self.left_bandwidth = None
        self.right_bandwidth = None

        setattr(self, 'model', model)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, df: pd.DataFrame):
        self._model = df
        self.left_bandwidth = min(df[ATM_MODEL_DATAFRAME_WAVE])
        self.right_bandwidth = max(df[ATM_MODEL_DATAFRAME_WAVE])


class IntensityContainer(object):
    def __init__(self, intensity, temperature, logg, metallicity):
        self.intensity = intensity
        self.temperature = temperature
        self.logg = logg
        self.metallicity = metallicity


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
    def radiance(temperature, logg, metallicity, atlas, **kwargs):
        atlas = validated_atlas(atlas)
        validate_atm(temperature, logg, metallicity, atlas)
        atm_files = NearestAtm.atm_files(temperature, logg, metallicity, atlas)
        atm_containers = read_atm_tables(atm_files)
        passbanded_atm_containers = apply_passband(atm_containers, kwargs["passband"])
        return compute_integral_si_intensity_from_passbanded_dict(passbanded_atm_containers)


class NaiveInterpolatedAtm(object):
    @staticmethod
    def radiance(temperature: list, logg: list, metallicity: float, atlas: str, **kwargs):
        """
        compute radiance for given atmospheric parametres with regards to given passbands

        :param temperature: list
        :param logg: list
        :param metallicity: float
        :param atlas: str
        :param kwargs:
        :**kwargs options**:
                * **left_bandwidth** * -- float; maximal allowed wavelength from left
                * **right_bandwidth** * -- float; maximal allowed wavelength from right
                * **passband** * -- elisa.observer.observer.PassbandContainer
        :return: list
        """
        # fixme: uncomment following line
        # validate_atm(temperature, logg, metallicity, atlas)
        atm_files = NaiveInterpolatedAtm.atm_files(temperature, logg, metallicity, atlas)
        atm_containers = read_atm_tables(atm_files)
        localized_atm_containers = NaiveInterpolatedAtm.interpolate(
            atm_containers,
            **dict(left_bandwidth=kwargs['left_bandwidth'],
                   right_bandwidth=kwargs['right_bandwidth'],
                   temperature=temperature,
                   logg=logg,
                   metallicity=metallicity)
        )
        passbanded_atm_containers = apply_passband(localized_atm_containers, kwargs["passband"])
        return compute_integral_si_intensity_from_passbanded_dict(passbanded_atm_containers)

    @staticmethod
    def compute_interpolation_weights(temperatures: list, top_atm_containers: list, bottom_atm_containers: list):
        top_temperatures = np.array([a.temperature for a in top_atm_containers])
        bottom_temperatures = np.array([a.temperature if a is not None else 0.0 for a in bottom_atm_containers])
        return (temperatures - bottom_temperatures) / (top_temperatures - bottom_temperatures)

    @staticmethod
    def compute_unknown_intensity(weight, top_atm_container, bottom_atm_container):
        """
        Depends on weight will compute (interpolate) intensities from surounded intensities
        related to given temperature.
        In case that top and bottom atmosphere model are not on the same wavelength then as base is taken a wavelength
        from top. Get values on the same wavelength as in case of top atmosphere, akima 1d interpolation is used
        and intensities (fluxes) are artificialy computed.


        :param weight: Iterable of float`s
        :param top_atm_container: AtmDataContainer
        :param bottom_atm_container: AtmDataContainer
        :return: tuple of list (flux, wave)
        """
        if bottom_atm_container is not None:
            do_akima = False \
                if np.all(
                np.array(top_atm_container.model[ATM_MODEL_DATAFRAME_WAVE], dtype="float") ==
                np.array(bottom_atm_container.model[ATM_MODEL_DATAFRAME_WAVE], dtype="float")) \
                else True
        else:
            return top_atm_container.model[ATM_MODEL_DATAFRAME_FLUX], top_atm_container.model[ATM_MODEL_DATAFRAME_WAVE]

        if do_akima:
            wavelength = top_atm_container.model[ATM_MODEL_DATAFRAME_WAVE]
            akima = interpolate.Akima1DInterpolator(bottom_atm_container.model[ATM_MODEL_DATAFRAME_WAVE],
                                                    bottom_atm_container.model[ATM_MODEL_DATAFRAME_FLUX])
            bottom_atm_container.model = pd.DataFrame({
                ATM_MODEL_DATAFRAME_FLUX: np.array(akima(wavelength)),
                ATM_MODEL_DATAFRAME_WAVE: np.array(wavelength)
            })
            bottom_atm_container.model.fillna(0.0, inplace=True)

        # reset index is neccessary; otherwise add/mult/... method od DataFrame
        # leads to nan if left and right frame differ in indices
        top_atm_container.model.reset_index(drop=True, inplace=True)
        bottom_atm_container.model.reset_index(drop=True, inplace=True)

        intensity = weight * (
            top_atm_container.model[ATM_MODEL_DATAFRAME_FLUX] - bottom_atm_container.model[ATM_MODEL_DATAFRAME_FLUX]
        ) + bottom_atm_container.model[ATM_MODEL_DATAFRAME_FLUX]

        return intensity, top_atm_container.model[ATM_MODEL_DATAFRAME_WAVE]

    @staticmethod
    def interpolate(atm_tables, **kwargs):
        """
        for given `on grid` tables of stellar atmospheres stored in `atm_tables` list will compute atmospheres
        for given parametres (temperature, logg, metallicity)

        atm_tables contain extended list of AtmDataContainer`s;
        first part (half) of atm_tables contain bottom atmospheres related to given temperature and second half of list
        contain atmospheres from top of temperature.
        In case that temperature match exactly atmospheric model, that model is stored in `top` atm_tables and in
        bottom is stored None value

        e.g.

        temperature = [7825.66, 4500, 19874.85]
        atm_tables = [<t1 - 7750>, None, <t3 - 19000>, <t4 - 8000>, <t5 - 4500>, <t6 - 20000>]

        bottom: <t1 - 7750>, None, <t3 - 19000>
        top: <t4 - 8000>, <t5 - 4500>, <t6 - 20000>

        :param atm_tables: list of AtmDataContainer`s
        :param kwargs:
        :**kwargs options**:
                * **temperature** * -- Iterable
                * **logg** * -- Iterable
                * **metallicity** * -- float
                * **left_bandwidth** * -- float; maximal allowed wavelength from left
                * **right_bandwidth** * -- float; maximal allowed wavelength from right
        :return: list of AtmDataContainer`s
        """

        temperature = kwargs.pop("temperature")
        logg = kwargs.pop("logg")
        metallicity = kwargs.pop("metallicity")

        bottom_atm, top_atm = atm_tables[:len(atm_tables) // 2], atm_tables[len(atm_tables) // 2:]
        left_bandwidth, right_bandwidth = kwargs.pop("left_bandwidth"), kwargs.pop("right_bandwidth")

        [strip_atm_container_by_bandwidth(a, left_bandwidth, right_bandwidth, inplace=True) for a in top_atm]
        # strip bottom container by top container bandtwidth to avoid to get NaN in akima interpolation
        # in ``compute_unknown_intensity`` based on top atm container wavelength
        [strip_atm_container_by_bandwidth(a, b.left_bandwidth, b.right_bandwidth, inplace=True)
         for a, b in zip(bottom_atm, top_atm)]
        interpolation_weights = NaiveInterpolatedAtm.compute_interpolation_weights(temperature, top_atm, bottom_atm)
        interpolated_atm_containers = list()

        for weight, t, g, bottom, top in zip(interpolation_weights, temperature, logg, bottom_atm, top_atm):
            intensity, wavelength = NaiveInterpolatedAtm.compute_unknown_intensity(weight, top, bottom)
            interpolated_atm_containers.append(
                AtmDataContainer(
                    model=pd.DataFrame(
                        {
                            ATM_MODEL_DATAFRAME_FLUX: np.array(intensity),
                            ATM_MODEL_DATAFRAME_WAVE: np.array(wavelength)
                        }
                    ),
                    temperature=t,
                    logg=g,
                    metallicity=metallicity
                )
            )
        return interpolated_atm_containers

    @staticmethod
    def atm_tables(fpaths):
        """
        read atmosphere tables as pandas.DataFrame`s

        :param fpaths: Iterable, list of paths to desired atm csv files
        :return: list of pandas.DataFrame`s
        """
        result_queue = multithread_atm_tables_reader_runner(fpaths)
        models = [qval for qval in utils.IterableQueue(result_queue)]
        models = [val[1] for val in sorted(models, key=lambda x: x[0])]
        return models

    @staticmethod
    def atm_files(temperature: Iterable, logg: Iterable, metallicity: float, atlas: str):
        """
        For given parameters will find out related atm csv tables and return list of paths to this csv files

        :param temperature: Iterable
        :param logg: Iterable
        :param metallicity: float
        :param atlas: str
        :return: list; list of str
        """
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


def strip_atm_container_by_bandwidth(atm_container, left_bandwidth, right_bandwidth, inplace=False):
    """
    strip atmosphere container model by given bandwidth (add +/- 1 value behind boundary)

    :param inplace: bool
    :param atm_container: AtmDataContainer
    :param left_bandwidth: float
    :param right_bandwidth: float
    :return: AtmDataContainer
    """
    if atm_container is not None:
        valid_indices = list(
            atm_container.model.index[
                atm_container.model[ATM_MODEL_DATAFRAME_WAVE].between(left_bandwidth, right_bandwidth, inclusive=True)
            ])
        left_extention_index = valid_indices[0] - 1 if valid_indices[0] > 1 else 0
        right_extention_index = valid_indices[-1] + 1 \
            if valid_indices[-1] < atm_container.model.last_valid_index() else valid_indices[-1]

        atm_container = atm_container if inplace else copy(atm_container)
        atm_container.model = atm_container.model.iloc[
            sorted(valid_indices + [left_extention_index] + [right_extention_index])
        ]
        atm_container.model = atm_container.model.drop_duplicates(ATM_MODEL_DATAFRAME_WAVE)
        return atm_container


def apply_passband(atm_containers: list, passband: dict):
    passbanded_atm_containers = dict()
    for band, band_container in passband.items():
        passbanded_atm_containers[band] = list()
        for atm_container in atm_containers:
            # strip atm container on passband bandwidth (reason to do it is, that container
            # is stripped on maximal bandwidth defined by all bands, not just by given single band)
            atm_container = strip_atm_container_by_bandwidth(
                atm_container=atm_container,
                left_bandwidth=band_container.left_bandwidth,
                right_bandwidth=band_container.right_bandwidth,
                inplace=False
            )
            # found passband throughput on atm defined wavelength
            passband_df = pd.DataFrame(
                {
                    PASSBAND_DATAFRAME_THROUGHPUT: band_container.akima(atm_container.model[ATM_MODEL_DATAFRAME_WAVE]),
                    PASSBAND_DATAFRAME_WAVE: atm_container.model[ATM_MODEL_DATAFRAME_WAVE]
                }
            )
            passband_df.fillna(0.0, inplace=True)
            atm_container.model[ATM_MODEL_DATAFRAME_FLUX] *= passband_df[PASSBAND_DATAFRAME_THROUGHPUT]
            passbanded_atm_containers[band].append(atm_container)
    return passbanded_atm_containers


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


def compute_integral_si_intensity_from_passbanded_dict(passbaned_dict: dict):
    return {
        band: compute_integral_si_intensity_from_atm_data_containers(passbanded_atm)
        for band, passbanded_atm in passbaned_dict.items()
    }


def compute_integral_si_intensity_from_atm_data_containers(atm_data_containers: list):
    """
    Returns intensity from given atmosphere models.
    If models are already strip by passband, result is also striped

    :param atm_data_containers: Iterable; list of AtmDataContainer`s
    :return: list; integrated `flux` from each AtmDataContainer on `wave` in given container
    """
    # todo: implement intensity contauner instead of simple float values

    return [
        IntensityContainer(
            intensity=np.pi * integrate.simps(adc.model[ATM_MODEL_DATAFRAME_FLUX] * adc.flux_to_si_mult,
                                              adc.model[ATM_MODEL_DATAFRAME_WAVE] * adc.wave_to_si_mult),
            temperature=adc.temperature,
            logg=adc.logg,
            metallicity=adc.metallicity
        )
        for adc in atm_data_containers
    ]


def read_atm_tables(fpaths):
    """
    returns spectrum profile for the atmospheric model that is the closest to the given parameters `temperature`, `logg`
    and `metallicity`

    :return:
    """
    result_queue = multithread_atm_tables_reader_runner(fpaths)
    models = [qval for qval in utils.IterableQueue(result_queue)]
    models = [val[1] for val in sorted(models, key=lambda x: x[0])]
    return models


if __name__ == "__main__":
    pass
