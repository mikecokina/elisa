import itertools
import json
import os
import sys
import warnings
import numpy as np
import pandas as pd

from queue import Queue
from threading import Thread
from typing import Iterable
from copy import deepcopy

from .logger import getLogger
from .base.error import (
    AtmosphereError,
    MetallicityError,
    TemperatureError,
    GravityError, ElisaError)
from scipy import (
    integrate,
    interpolate
)
from . import settings
from . import (
    umpy as up,
    utils,
    const,
    ld,
)
from . buffer import buffer
from . tensor.etensor import Tensor

logger = getLogger(__name__)


# * 1e-7 * 1e4 * 1e10 * (1.0/const.PI)

class AtmModel(object):
    def __init__(self, flux, wavelength):
        self.flux: np.array = flux
        self.wavelength: np.array = wavelength

    def _empty(self):
        return len(self.wavelength) == 0

    @property
    def empty(self):
        return self._empty()

    @classmethod
    def from_dataframe(cls, df):
        return cls(flux=np.array(df[settings.ATM_MODEL_DATAFRAME_FLUX], dtype=float),
                   wavelength=np.array(df[settings.ATM_MODEL_DATAFRAME_WAVE], dtype=float))

    def to_dataframe(self):
        return pd.DataFrame(
            {
                settings.ATM_MODEL_DATAFRAME_FLUX: self.flux,
                settings.ATM_MODEL_DATAFRAME_WAVE: self.wavelength
            }
        )

    def last_valid_index(self):
        return len(self.flux)

    def __getitem__(self, item):
        return AtmModel(flux=self.flux[item], wavelength=self.wavelength[item])

    def __len__(self):
        return len(self.wavelength)


class AtmDataContainer(object):
    def __init__(self, model, temperature, log_g, metallicity, fpath=''):
        self._model = AtmModel(flux=None, wavelength=None)
        self.temperature = temperature
        self.log_g = log_g
        self.metallicity = metallicity
        self.flux_unit = "flam"
        self.wave_unit = "angstrom"
        # in case this const.PI will stay here, there will be rendundant multiplication in intensity integration
        # flam = erg * s-1* cm-2 *A-1 = (10-7 * J) * s-1 * (10-2 * m)-2 * (10-10 * m)-1 =
        #        10-7 * 10**4 * 10**10 * J * s-1 * m-3
        self.flux_to_si_mult = 1e7  # * (1.0/const.PI)
        self.wave_to_si_mult = 1e-10
        self.left_bandwidth = np.nan
        self.right_bandwidth = np.nan
        self.fpath = fpath

        setattr(self, 'model', model)

    def is_empty(self):
        """
        Find out wheter model container which carries of atmospheric
        model AmtDatContainer instance is empty.

        :return: bool;
        """
        return self._model.empty

    @property
    def model(self):
        """
        Return atmospheric model instance.

        :return: elisa.atm.AtmModel;
        """
        return self._model

    @model.setter
    def model(self, data):
        """
        Setup model container which carries DataFrame of atmospheric model
        and left and right bandwidth of such container as well.

        :param data: Union[pandasDataFrame, elisa.atm.AtmModel];
        """
        self._model = AtmModel.from_dataframe(data) if isinstance(data, pd.DataFrame) else data
        self.left_bandwidth = self._model.wavelength.min()
        self.right_bandwidth = self._model.wavelength.max()


class IntensityContainer(object):
    """
    Intended to keep information about integrated radiance for given params.
    """

    def __init__(self, intensity, temperature, log_g, metallicity):
        """
        Initialise container with given parametres.

        :param intensity: float; integrated radiance
        :param temperature: float;
        :param log_g: float;
        :param metallicity: float;
        """
        self.intensity = intensity
        self.temperature = temperature
        self.log_g = log_g
        self.metallicity = metallicity


class NaiveInterpolatedAtm(object):
    @staticmethod
    def radiance(temperature, log_g, metallicity, atlas, **kwargs):
        """
        Compute radiance for given atmospheric parametres and given passbands.

        :param temperature: numpy.array;
        :param log_g: numpy.array;
        :param metallicity: float;
        :param atlas: str;
        :param kwargs:
        :**kwargs options**:
            * **left_bandwidth** * -- float; maximal allowed wavelength from left (Angstrom)
            * **right_bandwidth** * -- float; maximal allowed wavelength from right (Angstrom)
            * **passband** * -- Dict[str, elisa.observer.observer.PassbandContainer]
        :return: List;
        """
        if validated_atlas(atlas) == "bb":
            return NaiveInterpolatedAtm.black_body_radiance(temperature, **kwargs)
        return NaiveInterpolatedAtm.atlas_radiance(temperature, log_g, metallicity, atlas, **kwargs)

    @staticmethod
    def black_body_radiance(temperature, **kwargs):
        """
        Compute integrated flux based on values obtained from Planck Function (for purpose of black_body atmosphere).

        :param temperature: numpy.array;
        :param kwargs: Dict;
        :**kwargs options**:
            * **passband** * -- Dict[str, elisa.observer.observer.PassbandContainer]
        :return: Dict[str, float];
        """
        # setup multiplicators to convert quantities to SI
        flux_mult, wave_mult = const.PI, 1e-10
        # obtain localized atmospheres in matrix
        localized_atms = NaiveInterpolatedAtm.arange_black_body_localized_atms(temperature, kwargs["passband"])
        # integrate flux
        return compute_normal_radiances(localized_atms, flux_mult=flux_mult, wave_mult=wave_mult)

    @staticmethod
    def arange_black_body_localized_atms(temperature, passband_containers):
        """
        The function generates atmosphere models based on Planck Function.
        The all models are sitting on temperatures given by surface elements.

        :param temperature: numpy.array;
        :param passband_containers: elisa.observer.observer.PassbandContainer;
        :return: Dict[str, numpy.array];
        """
        localized_atms = dict()
        standard_wavelength = get_standard_wavelengths()

        # build temperature mask and avoid repeative computation
        # temperature values where decimal points are basicaly useless
        temperature = np.round(temperature, 0)
        temperature, reverse_map = np.unique(temperature, return_inverse=True)

        for band, pb_container in passband_containers.items():
            # how many wavelengths generate based on standard
            mask = np.logical_and(np.less_equal(standard_wavelength, pb_container.right_bandwidth),
                                  np.greater_equal(standard_wavelength, pb_container.left_bandwidth))
            hm_waves = len(standard_wavelength[mask])
            # wavelenghts in angstrom
            wavelengths = np.sort(np.unique(
                np.concatenate(
                    [np.linspace(pb_container.left_bandwidth, pb_container.right_bandwidth, hm_waves, True),
                     standard_wavelength[mask]])
            ))

            # compute flux in flam, apply passband and replace possible NaNs
            flux = np.nan_to_num([
                pb_container.akima(wavelengths) *
                planck_function(wavelengths * pb_container.wave_to_si_mult, _temperature)
                for _temperature in temperature
            ])
            # sometimes, there are small negative values on the boundwidth boundaries due to akima interpolation
            flux[np.less(flux, 0)] = 0.0
            # broadcast and fill localized atms
            localized_atms[band] = {"flux": flux[reverse_map], "wave": wavelengths}

        return localized_atms

    @staticmethod
    def get_atm_profiles(temperature, log_g, metallicity, atlas, **kwargs):
        """
        Returns atmosphere profiles for given surface parameters.

        :param temperature: Iterable[float];
        :param log_g: Iterable[float];
        :param metallicity: float;
        :param atlas: str; atmosphere model identificator (see settings.ATLAS_TO_ATM_FILE_PREFIX.keys())
        :param kwargs: Dict;
        :return: Tuple[Dict, numpy.float, numpy.float]; atmosphere profiles for each passband, flux multiplicator,
                                                        wave multiplicator;
        """
        l_bandw, r_bandw = kwargs["left_bandwidth"], kwargs["right_bandwidth"]
        passband_containers = kwargs["passband"]
        # related atmospheric files for each face (upper and lower)
        atm_files = NaiveInterpolatedAtm.atm_files(temperature, log_g, metallicity, atlas)
        # find unique atmosphere data files
        unique_atms, containers_map = read_unique_atm_tables(atm_files)
        # get multiplicators to transform containers from any units to si
        flux_mult, wave_mult = find_atm_si_multiplicators(unique_atms)
        # common wavelength coverage of atmosphere models
        # intersection of wavelengths of models
        global_left, global_right = find_global_atm_bandwidth(unique_atms)
        # strip unique atmospheres to passbands coverage
        unique_atms = strip_atm_containers_by_bandwidth(unique_atms, l_bandw, r_bandw,
                                                        global_left=global_left, global_right=global_right)

        # alignement of atmosphere containers wavelengths for convenience
        unique_atms = arange_atm_to_same_wavelength(unique_atms)
        passbanded_atm_containers = apply_passband(unique_atms, passband_containers,
                                                   global_left=global_left, global_right=global_right)

        flux_matrices = remap_passbanded_unique_atms_to_matrix(passbanded_atm_containers, containers_map)
        atm_containers = remap_passbanded_unique_atms_to_origin(passbanded_atm_containers, containers_map)
        localized_atms = NaiveInterpolatedAtm.interpolate_spectra(atm_containers, flux_matrices,
                                                                  temperature=temperature)

        return localized_atms, flux_mult, wave_mult

    @staticmethod
    def atlas_radiance(temperature, log_g, metallicity, atlas, **kwargs):
        """
        Returns normal radiance for given surface parameters.

        :param temperature: Iterable[float];
        :param log_g: Iterable[float];
        :param metallicity: float;
        :param atlas: str; atmosphere model identificator (see settings.ATLAS_TO_ATM_FILE_PREFIX.keys())
        :param kwargs:
        :return: Dict;
        """
        args = temperature, log_g, metallicity, atlas
        localized_atms, flux_mult, wave_mult = NaiveInterpolatedAtm.get_atm_profiles(*args, **kwargs)
        return compute_normal_radiances(localized_atms, flux_mult=flux_mult, wave_mult=wave_mult)

    @staticmethod
    def compute_interpolation_weights(temperatures, top_atm_containers, bottom_atm_containers):
        """
        Compute interpolation weights between two models of atmoshperes.
        Weights are computet as::

            (temperatures^4 - bottom_temperatures^4) / (top_temperatures^4 - bottom_temperatures^4)

        what means we use linear approach.
        If there is np.nan (it cames from same surounded values), such value is replaced with 1.0.
        1.0 is choosen to fit interpolation method and return correct atmosphere.

        :param temperatures: numpy.array[float];
        :param top_atm_containers: numpy.array[elisa.atm.AtmDataContainer];
        :param bottom_atm_containers: numpy.array[elisa.atm.AtmDataContainer];
        :return: numpy.array[float];
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            top_temperatures4 = np.power(np.array([a.temperature for a in top_atm_containers]), 4)
            bottom_temperatures4 = np.power(np.array([a.temperature for a in bottom_atm_containers]), 4)

            result = (np.power(temperatures, 4) - bottom_temperatures4) / (top_temperatures4 - bottom_temperatures4)

            result[up.isnan(result)] = 1.0
            return result

    @staticmethod
    def compute_unknown_intensity_from_surounded_containers(weight, top_atm_container, bottom_atm_container):
        """
        Depends on weight will compute (interpolate) intensities from surounded intensities
        related to given temperature.
        ! Top and bottom atmosphere model are have to be defined in same wavelengths !

        :param weight: Iterable[float];
        :param top_atm_container: elisa.atm.AtmDataContainer;
        :param bottom_atm_container: elisa.atm.AtmDataContainer;
        :return: Tuple[numpy.array, numpy.array]; (flux, wave);
        """
        if bottom_atm_container is None:
            return top_atm_container.model.flux, top_atm_container.model.wavelength

        # reset index is neccessary; otherwise add/mult/... method od DataFrame
        # leads to nan if left and right frame differ in indices
        top_atm_container.model.reset_index(drop=True, inplace=True)
        bottom_atm_container.model.reset_index(drop=True, inplace=True)

        intensity = weight * (
                top_atm_container.model.flux - bottom_atm_container.model.flux) + bottom_atm_container.model.flux

        return intensity, top_atm_container.model.wavelength

    @staticmethod
    def compute_unknown_intensity_from_surounded_flux_matrices(weights, top_flux_matrix, bottom_flux_matrix):
        weights = Tensor(weights)
        top_flux_matrix = Tensor(top_flux_matrix)
        bottom_flux_matrix = Tensor(bottom_flux_matrix)
        result = (weights * (top_flux_matrix.T - bottom_flux_matrix.T) + bottom_flux_matrix.T).T
        return result.to_ndarray()

    @staticmethod
    def interpolate_spectra(passbanded_atm_containers, flux_matrices, temperature):
        """
        From supplied elisa.atm.AtmDataContainer's, `flux_matrices` and `temeprature`.
        Interpolation is computed in vector form::

            (weights * (top_flux_matrix.T - bottom_flux_matrix.T) + bottom_flux_matrix.T).T

        where `top_flux_matrix` and `bottom_flux_matrix`, are entire matrix where rows are represented by fluxes.
        It also means, to be able do such interpolation, fluxes have to be on same wavelengths for each row.

        :param flux_matrices: Dict[str, numpy.array];

        ::

            {"passband": numpy.array (matrix)}

        :param passbanded_atm_containers: Dict[str, elisa.atm.AtmDataContainers];
        :param temperature: numpy.array[float];
        :return: Dict[str, numpy.array];
        """

        interp_band = dict()
        for band, flux_matrix in flux_matrices.items():
            band_atm = passbanded_atm_containers[band]
            bottom_flux, top_flux = flux_matrix[:len(flux_matrix) // 2], flux_matrix[len(flux_matrix) // 2:]
            bottom_atm, top_atm = band_atm[:len(band_atm) // 2], band_atm[len(band_atm) // 2:]

            logger.debug(f"computing atmosphere interpolation weights for band: {band}")
            interpolation_weights = NaiveInterpolatedAtm.compute_interpolation_weights(temperature, top_atm, bottom_atm)
            flux = NaiveInterpolatedAtm.compute_unknown_intensity_from_surounded_flux_matrices(
                interpolation_weights, top_flux, bottom_flux
            )
            interp_band[band] = {
                settings.ATM_MODEL_DATAFRAME_FLUX: flux,
                settings.ATM_MODEL_DATAFRAME_WAVE: find_atm_defined_wavelength(top_atm)
            }
        return interp_band

    @staticmethod
    def atm_files(temperature, log_g, metallicity, atlas):
        """
        Find out related atmospheric csv tables and return list of paths to them.

        :param temperature: Iterable[float];
        :param log_g: Iterable[float];
        :param metallicity: float;
        :param atlas: str; atmosphere model identificator (see settings.ATLAS_TO_ATM_FILE_PREFIX.keys())
        :return: List[str];
        """
        atlas = validated_atlas(atlas)
        log_g = utils.convert_gravity_acceleration_array(log_g, "log_cgs")

        g_array = np.array(atm_file_prefix_to_quantity_list("gravity", atlas))
        m_array = np.array(atm_file_prefix_to_quantity_list("metallicity", atlas))
        t_array = np.array(atm_file_prefix_to_quantity_list("temperature", atlas))

        g = utils.find_nearest_value_as_matrix(g_array, log_g)[0]
        m = utils.find_nearest_value_as_matrix(m_array, metallicity)[0][0]
        t = utils.find_surrounded_as_matrix(t_array, temperature)

        domain_df = pd.DataFrame({
            "temp": t.flatten('F'),
            "log_g": np.tile(g, 2),
            "mh": np.repeat(m, len(g) * 2)
        })
        directory = get_atm_directory(m, atlas)
        mh_name = domain_df["mh"].apply(lambda x: utils.numeric_metallicity_to_string(x))
        temp_name = domain_df["temp"].apply(lambda x: str(int(x)))
        log_g_name = domain_df["log_g"].apply(lambda x: utils.numeric_logg_to_string(x))
        fnames = str(atlas) + mh_name + "_" + temp_name + "_" + log_g_name

        return list(
            os.path.join(str(settings.ATLAS_TO_BASE_DIR[atlas]), str(directory)) + os.path.sep + fnames + ".csv"
        )


def arange_atm_to_same_wavelength(atm_containers):
    """
    Function aligns all atmosphere profiles to the same wavelengths.

    :param atm_containers: Iterable[elisa.atm.AtmDataContainer]; atmosphere containers which
                           wavelengths should be aligned
    :return: Iterable[elisa.atm.AtmDataContainer]; wavelength aligned atmospheric containers
    """

    wavelengths = np.unique(np.array([atm.model.wavelength for atm in atm_containers]).flatten())
    wavelengths.sort()
    result = list()

    # this code checks if the containers are already alligned
    s_size = sys.maxsize
    for atm in atm_containers:
        s_size = len(atm.model) if len(atm.model) < s_size else s_size

    # if yes, interpolation is unnecessary
    if s_size == len(wavelengths):
        return atm_containers

    # otherwise interpolation is utilized
    for atm in atm_containers:
        i = interpolate.Akima1DInterpolator(atm.model.wavelength, atm.model.flux)
        atm.model = AtmModel(wavelength=wavelengths, flux=np.nan_to_num(i(wavelengths)))
        result.append(atm)
    return result


def strip_atm_containers_by_bandwidth(atm_containers, left_bandwidth, right_bandwidth, **kwargs):
    """
    Strip all loaded atm models to common wavelength coverage.

    :param atm_containers: List[elisa.atm.AtmDataContainer];
    :param left_bandwidth: float;
    :param right_bandwidth: float;
    :param kwargs:
    :**kwargs options**:
        * **global_left** * -- float; global wavelength from left where flux for all supllied atmposhperes exist
        * **global_right** * -- float; global wavelength from right where flux for all supllied atmposhperes exist
    :return: List[elisa.atm.AtmDataContainer]
    """
    return [strip_atm_container_by_bandwidth(atm_container, left_bandwidth, right_bandwidth, **kwargs)
            for atm_container in atm_containers]


def strip_atm_container_by_bandwidth(atm_container, left_bandwidth, right_bandwidth, **kwargs):
    """
    Strip atmosphere container model by given bandwidth.
    Usually is model in container defined somewhere in between of left and right bandwidth, never exactly in such
    wavelength. To strip container exactly on bandwidth wavelength, interpolation has to be done. In case, when
    model of any atmosphere has smaller real bandwidth, than bandwidth defined by arguments `right_bandwidth` and
    `left_bandwidth` (it happens in case of bolometric passband), global bandwidth of given atmospheres is used.
    Right gloal bandwidth is obtained as min of all maximal wavelengts from all models and left is max of all mins.


    :param atm_container: elisa.atm.AtmDataContainer;
    :param left_bandwidth: float;
    :param right_bandwidth: float;
    :param kwargs:
    :**kwargs options**:
        * **global_left** * -- float; global wavelength from left where flux for all supllied atmposhperes exist
        * **global_right** * -- float; global wavelength from right where flux for all supllied atmposhperes exist
        * **inplace** * -- bool; if set to True; instead of creation of new DataFrames in
                                 elisa.atm.AtmDataContainers, just existing is inplaced (changed)

    :return: elisa.atm.AtmDataContainer;
    """
    inplace = kwargs.get('inplace', False)
    if atm_container.is_empty():
        ValueError('Atmosphere container is empty.')

    # evaluate whether use argument bandwidth or global bandwidth
    # use case when use global bandwidth is in case of bolometric `filter`, where bandwidth in observer
    # is set as generic left = 0 and right sys.float.max
    atm_model = atm_container.model

    if atm_model.wavelength.min() > left_bandwidth or atm_model.wavelength.max() < right_bandwidth:
        _min, _max = find_global_atm_bandwidth([atm_container])
        # use `global_left` if defined (min of wavelengts where exists intersection of atmospheric models)
        #   or current model left wavelength boundary
        # use `global_righ` if defined (max of wavelengts where exists intersection of atmospheric models) or current
        #   or current model right wavelength boundary
        left_bandwidth, right_bandwidth = kwargs.get("global_left", _min), kwargs.get("global_right", _max)

        if not kwargs.get('global_left') or not kwargs.get('global_right'):
            warnings.warn(f"argument bandwidth is out of bound for supplied atmospheric model\n"
                          f"to avoid interpolation error in boundary wavelength, bandwidth was defined as "
                          f"max {_max} and min {_min} of wavelengt in given model table\n"
                          f"it might leads to error in atmosphere interpolation\n"
                          f"to avoid this problem, please specify global_left and global_right bandwidth as "
                          f"kwargs for given method and make sure all models wavelengths "
                          f"are greater or equal to such limits")
    return strip_to_bandwidth(atm_container, left_bandwidth, right_bandwidth, inplace=inplace)


def strip_to_bandwidth(atm_container, left_bandwidth, right_bandwidth, inplace=False):
    """
    Function directly strips atm container to given bandwidth.

    :param atm_container: elisa.atm.AtmDataContainer; atm container to strip
    :param left_bandwidth: float;
    :param right_bandwidth: float;
    :param inplace: if True `atm_container` is overwritten by striped atmosphere container
    :return: elisa.atm.AtmDataContainer;
    """
    # indices in bandwidth
    valid_indices = list(np.where(np.logical_and(np.greater(atm_container.model.wavelength, left_bandwidth),
                                                 np.less(atm_container.model.wavelength, right_bandwidth)))[0])

    # extend left and right index (left - 1 and right + 1)
    left_extention_index = valid_indices[0] - 1 if valid_indices[0] >= 1 else 0
    right_extention_index = valid_indices[-1] + 1 \
        if valid_indices[-1] < atm_container.model.last_valid_index() else valid_indices[-1]
    atm_cont = atm_container if inplace else deepcopy(atm_container)
    atm_cont.model = atm_cont.model[np.unique([left_extention_index] + valid_indices + [right_extention_index])]
    return extend_atm_container_on_bandwidth_boundary(atm_cont, left_bandwidth, right_bandwidth)


def find_global_atm_bandwidth(atm_containers):
    """
    Function finds common wavelength coverage of the atmosphere models.
    Find intersection of wavelengths of models.

    :param atm_containers: elisa.atm.AtmDataContainer;
    :return: Tuple[float, float]; minimum, maximum wavelength of common coverage (in Angstrom)
    """
    bounds = np.array([[atm.model.wavelength.min(), atm.model.wavelength.max()] for atm in atm_containers])
    return bounds[:, 0].max(), bounds[:, 1].min()


def extend_atm_container_on_bandwidth_boundary(atm_container, left_bandwidth, right_bandwidth):
    """
    Function crops the wavelength boundaries of the atmosphere model to the precise boundaries defined by
    `left_bandwidth` and `right_bandwidth`.

    :param atm_container: elisa.atm.AtmDataContainer;
    :param left_bandwidth: float;
    :param right_bandwidth: float;
    :return: elisa.atm.AtmDataContainer;
    """
    interpolator = interpolate.Akima1DInterpolator(atm_container.model.wavelength, atm_container.model.flux)

    # interpolating values precisely on the border of the filter(s) coverage
    on_border_flux = interpolator([left_bandwidth, right_bandwidth])
    if np.isnan(on_border_flux).any():
        raise AtmosphereError('Interpolation on bandwidth boundaries leed to NaN value.')
    atm_model: AtmModel = atm_container.model
    atm_model.wavelength[np.array([0, -1])] = [left_bandwidth, right_bandwidth]
    atm_model.flux[np.array([0, -1])] = [on_border_flux[0], on_border_flux[1]]
    atm_model.flux = np.round(atm_model.flux, 10)

    atm_container.model = atm_model
    # continute here
    return atm_container


def apply_passband(atm_containers, passband, **kwargs):
    """
    Function applies passband response functions to the stripped atmosphere models.

    :param atm_containers: elisa.atm.AtmDataContainer;
    :param passband: Dict[str, PassbandContainer];
    :return: Dict[str, elisa.atm.AtmDataContainer];
    """
    passbanded_atm_containers = dict()
    logger.debug("applying passband functions on given atmospheres")

    for band, band_container in passband.items():
        if band in ['bolometric']:
            band_container.left_bandwidth = kwargs.get('global_left', band_container.left_bandwidth)
            band_container.right_bandwidth = kwargs.get('global_right', band_container.right_bandwidth)

        passbanded_atm_containers[band] = list()
        for atm_container in atm_containers:
            # strip atm container on passband bandwidth (reason to do it is, that container
            # is stripped on maximal bandwidth defined by all bands, not just by given single band)
            atm_container = strip_to_bandwidth(
                atm_container=deepcopy(atm_container),
                left_bandwidth=band_container.left_bandwidth,
                right_bandwidth=band_container.right_bandwidth,
                inplace=False
            )

            # found passband throughput on atm defined wavelength
            passband_throughput = np.nan_to_num(band_container.akima(atm_container.model.wavelength))

            atm_container.model.flux *= passband_throughput
            passbanded_atm_containers[band].append(atm_container)
    logger.debug("passband application finished")
    return passbanded_atm_containers


def build_atm_validation_hypertable(atlas):
    """
    Prepare validation hypertable to validate atmospheric model (whether is in interpolation bounds).

    :param atlas: str;
    :return: Dict;
    """
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


def is_out_of_bound(in_arr, values, tolerance):
    """
    Figure out whether `values` are in `in_arr`. Use `tolerance` if you there is allowed.

    :param in_arr: numpy.array;
    :param values: numpy.array;
    :param tolerance: float;
    :return: List[bool];
    """
    values = [values] if not isinstance(values, Iterable) else values
    top, bottom = max(in_arr) + tolerance, min(in_arr) - tolerance
    return [False if bottom <= val <= top else True for val in values]


# pay attention to those methods bellow
# in the future for different atm model might happen that function won't be valid anymore
def validate_temperature(temperature, atlas, _raise=True):
    """
    Validate `temperature`s for existing `atlas`.

    :param temperature: numpy.array;
    :param atlas: str;
    :param _raise: bool; if True, raise ValueError
    :return: bool;
    """
    atlas = validated_atlas(atlas)
    allowed = sorted(atm_file_prefix_to_quantity_list("temperature", atlas))
    invalid = any([True if (allowed[-1] < t or t < allowed[0]) else False for t in temperature])
    if invalid:
        if _raise:
            msg = "Any temperature in system atmosphere is out of bound; it is usually caused by invalid physical model"
            raise TemperatureError(msg)
        return False
    return True


def validate_metallicity(metallicity, atlas, _raise=True):
    """
    Validate `metallicity`s for existing `atlas`.

    :param metallicity: float;
    :param atlas: float;
    :param _raise: bool; if True, raise ValueError
    :return: bool;
    """
    out_of_bound_tol = 0.1  # how far `out of bound` can any value of metallicity runs
    atlas = validated_atlas(atlas)
    allowed = sorted(atm_file_prefix_to_quantity_list("metallicity", atlas))
    out_of_bound = is_out_of_bound(allowed, metallicity, out_of_bound_tol)
    if any(out_of_bound):
        if _raise:
            raise MetallicityError(f"Any metallicity in system atmosphere is out of bound, allowed values "
                                   f"are in range <{min(allowed) - out_of_bound_tol}, {max(allowed) + out_of_bound_tol}"
                                   f">; it is usually caused by invalid physical model")
        return False
    return True


def validate_logg_temperature_constraint(temperature, log_g, atlas, _raise=True):
    """
    Validate `logg`s for existing `atlas` and `temperature`.

    :param temperature: numpy.array;
    :param log_g: numpy.array;
    :param atlas: str;
    :param _raise: bool; if True, raise ValueError
    :return: bool;
    """
    # it has a different name because there is a different interface
    validation_hypertable = build_atm_validation_hypertable(atlas)
    allowed = sorted(atm_file_prefix_to_quantity_list("temperature", atlas))

    invalid = [
        is_out_of_bound(validation_hypertable[
                            str(int(utils.find_nearest_value(allowed, t)[0]))
                        ]["gravity"], [g], 0.1)[0] for t, g in zip(temperature, log_g)]
    if np.any(invalid):
        if _raise:
            raise GravityError("Any gravity (log_g) in system atmosphere is out of bound; "
                               "it is usually caused by invalid physical model")
        return False
    return True


def validate_atm(temperature, log_g, metallicity, atlas, _raise=True):
    """
    Validate atmosphere.
    Run methods::

        validate_temperature
        validate_metallicity
        validate_logg_temperature_constraint

    If anything is not right and `_raise` set to True, raise ValueError.

    :param temperature: numpy.array;
    :param log_g: numpy.array;
    :param metallicity: float;
    :param atlas: str;
    :param _raise: bool; if True, raise ValueError
    :return: bool;
    """
    try:
        metallicity = [metallicity] * len(temperature) if not isinstance(metallicity, Iterable) else metallicity
        validate_temperature(temperature, atlas)
        validate_metallicity(metallicity, atlas)
        validate_logg_temperature_constraint(temperature, log_g, atlas)
    except (ElisaError, ValueError):
        if not _raise:
            return False
        raise
    return True


def atm_file_prefix_to_quantity_list(qname: str, atlas: str):
    """
    Get list of available values for given atm domain quantity, e.g. list of temperatures available in atlas CK04.

    :param qname: str; e.g. `temperature`, `metallicity`, `gravity`
    :param atlas: str; e.g. `castelli` or `ck04`
    :return: List
    """
    atlas = validated_atlas(atlas)
    return getattr(const, f"{str(atlas).upper()}_{str(settings.ATM_DOMAIN_QUANTITY_TO_VARIABLE_SUFFIX[qname])}")


def validated_atlas(atlas):
    """
    Get validated atm atlas, e.g. `castelli` or `ck04` transform to `ck`, it matches folder
    and file prefix for given atlas.

    :param atlas: str; e.g. `castelli` or `ck04`
    :return: str;
    """
    try:
        return settings.ATM_ATLAS_NORMALIZER[atlas]
    except KeyError:
        raise KeyError(f'Incorrect atlas. Following are allowed: {", ".join(settings.ATM_ATLAS_NORMALIZER.keys())}')


def parse_domain_quantities_from_atm_table_filename(filename: str):
    """
    Parse filename to given quantities, e.g. ckm05_3500_g15.csv parse to tuple (-0.5, 3500, 1.5)

    :param filename: str;
    :return: Tuple[float, float, float];
    """
    return get_temperature_from_atm_table_filename(filename), get_logg_from_atm_table_filename(
        filename), get_metallicity_from_atm_table_filename(filename)


def get_metallicity_from_atm_table_filename(filename):
    """
    Get metallicity as number from filename / directory.

    :param filename: str;
    :return: float;
    """
    m = str(filename).split("_")[0][-3:]
    sign = 1 if str(m).startswith("p") else -1
    value = float(m[1:]) / 10.0
    return value * sign


def get_temperature_from_atm_table_filename(filename):
    """
    Get temperature from filename / directory name.

    :param filename: str;
    :return: float;
    """
    return float(str(filename).split("_")[1])


def get_logg_from_atm_table_filename(filename):
    """
    Get logg from filename / directory name.

    :param filename: str;
    :return: float;
    """
    filename = filename if not str(filename).endswith(".csv") else str(filename).replace('.csv', '')
    g = str(filename).split("_")[2][1:]
    return int(g) / 10.0


def get_atm_table_filename(temperature, log_g, metallicity, atlas):
    """
    Get filename based on given descriptive values.

    :param temperature: float;
    :param log_g: float;
    :param metallicity: float;
    :param atlas: str; e.g. `castelli` or `ck04`
    :return: str;
    """
    prefix = validated_atlas(atlas)
    retval = f"{prefix}{utils.numeric_metallicity_to_string(metallicity)}_" \
             f"{int(temperature)}_{utils.numeric_logg_to_string(log_g)}.csv"
    return retval


def get_atm_directory(metallicity, atlas):
    """
    Get table directory name based on given descriptive values.

    :param metallicity: float
    :param atlas: str; e.g. `castelli` or `ck04`
    :return: str
    """
    prefix = validated_atlas(atlas)
    return f"{prefix}{utils.numeric_metallicity_to_string(metallicity)}"


def get_atm_table(temperature, log_g, metallicity, atlas):
    """
    Get dataframe for flux and wavelengths for given values and atlas.
    (Read csv file)

    :param temperature: float;
    :param log_g: float;
    :param metallicity: float;
    :param atlas: str - e.g. `castelli` or `ck04`
    :return: pandas.DataFrame;
    """
    source = settings.ATLAS_TO_BASE_DIR[atlas]
    directory = get_atm_directory(metallicity, atlas)
    filename = get_atm_table_filename(temperature, log_g, metallicity, atlas)
    path = os.path.join(source, directory, filename) if directory is not None else os.path.join(source, filename)

    if not os.path.isfile(path):
        raise FileNotFoundError(f"there is no file like {path}")
    return pd.read_csv(path, dtype=settings.ATM_MODEL_DATAFARME_DTYPES)


def get_list_of_all_atm_tables(atlas):
    """
    Get list of all available atm table files stored in configured location.

    :param atlas: str; e.g. `castelli` or `ck04`
    :return: List[str];
    """
    source = settings.ATLAS_TO_BASE_DIR[validated_atlas(atlas)]
    matches = list()
    for root, dirnames, filenames in os.walk(source):
        for filename in filenames:
            if filename.endswith(('.csv',)):
                matches.append(os.path.join(root, filename))
    return matches


def multithread_atm_tables_reader(path_queue, error_queue, result_queue):
    """
    Multithread reader of atmosphere csv files.

    :param path_queue: Queue;
    :param error_queue: Queue;
    :param result_queue: Queue;
    """
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
            types = {'flux': np.float, 'wave': np.float}
            t, l, m = parse_domain_quantities_from_atm_table_filename(os.path.basename(file_path))
            atm_container = AtmDataContainer(pd.read_csv(file_path, dtype=types), t, l, m, file_path)
            result_queue.put((index, atm_container))
        except Exception as we:
            error_queue.put(we)
            break


def multithread_atm_tables_reader_runner(fpaths):
    """
    Run multithread reader of csv files containing atmospheric models.

    :param fpaths: Iterable[str];
    :return: Queue;
    """
    n_threads = settings.NUMBER_OF_THREADS

    path_queue = Queue(maxsize=len(fpaths) + n_threads)
    result_queue = Queue()
    error_queue = Queue()

    threads = list()
    try:
        for index, fpath in enumerate(fpaths):
            if not os.path.isfile(fpath):
                logger.debug(f"accessing atmosphere file {fpath}")
                raise AtmosphereError(f"file {fpath} doesn't exist. Your atmosphere tables are either not properly "
                                      f"installed or atmosphere parameters of your stellar model are not covered by "
                                      f"the currently used table.")
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


def compute_normal_radiances(matrices_dict, flux_mult=1.0, wave_mult=1.0):
    """
    Run `compute_normal_intensity` method for each band in `matrices_dict`.

    :param matrices_dict: Dict;
    :param flux_mult: float;
    :param wave_mult: float;
    :return: Dict[str, float];
    """
    return {
        band: compute_normal_intensity(
            spectral_flux=dflux[settings.ATM_MODEL_DATAFRAME_FLUX],
            wavelength=dflux[settings.ATM_MODEL_DATAFRAME_WAVE],
            flux_mult=flux_mult,
            wave_mult=wave_mult
        )
        for band, dflux in matrices_dict.items()
    }


def compute_normal_intensity(spectral_flux, wavelength, flux_mult=1.0, wave_mult=1.0):
    """
    Calculates normal flux for all surface faces.

    :param spectral_flux: numpy.array; interpolated atmosphere models for each face (N_face x wavelength)
    :param wavelength: numpy.array or Series; wavelengths of atmosphere models
    :param flux_mult: float;
    :param wave_mult: float;
    :return: numpy.array;
    """
    return flux_mult * wave_mult * integrate.simps(spectral_flux, wavelength, axis=1)


def compute_integral_si_intensity_from_passbanded_dict(passbaned_dict):
    return {
        band: compute_integral_si_intensity_from_atm_data_containers(passbanded_atm)
        for band, passbanded_atm in passbaned_dict.items()
    }


def compute_integral_si_intensity_from_atm_data_containers(atm_data_containers):
    """
    Returns intensity from given atmosphere models.
    If models are already strip by passband, result is also striped

    :param atm_data_containers: Iterable[elisa.atm.AtmDataContainer];
    :return: List[elisa.atm.AtmDataContainer]; integrated `flux` from each elisa.atm.AtmDataContainer
                                               on `wave` in given container
    """
    return [
        IntensityContainer(
            intensity=const.PI * integrate.simps(adc.model.flux * adc.flux_to_si_mult,
                                                 adc.model.wavelength * adc.wave_to_si_mult),
            temperature=adc.temperature,
            log_g=adc.log_g,
            metallicity=adc.metallicity
        )
        for adc in atm_data_containers
    ]


def unique_atm_fpaths(fpaths):
    """
    Group atm table names and return such set and map to origin list.

    :param fpaths: List[str];
    :return: Tuple[str, Dict];

    ::

        (path set - set of unique atmosphere file names,
         map - Dict where every unique atm file has listed indices where it occures)
    """
    fpaths_set = set(fpaths)
    fpaths_map = {key: list() for key in fpaths_set}
    for idx, key in enumerate(fpaths):
        fpaths_map[key].append(idx)
    return fpaths_set, fpaths_map


def remap_passbanded_unique_atms_to_origin(passbanded_containers, fpaths_map):
    """
    Remap atm containers in supplied order by `fpaths_map`.

    :param passbanded_containers: Dict[str, elisa.atm.AtmDataContainer]
    :param fpaths_map: Dict[str, List[int]]; map
    :return: Dict[str, List];
    """
    return {band: remap_unique_atm_container_to_origin(atm, fpaths_map) for band, atm in passbanded_containers.items()}


def remap_unique_atm_container_to_origin(models, fpaths_map):
    """
    Remap atm container in supplied order by `fpaths_map`.

    :warning: assigned containers are mutable, if you will change content of any container, changes will affect
              any other container with same reference

    :param models: List[AtmDatContainer];
    :param fpaths_map: :param fpaths_map: Dict[str, List[int]]; map
    :return: List[elisa.atm.AtmDataContainer];
    """
    models_arr = np.empty(max(list(itertools.chain.from_iterable(fpaths_map.values()))) + 1, dtype='O')
    for model in models:
        models_arr[fpaths_map[model.fpath]] = model
    return models_arr


def read_unique_atm_tables(fpaths):
    """
    Returns atmospheric spectra from table files which encompass the range of surface parameters on the component's
    surface

    :parma fpaths; List[str];
    :return: Tuple[elisa.atm.AtmDataContainers, Dict[str, List]];

    ::

        (List of unique elisa.atm.AtmDataContainers, map - dict where every unique atm file has listed
        indices where it occures)
    """
    fpaths, fpaths_map = unique_atm_fpaths(fpaths)

    # check if the atm table is in the buffer
    models, load_fpaths = [], []
    for fpath in fpaths:
        if fpath in buffer.ATMOSPHERE_TABLES:
            models.append(buffer.ATMOSPHERE_TABLES[fpath])
        else:
            load_fpaths.append(fpath)

    if len(load_fpaths) > 0:
        result_queue = multithread_atm_tables_reader_runner(load_fpaths)
        loaded_models = [qval[1] for qval in utils.IterableQueue(result_queue) if qval[1] is not None]
        # add loaded atmospheres to atm buffer
        for ii, fpath in enumerate(load_fpaths):
            buffer.ATMOSPHERE_TABLES[fpath] = loaded_models[ii]
        models += loaded_models
    # clean buffer
    buffer.reduce_buffer(buffer.ATMOSPHERE_TABLES)
    return models, fpaths_map


def find_atm_si_multiplicators(atm_containers):
    """
    Get atm flux and wavelength multiplicator from `atm_containers`.
    It assume, all containers have the same multiplicators, so it returns values from first one.

    :param atm_containers: List[AtmDatacontainer];
    :return: Tuple[float, float];

    ::

        (flux multiplicator, wavelength multiplicator)

    """
    for atm_container in atm_containers:
        return atm_container.flux_to_si_mult, atm_container.wave_to_si_mult
    raise ValueError('No valid atmospheric container has been supplied to method.')


def find_atm_defined_wavelength(atm_containers):
    """
    Get wavelength from first container from `atm_containers` list.
    It assume all containers has already aligned wavelengths to same.

    :param atm_containers: Iterable[elisa.atm.AtmDataContainer];
    :return: numpy.array[float];
    """
    for atm_container in atm_containers:
        return atm_container.model.wavelength
    raise AtmosphereError('No valid atmospheric container has been supplied to method.')


def remap_passbanded_unique_atms_to_matrix(passbanded_containers, fpaths_map):
    """
    Run `remap_passbanded_unique_atm_to_matrix` for reach container in `passbanded_containers`.

    :param passbanded_containers: List[];
    :param fpaths_map: Dict[str, List[int]]; map - atmosphere container to faces
    :return: Dict[str, numpy.array];
    """
    return {band: remap_passbanded_unique_atm_to_matrix(atm, fpaths_map) for band, atm in passbanded_containers.items()}


def remap_passbanded_unique_atm_to_matrix(atm_containers, fpaths_map):
    """
    Creating matrix of atmosphere models for each face.

    :param atm_containers: List[elisa.atm.AtmDataContainer]; list of unique atmosphere containers from tables
    :param fpaths_map: Dict[str, List[int]]; map - atmosphere container to faces
    :return: numpy.array; matrix of atmosphere models
    """
    total = max(list(itertools.chain.from_iterable(fpaths_map.values()))) + 1
    wavelengths_defined = find_atm_defined_wavelength(atm_containers)
    wavelengths_length = len(wavelengths_defined)
    models_matrix = up.zeros((total, wavelengths_length))

    for atm_container in atm_containers:
        models_matrix[fpaths_map[atm_container.fpath]] = atm_container.model.flux
    return models_matrix


def correct_normal_radiance_to_optical_depth(normal_radiances, ld_cfs):
    """
    Correcting normal radiance values by increment that will correct inacuracy caused by using too shallow optical depth
    for the middle of the disk. Correction was derived analytically from spherical model.

    :param normal_radiances: Dict; dict(component: dict(filter: normal radiances for each face))
    :param ld_cfs: Dict; dict(component: dict(filter: limb darkening coefficients for each face))
    :return: Dict;
    """
    for star, component_normal_radiances in normal_radiances.items():
        ld_coefficients = ld_cfs[star]['bolometric'].T

        coeff = ld.calculate_integrated_limb_darkening_factor(limb_darkening_law=settings.LIMB_DARKENING_LAW,
                                                              coefficients=ld_coefficients)

        normal_radiances[star] = {
            band: normal_radiance / coeff for band, normal_radiance in component_normal_radiances.items()
        }

    return normal_radiances


def planck_function(wavelegth, temperature):
    """
    Standard Planck funcntion.
    :param wavelegth: Union[float, numpy.array]; wavelengths
    :param temperature: float; temperature
    :return: Union[float, numpy.array]
    """
    h = (2.0 * const.PLANCK_CONST * const.C ** 2) / np.power(wavelegth, 5)
    e = (const.PLANCK_CONST * const.C) / (wavelegth * const.BOLTZMAN_CONST * temperature)
    return h / (np.exp(e) - 1.0)


def get_standard_wavelengths():
    """
    Obtain standard wavelengths used in Castelli-Kurucz tables.
    :return: numpy.array
    """
    with open(os.path.join(settings.DATA_PATH, "wavelength.json"), "r") as f:
        return np.array(json.loads(f.read()))
