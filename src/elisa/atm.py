from __future__ import annotations

import itertools
import json
import warnings
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import interpolate

from elisa import (
    const,
    ld,
    settings,
    utils,
)
from elisa import (
    umpy as up,
)
from elisa.base.error import (
    AtmosphereError,
    ElisaError,
    GravityError,
    MetallicityError,
    TemperatureError,
)
from elisa.base.types import FLOAT
from elisa.buffer import buffer
from elisa.logger import getLogger
from elisa.tensor.etensor import Tensor

# Import here only for type checking to avoid runtime import cycles and keep these
# names visible to static analysis tools and IDEs. These imports are evaluated
# only during type checking and therefore do not affect runtime import order.
if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.observer.passband import PassbandContainer
    from elisa.types import Float, Int

logger = getLogger(__name__)


# * 1e-7 * 1e4 * 1e10 * (1.0/const.PI)


class AtmModel:
    """Atmospheric model container for flux and wavelength data.

    Holds parallel arrays of ``flux`` and ``wavelength`` that represent a
    single atmospheric spectrum. The arrays are stored as numpy arrays of
    type :data:`elisa.base.types.FLOAT`.
    """

    def __init__(self, flux: NDArray[Float] | None, wavelength: NDArray[Float] | None) -> None:
        """Initialize an :class:`AtmModel` instance.

        :param flux: Flux array in flam (spectral flux values). If ``None``,
            an empty model is created.
        :type flux: numpy.typing.NDArray[elisa.types.Float] | None
        :param wavelength: Wavelength array in Angstrom. If ``None``, the
            model is considered empty.
        :type wavelength: numpy.typing.NDArray[elisa.types.Float] | None
        :returns: None
        :rtype: None
        """
        self.flux: NDArray | None = np.array(flux, dtype=FLOAT) if flux is not None else None
        self.wavelength: NDArray | None = np.array(wavelength, dtype=FLOAT) if wavelength is not None else None

    def _empty(self) -> bool:
        """Check if the model contains no data."""
        return self.wavelength is None or len(self.wavelength) == 0

    @property
    def empty(self) -> bool:
        """Check if the model is empty."""
        return self._empty()

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> AtmModel:
        """Create an :class:`AtmModel` from a :class:`pandas.DataFrame`.

        The dataframe must contain columns named according to
        :data:`settings.ATM_MODEL_DATAFRAME_FLUX` and
        :data:`settings.ATM_MODEL_DATAFRAME_WAVE`.

        :param df: DataFrame with flux and wavelength columns.
        :type df: pandas.DataFrame
        :returns: Constructed :class:`AtmModel`.
        :rtype: AtmModel
        """
        return cls(
            flux=np.array(df[settings.ATM_MODEL_DATAFRAME_FLUX], dtype=FLOAT),
            wavelength=np.array(df[settings.ATM_MODEL_DATAFRAME_WAVE], dtype=FLOAT),
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return the model represented as a :class:`pandas.DataFrame`.

        The returned DataFrame uses the column names defined in
        :data:`settings.ATM_MODEL_DATAFRAME_FLUX` and
        :data:`settings.ATM_MODEL_DATAFRAME_WAVE`.

        :returns: DataFrame with flux and wavelength columns.
        :rtype: pandas.DataFrame
        """
        return pd.DataFrame(
            {
                settings.ATM_MODEL_DATAFRAME_FLUX: self.flux,
                settings.ATM_MODEL_DATAFRAME_WAVE: self.wavelength,
            },
        )

    def last_valid_index(self) -> int:
        """Return the length of the flux array (index past the last element).

        :returns: Number of wavelength points (one past the last valid index).
        :rtype: int
        """
        return len(self.flux)

    def __getitem__(self, item: int | slice | NDArray[Int]) -> AtmModel:
        """Return a new :class:`AtmModel` selected by index or indices.

        The ``item`` may be:

        - an ``int`` index,
        - a ``slice``, or
        - a numpy integer index array (``NDArray[int]``) selecting arbitrary
          positions.

        :param item: Indexer selecting positions from ``flux`` and
            ``wavelength``.
        :type item: int | slice | numpy.typing.NDArray[int]
        :returns: A new :class:`AtmModel` containing the sliced arrays.
        :rtype: AtmModel
        """
        return AtmModel(flux=self.flux[item], wavelength=self.wavelength[item])

    def __len__(self) -> int:
        """Get the number of wavelength points in the model."""
        return len(self.wavelength)


class AtmDataContainer:
    """Container holding an atmospheric model and its metadata.

    Stores an :class:`AtmModel` together with atmospheric parameters
    (temperature, log_g, metallicity) and multiplicators used to convert
    model values to SI units.
    """

    def __init__(
        self,
        model: AtmModel | pd.DataFrame,
        temperature: Float,
        log_g: Float,
        metallicity: Float,
        fpath: str = "",
    ) -> None:
        """Initialize an :class:`AtmDataContainer`.

        :param model: The atmospheric model, either an :class:`AtmModel` or a
            :class:`pandas.DataFrame` with appropriate columns.
        :type model: AtmModel | pandas.DataFrame
        :param temperature: Effective temperature in Kelvin.
        :type temperature: Float
        :param log_g: Surface gravity (log10(cm/s^2)).
        :type log_g: Float
        :param metallicity: Metallicity [Fe/H].
        :type metallicity: Float
        :param fpath: Optional file path to the source atmosphere table.
        :type fpath: str
        :returns: None
        :rtype: None
        """
        self._model = AtmModel(flux=None, wavelength=None)
        self.temperature: Float = temperature
        self.log_g: Float = log_g
        self.metallicity: Float = metallicity
        self.flux_unit: str = "flam"
        self.wave_unit: str = "angstrom"
        # Note: const.PI will cause redundant multiplication in intensity
        # integration if kept here. flam = erg·s⁻¹·cm⁻²·Å⁻¹ =  (10⁻⁷ J)·s⁻¹·(10⁻² m)⁻²·(10⁻¹⁰ m)⁻¹
        # = 10⁻⁷ x 10⁴ x 10¹⁰ J·s⁻¹·m⁻³
        self.flux_to_si_mult: Float = 1e7  # * (1.0/const.PI)
        self.wave_to_si_mult: Float = 1e-10
        self._left_bandwidth: Float = np.nan
        self._right_bandwidth: Float = np.nan
        self.fpath: str = fpath

        self.model = model

    @property
    def left_bandwidth(self) -> Float:
        """Left bandwidth (minimum wavelength) of the contained model.

        :returns: Left bandwidth in Angstrom.
        :rtype: Float
        """
        return self._left_bandwidth

    @property
    def right_bandwidth(self) -> Float:
        """Right bandwidth (maximum wavelength) of the contained model.

        :returns: Right bandwidth in Angstrom.
        :rtype: Float
        """
        return self._right_bandwidth

    @left_bandwidth.setter
    def left_bandwidth(self, value: Float) -> None:
        """Set the left bandwidth (minimum wavelength) of the model.

        :param value: Left bandwidth in Angstrom.
        :type value: Float
        """
        self._left_bandwidth = value

    @right_bandwidth.setter
    def right_bandwidth(self, value: Float) -> None:
        """Set the right bandwidth (maximum wavelength) of the model.

        :param value: Right bandwidth in Angstrom.
        :type value: Float
        """
        self._right_bandwidth = value

    def is_empty(self) -> bool:
        """Check if the atmospheric model container is empty.

        :returns: True if no model data is present.
        :rtype: bool
        """
        return self._model.empty

    @property
    def model(self) -> AtmModel:
        """Get the atmospheric model instance.

        :returns: The contained AtmModel.
        :rtype: AtmModel
        """
        return self._model

    @model.setter
    def model(self, data: AtmModel | pd.DataFrame) -> None:
        """Assign the contained atmospheric model and update bandwidth info.

        If ``data`` is a :class:`pandas.DataFrame`, it is converted to
        :class:`AtmModel` via :meth:`AtmModel.from_dataframe`.

        :param data: Atmospheric model or DataFrame with model columns.
        :type data: AtmModel | pandas.DataFrame
        :returns: None
        :rtype: None
        """
        self._model = AtmModel.from_dataframe(data) if isinstance(data, pd.DataFrame) else data

        self._left_bandwidth = self._model.wavelength.min()
        self._right_bandwidth = self._model.wavelength.max()


class IntensityContainer:
    """Container for integrated radiance data with associated atmospheric parameters."""

    def __init__(
        self,
        intensity: Float,
        temperature: Float,
        log_g: Float,
        metallicity: Float,
    ) -> None:
        """Initialize an intensity container with integrated radiance values.

        :param intensity: Integrated radiance value.
        :type intensity: Float
        :param temperature: Effective temperature in K.
        :type temperature: Float
        :param log_g: Surface gravity in log10(cm/s²).
        :type log_g: Float
        :param metallicity: Metallicity [Fe/H].
        :type metallicity: Float
        """
        self.intensity: Float = intensity
        self.temperature: Float = temperature
        self.log_g: Float = log_g
        self.metallicity: Float = metallicity


class NaiveInterpolatedAtm:
    """Atmosphere radiance computation using simple interpolation methods."""

    @staticmethod
    def radiance(
        temperature: NDArray,
        log_g: NDArray,
        metallicity: Float,
        atlas: str,
        **kwargs,
    ) -> dict:
        """Compute radiance for given atmospheric parameters and passbands.

        :param temperature: numpy.array;
        :param log_g: numpy.array;
        :param metallicity: float;
        :param atlas: str;
        :param kwargs:
        :**kwargs options**:
            * **left_bandwidth** * -- float; maximal allowed wavelength from left (Angstrom)
            * **right_bandwidth** * -- float; maximal allowed wavelength from right (Angstrom)
            * **passband** * -- dict[str, elisa.observer.observer.PassbandContainer]
        :return: List;
        """
        if validated_atlas(atlas) == "bb":
            return NaiveInterpolatedAtm.black_body_radiance(temperature, **kwargs)
        return NaiveInterpolatedAtm.atlas_radiance(
            temperature,
            log_g,
            metallicity,
            atlas,
            **kwargs,
        )

    @staticmethod
    def black_body_radiance(temperature: NDArray, **kwargs) -> dict[str, NDArray[Float]]:
        """Compute integrated flux per face for each passband using the Planck function.

        For a set of temperatures and passbands, this returns a mapping
        from passband name to a 1D numpy array (type FLOAT) containing the
        integrated flux for each input face.

        :param temperature: Array of temperatures (K).
        :type temperature: NDArray
        :param kwargs: Additional keyword arguments; expects ``passband``.
        :returns: Mapping passband → 1D numpy array of integrated flux
            values (one per face).
        :rtype: dict[str, numpy.typing.NDArray[elisa.types.Float]]
        """
        # setup multiplicators to convert quantities to SI
        flux_mult, wave_mult = const.PI, 1e-10
        # obtain localized atmospheres in matrix
        localized_atms = NaiveInterpolatedAtm.arange_black_body_localized_atms(temperature, kwargs["passband"])
        # integrate flux
        return compute_normal_radiances(localized_atms, flux_mult=flux_mult, wave_mult=wave_mult)

    @staticmethod
    def arange_black_body_localized_atms(
        temperature: NDArray[Float],
        passband_containers: dict[str, PassbandContainer],
    ) -> dict[str, dict[str, NDArray]]:
        """Generate atmosphere models based on Planck Function.

        The all models are sitting on temperatures given by surface elements.

        :param temperature: numpy.ndarray;
        :param passband_containers: dict[str, PassbandContainer];
        :return: dict[str, numpy.ndarray];
        """
        localized_atms = {}
        standard_wavelength = get_standard_wavelengths()

        # build temperature mask and avoid repeative computation
        # temperature values where decimal points are basicaly useless
        temperature = np.round(temperature, 0)
        temperature, reverse_map = np.unique(temperature, return_inverse=True)

        for band, pb_container in passband_containers.items():
            # how many wavelengths generate based on standard
            mask = np.logical_and(
                np.less_equal(standard_wavelength, pb_container.right_bandwidth),
                np.greater_equal(standard_wavelength, pb_container.left_bandwidth),
            )
            hm_waves = len(standard_wavelength[mask])
            # wavelenghts in angstrom
            wavelengths = np.sort(
                np.unique(
                    np.concatenate(
                        [
                            np.linspace(
                                pb_container.left_bandwidth,
                                pb_container.right_bandwidth,
                                hm_waves,
                                endpoint=True,
                            ),
                            standard_wavelength[mask],
                        ],
                    ),
                ),
            )

            # compute flux in flam, apply passband and replace possible NaNs
            flux = np.nan_to_num(
                [
                    pb_container.akima(wavelengths)
                    * planck_function(wavelengths * pb_container.wave_to_si_mult, _temperature)
                    for _temperature in temperature
                ],
            )
            # sometimes, there are small negative values on the boundwidth boundaries due to akima interpolation
            flux[np.less(flux, 0)] = 0.0
            # broadcast and fill localized atms
            localized_atms[band] = {
                "flux": flux[reverse_map],
                "wave": wavelengths,
            }

        return localized_atms

    @staticmethod
    def get_atm_profiles(
        temperature: NDArray[Float],
        log_g: NDArray[Float],
        metallicity: Float,
        atlas: str,
        **kwargs,
    ) -> tuple[dict[str, dict[str, NDArray]], Float, Float]:
        """Return atmosphere profiles for given surface parameters.

        :param temperature: Iterable[float];
        :param log_g: Iterable[float];
        :param metallicity: float;
        :param atlas: str; atmosphere model identificator (see settings.ATLAS_TO_ATM_FILE_PREFIX.keys())
        :param kwargs: dict;
        :return: Tuple[dict, numpy.float, numpy.float]; atmosphere profiles for each passband, flux multiplicator,
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
        unique_atms = strip_atm_containers_by_bandwidth(
            unique_atms,
            l_bandw,
            r_bandw,
            global_left=global_left,
            global_right=global_right,
        )

        # alignement of atmosphere containers wavelengths for convenience
        unique_atms = arange_atm_to_same_wavelength(unique_atms)
        passbanded_atm_containers = apply_passband(
            unique_atms,
            passband_containers,
            global_left=global_left,
            global_right=global_right,
        )

        flux_matrices = remap_passbanded_unique_atms_to_matrix(
            passbanded_atm_containers,
            containers_map,
        )
        atm_containers = remap_passbanded_unique_atms_to_origin(
            passbanded_atm_containers,
            containers_map,
        )
        localized_atms = NaiveInterpolatedAtm.interpolate_spectra(
            atm_containers,
            flux_matrices,
            temperature=temperature,
        )

        return localized_atms, flux_mult, wave_mult

    @staticmethod
    def atlas_radiance(
        temperature: NDArray[Float],
        log_g: NDArray[Float],
        metallicity: Float,
        atlas: str,
        **kwargs,
    ) -> dict[str, NDArray[Float]]:
        """Compute integrated normal radiance per face for each passband from atlas models.

        :param temperature: Iterable of temperatures (K).
        :type temperature: NDArray[Float]
        :param log_g: Iterable of surface gravities (log10(cm/s^2)).
        :type log_g: NDArray[Float]
        :param metallicity: Metallicity [Fe/H].
        :type metallicity: Float
        :param atlas: Atlas identifier (see
            :data:`settings.ATLAS_TO_ATM_FILE_PREFIX`).
        :type atlas: str
        :param kwargs: Additional parameters forwarded to
            :meth:`NaiveInterpolatedAtm.get_atm_profiles` (e.g. passband,
            left_bandwidth, right_bandwidth).
        :returns: Mapping passband → 1D numpy array of integrated normal
            radiances (one value per face).
        :rtype: dict[str, numpy.typing.NDArray[elisa.types.Float]]
        """
        args = temperature, log_g, metallicity, atlas
        localized_atms, flux_mult, wave_mult = NaiveInterpolatedAtm.get_atm_profiles(*args, **kwargs)
        return compute_normal_radiances(localized_atms, flux_mult=flux_mult, wave_mult=wave_mult)

    @staticmethod
    def compute_interpolation_weights(
        temperatures: NDArray[Float],
        top_atm_containers: list[AtmDataContainer],
        bottom_atm_containers: list[AtmDataContainer],
    ) -> NDArray[Float]:
        """Compute interpolation weights between two models of atmoshperes.

        Weights are computet as::

            (temperatures^4 - bottom_temperatures^4) / (top_temperatures^4 - bottom_temperatures^4)

        what means we use linear approach.
        If there is np.NaN (it cames from same surounded values), such value is replaced with 1.0.
        1.0 is choosen to fit interpolation method and return correct atmosphere.

        :param temperatures: numpy.array[float];
        :param top_atm_containers: numpy.ndarray[elisa.atm.AtmDataContainer];
        :param bottom_atm_containers: numpy.ndarray[elisa.atm.AtmDataContainer];
        :return: numpy.ndarray[float];
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            # Vectorized temperature extraction
            top_temperatures4 = np.power(np.asarray([a.temperature for a in top_atm_containers], dtype=FLOAT), 4)
            bottom_temperatures4 = np.power(np.asarray([a.temperature for a in bottom_atm_containers], dtype=FLOAT), 4)

            result = (np.power(temperatures, 4) - bottom_temperatures4) / (top_temperatures4 - bottom_temperatures4)

            result[up.isnan(result)] = 1.0
            return result

    @staticmethod
    def compute_unknown_intensity_from_surounded_containers(
        weight: NDArray[Float],
        top_atm_container: AtmDataContainer,
        bottom_atm_container: AtmDataContainer,
    ) -> tuple[NDArray[Float], NDArray[Float]]:
        """Compute (interpolate) intensities from surounded intensities related to given temperature.

        Depends on weight will compute (interpolate) intensities from
        surounded intensities related to given temperature.

        ! Top and bottom atmosphere model are have to be defined in same wavelengths !

        :param weight: Iterable[float];
        :param top_atm_container: elisa.atm.AtmDataContainer;
        :param bottom_atm_container: elisa.atm.AtmDataContainer;
        :return: Tuple[numpy.array, numpy.ndarray]; (flux, wave);
        """
        if bottom_atm_container is None:
            return top_atm_container.model.flux, top_atm_container.model.wavelength

        intensity = (
            weight * (top_atm_container.model.flux - bottom_atm_container.model.flux) + bottom_atm_container.model.flux
        )

        return intensity, top_atm_container.model.wavelength

    @staticmethod
    def compute_unknown_intensity_from_surounded_flux_matrices(
        weights: NDArray[Float],
        top_flux_matrix: NDArray[Float],
        bottom_flux_matrix: NDArray[Float],
    ) -> NDArray[Float]:
        """Compute (interpolate) intensities from surounded flux matrices related to given temperature."""
        import time  # noqa: PLC0415

        t = time.time()

        weights = Tensor(weights)
        top_flux_matrix = Tensor(top_flux_matrix)
        bottom_flux_matrix = Tensor(bottom_flux_matrix)
        result = (weights * (top_flux_matrix.T - bottom_flux_matrix.T) + bottom_flux_matrix.T).T

        elapsed = time.time() - t
        logger.info("time %.6f seconds for interpolation of flux matrices", elapsed)
        return result.get()

    @staticmethod
    def interpolate_spectra(
        passbanded_atm_containers: dict[str, list[AtmDataContainer]],
        flux_matrices: dict[str, NDArray[Float]],
        temperature: NDArray[Float],
    ) -> dict[str, dict[str, NDArray[Float]]]:
        """Interpolate spectra for given temperature based on surounded spectra related to upper and lower temperatures.

        From supplied elisa.atm.AtmDataContainer's, `flux_matrices` and `temeprature`.
        Interpolation is computed in vector form::

            (weights * (top_flux_matrix.T - bottom_flux_matrix.T) + bottom_flux_matrix.T).T

        where `top_flux_matrix` and `bottom_flux_matrix`, are entire matrix where rows are represented by fluxes.
        It also means, to be able to do such interpolation, fluxes have to be on same wavelengths for each row.

        :param flux_matrices: dict[str, numpy.ndarray];

        ::

            {"passband": numpy.ndarray (matrix)}

        :param passbanded_atm_containers: dict[str, elisa.atm.AtmDataContainers];
        :param temperature: numpy.ndarray[float];
        :return: dict[str, numpy.ndarray];
        """
        interp_band = {}
        for band, flux_matrix in flux_matrices.items():
            band_atm = passbanded_atm_containers[band]
            bottom_flux, top_flux = flux_matrix[: len(flux_matrix) // 2], flux_matrix[len(flux_matrix) // 2 :]
            bottom_atm, top_atm = band_atm[: len(band_atm) // 2], band_atm[len(band_atm) // 2 :]

            logger.debug("computing atmosphere interpolation weights for band: %s", band)
            interpolation_weights = NaiveInterpolatedAtm.compute_interpolation_weights(
                temperature,
                top_atm,
                bottom_atm,
            )
            flux = NaiveInterpolatedAtm.compute_unknown_intensity_from_surounded_flux_matrices(
                interpolation_weights,
                top_flux,
                bottom_flux,
            )
            interp_band[band] = {
                settings.ATM_MODEL_DATAFRAME_FLUX: flux,
                settings.ATM_MODEL_DATAFRAME_WAVE: find_atm_defined_wavelength(top_atm),
            }
        return interp_band

    @staticmethod
    def atm_files(
        temperature: NDArray[Float],
        log_g: NDArray[Float],
        metallicity: Float,
        atlas: str,
    ) -> list[str | Path]:
        """Find out related atmospheric csv tables and return list of paths to them.

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

        domain_df = pd.DataFrame(
            {
                "temp": t.flatten("F"),
                "log_g": np.tile(g, 2),
                "mh": np.repeat(m, len(g) * 2),
            },
        )
        directory = get_atm_directory(m, atlas)

        # Vectorized string conversions (faster than pandas .apply())
        # Metallicity: 0.5 -> 'p05', -1.1 -> 'm11'
        signs = np.where(domain_df["mh"].to_numpy() >= 0, "p", "m")
        abs_mh = np.abs(domain_df["mh"].to_numpy() * 10).astype(int)
        mh_name = np.char.add(signs, np.char.zfill(abs_mh.astype(str), 2))

        # Temperature: simple int conversion
        temp_name = domain_df["temp"].to_numpy().astype(int).astype(str)

        # Surface gravity: 0.5 -> 'g05', 1.0 -> 'g10'
        logg_int = (domain_df["log_g"].to_numpy() * 10).astype(int)
        log_g_name = np.char.add("g", np.char.zfill(logg_int.astype(str), 2))

        # Build filenames
        fnames = str(atlas) + mh_name + "_" + temp_name + "_" + log_g_name

        return [str(Path(settings.ATLAS_TO_BASE_DIR[atlas]) / directory / (fname + ".csv")) for fname in fnames]


def arange_atm_to_same_wavelength(
    atm_containers: list[AtmDataContainer],
) -> list[AtmDataContainer]:
    """Align a list of atmosphere containers to a common wavelength grid.

    The function computes the sorted union of all wavelengths present in
    ``atm_containers`` and interpolates each container's spectrum to that
    grid using an Akima interpolator. If all containers already share the
    same wavelength length, the original list is returned unchanged.

    :param atm_containers: Iterable of :class:`AtmDataContainer` objects
        to align.
    :type atm_containers: list[AtmDataContainer]
    :returns: A new list of :class:`AtmDataContainer` objects with aligned
        ``wavelength`` and ``flux`` arrays.
    :rtype: list[AtmDataContainer]
    """
    # Check if all containers already have identical wavelengths
    first_wavelengths = atm_containers[0].model.wavelength
    if all(np.array_equal(first_wavelengths, atm.model.wavelength) for atm in atm_containers[1:]):
        return atm_containers

    # If not aligned, compute union and interpolate
    wavelengths = np.unique(np.concatenate([atm.model.wavelength for atm in atm_containers]))
    wavelengths.sort()
    result = []

    # Interpolate each container to common wavelengths
    for atm in atm_containers:
        i = interpolate.Akima1DInterpolator(atm.model.wavelength, atm.model.flux)
        atm.model = AtmModel(
            wavelength=wavelengths,
            flux=np.nan_to_num(i(wavelengths)),
        )
        result.append(atm)
    return result


def strip_atm_containers_by_bandwidth(
    atm_containers: list[AtmDataContainer],
    left_bandwidth: Float,
    right_bandwidth: Float,
    **kwargs,
) -> list[AtmDataContainer]:
    """Strip all loaded atmosphere models to a common wavelength coverage.

    Applies bandwidth stripping to each container. Additional keyword
    arguments (e.g., global_left, global_right) are passed through to
    the strip_atm_container_by_bandwidth function.

    :param atm_containers: List of atmosphere containers to strip.
    :type atm_containers: list[AtmDataContainer]
    :param left_bandwidth: Left (minimum) wavelength boundary in Angstrom.
    :type left_bandwidth: Float
    :param right_bandwidth: Right (maximum) wavelength boundary in Angstrom.
    :type right_bandwidth: Float
    :param kwargs: Additional keyword arguments for bandwidth stripping.
    :returns: List of bandwidth-stripped atmosphere containers.
    :rtype: list[AtmDataContainer]
    """
    return [
        strip_atm_container_by_bandwidth(
            atm_container,
            left_bandwidth,
            right_bandwidth,
            **kwargs,
        )
        for atm_container in atm_containers
    ]


def strip_atm_container_by_bandwidth(
    atm_container: AtmDataContainer,
    left_bandwidth: Float,
    right_bandwidth: Float,
    **kwargs,
) -> AtmDataContainer:
    """Strip an atmosphere container model to a specified wavelength bandwidth.

    If the model does not span the requested bandwidth, global bandwidth
    values (if provided) are used instead. May issue a warning if argument
    bandwidth is out of bounds for the supplied model.

    :param atm_container: Atmosphere container to strip.
    :type atm_container: AtmDataContainer
    :param left_bandwidth: Left (minimum) wavelength boundary in Angstrom.
    :type left_bandwidth: Float
    :param right_bandwidth: Right (maximum) wavelength boundary in Angstrom.
    :type right_bandwidth: Float
    :param kwargs: Optional bandwidth parameters (global_left, global_right,
        inplace). See Notes.
    :returns: Bandwidth-stripped atmosphere container.
    :rtype: AtmDataContainer

    **kwargs options:**

    - **global_left** (FLOAT): Global min wavelength where intersection exists.
    - **global_right** (FLOAT): Global max wavelength where intersection exists.
    - **inplace** (bool): If True, modify the input container in place;
      otherwise return a copy.
    """
    inplace = kwargs.get("inplace", False)
    if atm_container.is_empty():
        msg = "Atmosphere container is empty."
        raise ValueError(msg)

    # evaluate whether you use argument bandwidth or global bandwidth
    # use case when use global bandwidth is in case of bolometric `filter`, where bandwidth in observer
    # is set as generic left = 0 and right sys.float.max
    atm_model = atm_container.model

    if atm_model.wavelength.min() > left_bandwidth or atm_model.wavelength.max() < right_bandwidth:
        _min, _max = find_global_atm_bandwidth([atm_container])
        # use `global_left` if defined (min of wavelengts where exists intersection of atmospheric models)
        #   or current model left wavelength boundary
        # use `global_righ` if defined (max of wavelengts where exists intersection of atmospheric models) or current
        #    model right wavelength boundary
        left_bandwidth, right_bandwidth = kwargs.get("global_left", _min), kwargs.get("global_right", _max)

        if not kwargs.get("global_left") or not kwargs.get("global_right"):
            msg = (
                f"argument bandwidth is out of bound for supplied atmospheric model\n"
                f"to avoid interpolation error in boundary wavelength, "
                f"bandwidth was defined as max {_max} and min {_min} "
                f"of wavelength in given model table\n"
                f"it might lead to error in atmosphere interpolation\n"
                f"to avoid this problem, please specify global_left and "
                f"global_right bandwidth as kwargs for given method and "
                f"make sure all models wavelengths are greater or equal to such limits"
            )
            warnings.warn(msg, stacklevel=2)

    return strip_to_bandwidth(
        atm_container,
        left_bandwidth,
        right_bandwidth,
        inplace=inplace,
    )


def strip_to_bandwidth(
    atm_container: AtmDataContainer,
    left_bandwidth: Float,
    right_bandwidth: Float,
    *,
    inplace: bool = False,
) -> AtmDataContainer:
    """Select wavelength points from ``atm_container`` inside a bandwidth.

    The function selects indices with wavelengths strictly between
    ``left_bandwidth`` and ``right_bandwidth``, extends the selection to
    include neighboring boundary points (to support accurate
    interpolation), and then calls
    :func:`extend_atm_container_on_bandwidth_boundary` to ensure the
    returned model has exact boundary values.

    :param atm_container: Atmosphere container to strip.
    :type atm_container: AtmDataContainer
    :param left_bandwidth: Left (minimum) wavelength boundary in
        Angstrom.
    :type left_bandwidth: Float
    :param right_bandwidth: Right (maximum) wavelength boundary in
        Angstrom.
    :type right_bandwidth: Float
    :param inplace: If ``True``, modify and return the original container;
        otherwise work on a deep copy and return it.
    :type inplace: bool
    :returns: The bandwidth-stripped :class:`AtmDataContainer`.
    :rtype: AtmDataContainer
    """
    # Select indices within bandwidth
    valid_indices = list(
        np.where(
            np.logical_and(
                np.greater(atm_container.model.wavelength, left_bandwidth),
                np.less(atm_container.model.wavelength, right_bandwidth),
            ),
        )[0],
    )

    # Extend selection to include boundary points
    left_extention_index = valid_indices[0] - 1 if valid_indices[0] >= 1 else 0
    right_extention_index = (
        valid_indices[-1] + 1 if valid_indices[-1] < atm_container.model.last_valid_index() else valid_indices[-1]
    )

    atm_cont = atm_container if inplace else deepcopy(atm_container)
    atm_cont.model = atm_cont.model[
        np.unique(
            [left_extention_index, *valid_indices, right_extention_index],
        )
    ]

    return extend_atm_container_on_bandwidth_boundary(
        atm_cont,
        left_bandwidth,
        right_bandwidth,
    )


def find_global_atm_bandwidth(
    atm_containers: list[AtmDataContainer],
) -> tuple[Float, Float]:
    """Find common wavelength coverage of atmosphere models.

    Computes the intersection of wavelength ranges from all supplied
    containers. Returns the highest minimum wavelength and the lowest
    maximum wavelength across all models.

    :param atm_containers: List of atmosphere containers.
    :type atm_containers: list[AtmDataContainer]
    :returns: Tuple of (min_wavelength, max_wavelength) for common coverage.
    :rtype: tuple[Float, Float]
    """
    # Vectorized extraction: get min of all maxima and max of all minima
    mins = np.array([atm.model.wavelength.min() for atm in atm_containers])
    maxs = np.array([atm.model.wavelength.max() for atm in atm_containers])
    return float(mins.max()), float(maxs.min())


def extend_atm_container_on_bandwidth_boundary(
    atm_container: AtmDataContainer,
    left_bandwidth: Float,
    right_bandwidth: Float,
) -> AtmDataContainer:
    """Ensure the container's first and last wavelength points match the requested bandwidth boundaries.

    The function interpolates the model using an Akima1D interpolator and
    then replaces the first and last samples with exact values at
    ``left_bandwidth`` and ``right_bandwidth``. If interpolation yields
    NaN for either boundary, :class:`AtmosphereError` is raised.

    :param atm_container: Atmosphere container to modify.
    :type atm_container: AtmDataContainer
    :param left_bandwidth: Left (minimum) wavelength boundary in
        Angstrom.
    :type left_bandwidth: Float
    :param right_bandwidth: Right (maximum) wavelength boundary in
        Angstrom.
    :type right_bandwidth: Float
    :returns: The modified :class:`AtmDataContainer` with exact boundary
        wavelength/flux values.
    :rtype: AtmDataContainer
    :raises AtmosphereError: If interpolation produces NaN values at the
        boundaries.
    """
    interpolator = interpolate.Akima1DInterpolator(
        atm_container.model.wavelength,
        atm_container.model.flux,
    )

    # Interpolate flux at exact boundary wavelengths
    on_border_flux: NDArray[Float] = interpolator([left_bandwidth, right_bandwidth])
    if np.isnan(on_border_flux).any():
        msg = "Interpolation on bandwidth boundaries resulted in NaN value."
        raise AtmosphereError(msg)

    atm_model: AtmModel = atm_container.model
    atm_model.wavelength[np.array([0, -1])] = [left_bandwidth, right_bandwidth]
    atm_model.flux[np.array([0, -1])] = [on_border_flux[0], on_border_flux[1]]
    atm_model.flux = np.round(atm_model.flux, 10)

    atm_container.model = atm_model
    return atm_container


def apply_passband(
    atm_containers: list[AtmDataContainer],
    passband: dict,
    **kwargs,
) -> dict[str, list[AtmDataContainer]]:
    """Apply passband throughput to atmosphere containers.

    For each passband, the function strips each container to the passband's
    wavelength coverage and multiplies the spectrum by the passband
    transmission (provided by a :class:`PassbandContainer.akima` callable).

    :param atm_containers: List of atmosphere containers to process.
    :type atm_containers: list[AtmDataContainer]
    :param passband: Mapping of passband name to
        :class:`elisa.observer.passband.PassbandContainer`.
    :type passband: dict[str, PassbandContainer]
    :param kwargs: Optional arguments forwarded to :func:`strip_to_bandwidth`.
    :returns: Mapping from passband name to a list of processed
        :class:`AtmDataContainer` objects.
    :rtype: dict[str, list[AtmDataContainer]]
    """
    passbanded_atm_containers = {}
    logger.debug("applying passband functions on given atmospheres")

    for band, band_container in passband.items():
        if band == "bolometric":
            band_container.left_bandwidth = kwargs.get(
                "global_left",
                band_container.left_bandwidth,
            )
            band_container.right_bandwidth = kwargs.get(
                "global_right",
                band_container.right_bandwidth,
            )

        passbanded_atm_containers[band] = []
        for atm_container in atm_containers:
            # Strip to passband-specific bandwidth
            atm_container = strip_to_bandwidth(  # noqa: PLW2901
                atm_container=deepcopy(atm_container),
                left_bandwidth=band_container.left_bandwidth,
                right_bandwidth=band_container.right_bandwidth,
                inplace=False,
            )

            # Apply passband response
            passband_throughput = np.nan_to_num(
                band_container.akima(atm_container.model.wavelength),
            )
            atm_container.model.flux *= passband_throughput
            passbanded_atm_containers[band].append(atm_container)

    logger.debug("passband application finished")
    return passbanded_atm_containers


def build_atm_validation_hypertable(atlas: str) -> dict:
    """Build a validation table for atmosphere model parameter bounds.

    Creates a nested dictionary mapping temperature → gravity values for
    each temperature in the atlas. Used to validate that log_g values are
    valid for their corresponding temperatures.

    :param atlas: Atmosphere atlas name (e.g., 'ck04', 'k93').
    :type atlas: str
    :returns: Dictionary with temperatures as keys and dicts of gravity and
        metallicity allowable values.
    :rtype: dict
    """
    atlas = validated_atlas(atlas)
    all_files = get_list_of_all_atm_tables(atlas)
    filenames = (Path(f).name for f in all_files)
    quantities = sorted(
        [parse_domain_quantities_from_atm_table_filename(f) for f in filenames],
        key=lambda x: x[0],
    )
    temp_qroups = itertools.groupby(quantities, key=lambda x: x[0])
    return {
        str(int(temp_qroup[0])): {
            "gravity": sorted(set(np.array(list(temp_qroup[1])).T[1])),
            "metallicity": atm_file_prefix_to_quantity_list("metallicity", atlas),
        }
        for temp_qroup in temp_qroups
    }


def is_out_of_bound(
    in_arr: NDArray | list,
    values: Float | Iterable,
    tolerance: Float,
) -> list[bool]:
    """Check if values are outside the bounds of an array with tolerance.

    :param in_arr: Reference array defining valid range.
    :type in_arr: NDArray | list
    :param values: Value(s) to check.
    :type values: Float | Iterable
    :param tolerance: Tolerance for out-of-bounds check.
    :type tolerance: Float
    :returns: List of booleans; True where value is out of bounds.
    :rtype: list[bool]
    """
    values = [values] if not isinstance(values, Iterable) else values
    top, bottom = max(in_arr) + tolerance, min(in_arr) - tolerance
    return [not bottom <= val <= top for val in values]


# pay attention to those methods below
# in the future for different atm model might happen that function won't be valid anymore


def validate_temperature(
    temperature: NDArray,
    atlas: str,
    *,
    _raise: bool = True,
) -> bool:
    """Validate that temperatures are within atlas bounds.

    :param temperature: Temperature values to validate in K.
    :type temperature: NDArray
    :param atlas: Atmosphere atlas name (e.g., 'ck04', 'k93').
    :type atlas: str
    :param _raise: If True, raise TemperatureError on invalid values;
        otherwise return False.
    :type _raise: bool
    :returns: True if all temperatures are valid, False otherwise.
    :rtype: bool
    :raises TemperatureError: If temperatures are out of bounds and _raise=True.
    """
    atlas = validated_atlas(atlas)
    allowed = sorted(atm_file_prefix_to_quantity_list("temperature", atlas))
    invalid = any(bool(allowed[-1] < t or t < allowed[0]) for t in temperature)
    if invalid:
        if _raise:
            msg = "Any temperature in system atmosphere is out of bound; it is usually caused by invalid physical model"
            raise TemperatureError(msg)
        return False
    return True


def validate_metallicity(
    metallicity: Float | Iterable,
    atlas: str,
    *,
    _raise: bool = True,
) -> bool:
    """Validate that metallicity is within atlas bounds.

    :param metallicity: Metallicity value(s) [Fe/H] to validate.
    :type metallicity: Float | Iterable
    :param atlas: Atmosphere atlas name (e.g., 'ck04', 'k93').
    :type atlas: str
    :param _raise: If True, raise MetallicityError on invalid values;
        otherwise return False.
    :type _raise: bool
    :returns: True if metallicity is valid, False otherwise.
    :rtype: bool
    :raises MetallicityError: If metallicity is out of bounds and _raise=True.
    """
    out_of_bound_tol = 0.1  # Tolerance for out-of-bounds values
    atlas = validated_atlas(atlas)
    allowed = sorted(atm_file_prefix_to_quantity_list("metallicity", atlas))
    out_of_bound = is_out_of_bound(allowed, metallicity, out_of_bound_tol)
    if any(out_of_bound):
        if _raise:
            min_allowed = min(allowed) - out_of_bound_tol
            max_allowed = max(allowed) + out_of_bound_tol
            msg = (
                f"Any metallicity in system atmosphere is out of bound, "
                f"allowed values are in range <{min_allowed}, {max_allowed}>; "
                f"it is usually caused by invalid physical model"
            )
            raise MetallicityError(msg)
        return False
    return True


def validate_logg_temperature_constraint(
    temperature: NDArray,
    log_g: NDArray,
    atlas: str,
    *,
    _raise: bool = True,
) -> bool:
    """Validate that log_g values are consistent with temperature in atlas.

    Checks that each log_g value is valid for its corresponding temperature
    according to the atmosphere atlas bounds.

    :param temperature: Temperature values in K.
    :type temperature: NDArray
    :param log_g: Surface gravity values in log10(cm/s²).
    :type log_g: NDArray
    :param atlas: Atmosphere atlas name (e.g., 'ck04', 'k93').
    :type atlas: str
    :param _raise: If True, raise GravityError on invalid values;
        otherwise return False.
    :type _raise: bool
    :returns: True if all log_g values are valid, False otherwise.
    :rtype: bool
    :raises GravityError: If log_g is out of bounds and _raise=True.
    """
    validation_hypertable = build_atm_validation_hypertable(atlas)
    allowed = sorted(atm_file_prefix_to_quantity_list("temperature", atlas))

    invalid = [
        is_out_of_bound(
            validation_hypertable[str(int(utils.find_nearest_value(allowed, t)[0]))]["gravity"],
            [g],
            0.1,
        )[0]
        for t, g in zip(temperature, log_g, strict=True)
    ]
    if np.any(invalid):
        if _raise:
            msg = (
                "Any gravity (log_g) in system atmosphere is out of bound; "
                "it is usually caused by invalid physical model"
            )
            raise GravityError(msg)
        return False
    return True


def validate_atm(
    temperature: NDArray,
    log_g: NDArray,
    metallicity: Float | Iterable,
    atlas: str,
    *,
    _raise: bool = True,
) -> bool:
    """Validate atmosphere parameters against atlas bounds.

    Runs validate_temperature, validate_metallicity, and
    validate_logg_temperature_constraint in sequence.

    :param temperature: Temperature values in K.
    :type temperature: NDArray
    :param log_g: Surface gravity in log10(cm/s²).
    :type log_g: NDArray
    :param metallicity: Metallicity value(s) [Fe/H].
    :type metallicity: Float | Iterable
    :param atlas: Atmosphere atlas name (e.g., 'ck04', 'k93').
    :type atlas: str
    :param _raise: If True, raise error on invalid values;
        otherwise return False.
    :type _raise: bool
    :returns: True if all atmosphere parameters are valid, False otherwise.
    :rtype: bool
    :raises TemperatureError, MetallicityError, GravityError: If
        respective parameters are out of bounds and _raise=True.
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


def atm_file_prefix_to_quantity_list(qname: str, atlas: str) -> list:
    """Get list of available values for atmosphere domain quantity.

    Retrieves the set of allowable values for a given quantity (temperature,
    gravity, or metallicity) for a specified atmosphere atlas.

    :param qname: Quantity name ('temperature', 'gravity', or 'metallicity').
    :type qname: str
    :param atlas: Atmosphere atlas name (e.g., 'castelli', 'ck04').
    :type atlas: str
    :returns: List of available values for the quantity in the atlas.
    :rtype: list
    """
    atlas = validated_atlas(atlas)
    return getattr(
        const,
        f"{str(atlas).upper()}_{settings.ATM_DOMAIN_QUANTITY_TO_VARIABLE_SUFFIX[qname]!s}",
    )


def validated_atlas(atlas: str) -> str:
    """Normalize and validate atmosphere atlas identifier.

    Converts various atlas name formats (e.g., 'castelli', 'ck04', 'k93')
    to their canonical short forms ('ck', 'k', etc.) used for file paths
    and prefixes.

    :param atlas: Atmosphere atlas name (e.g., 'castelli', 'ck04').
    :type atlas: str
    :returns: Canonical atlas identifier.
    :rtype: str
    :raises KeyError: If atlas is not in the supported list.
    """
    try:
        return settings.ATM_ATLAS_NORMALIZER[atlas]
    except KeyError as exc:
        allowed = ", ".join(settings.ATM_ATLAS_NORMALIZER.keys())
        msg = f"Incorrect atlas: {atlas}. Following are allowed: {allowed}"
        raise KeyError(msg) from exc


def parse_domain_quantities_from_atm_table_filename(filename: str) -> tuple[Float, Float, Float]:
    """Parse atmosphere table filename to extract domain quantities.

    Extracts temperature, log_g, and metallicity from a filename following
    the Castelli-Kurucz naming convention (e.g., 'ckm05_3500_g15.csv').

    :param filename: Atmosphere table filename or path.
    :type filename: str
    :returns: Tuple of (temperature, log_g, metallicity) extracted from filename.
    :rtype: tuple[Float, Float, Float]
    """
    return (
        get_temperature_from_atm_table_filename(filename),
        get_logg_from_atm_table_filename(filename),
        get_metallicity_from_atm_table_filename(filename),
    )


def get_metallicity_from_atm_table_filename(filename: str) -> Float:
    """Extract metallicity value from atmosphere table filename.

    Parses the metallicity component from the filename prefix
    (e.g., 'ckm05' → -0.5, 'ckp02' → 0.2).

    :param filename: Atmosphere table filename or path.
    :type filename: str
    :returns: Metallicity [Fe/H] value.
    :rtype: Float
    """
    m = str(filename).split("_")[0][-3:]
    sign = 1 if str(m).startswith("p") else -1
    value = float(m[1:]) / 10.0
    return value * sign


def get_temperature_from_atm_table_filename(filename: str) -> Float:
    """Extract temperature value from atmosphere table filename.

    Parses the temperature component from the filename
    (e.g., 'ckm05_3500_g15.csv' → 3500).

    :param filename: Atmosphere table filename or path.
    :type filename: str
    :returns: Effective temperature in K.
    :rtype: Float
    """
    return float(str(filename).split("_")[1])


def get_logg_from_atm_table_filename(filename: str) -> Float:
    """Extract surface gravity value from atmosphere table filename.

    Parses the log_g component from the filename
    (e.g., 'ckm05_3500_g15.csv' → 1.5).

    :param filename: Atmosphere table filename or path.
    :type filename: str
    :returns: Surface gravity in log10(cm/s²).
    :rtype: Float
    """
    filename = filename if not str(filename).endswith(".csv") else str(filename).replace(".csv", "")
    g = str(filename).split("_")[2][1:]
    return int(g) / 10.0


def get_atm_table_filename(
    temperature: Float,
    log_g: Float,
    metallicity: Float,
    atlas: str,
) -> str:
    """Construct atmosphere table filename from parameters.

    Generates a filename following Castelli-Kurucz naming convention
    based on the supplied atmospheric parameters.

    :param temperature: Effective temperature in K.
    :type temperature: Float
    :param log_g: Surface gravity in log10(cm/s²).
    :type log_g: Float
    :param metallicity: Metallicity [Fe/H].
    :type metallicity: Float
    :param atlas: Atmosphere atlas name (e.g., 'castelli', 'ck04').
    :type atlas: str
    :returns: Filename for the atmosphere table.
    :rtype: str
    """
    prefix = validated_atlas(atlas)
    return (
        f"{prefix}{utils.numeric_metallicity_to_string(metallicity)}_"
        f"{int(temperature)}_{utils.numeric_logg_to_string(log_g)}.csv"
    )


def get_atm_directory(metallicity: Float, atlas: str) -> str:
    """Construct atmosphere table directory name from parameters.

    Generates a directory name following Castelli-Kurucz naming convention
    based on metallicity.

    :param metallicity: Metallicity [Fe/H].
    :type metallicity: Float
    :param atlas: Atmosphere atlas name (e.g., 'castelli', 'ck04').
    :type atlas: str
    :returns: Directory name for the atmosphere table.
    :rtype: str
    """
    prefix = validated_atlas(atlas)
    return f"{prefix}{utils.numeric_metallicity_to_string(metallicity)}"


def get_atm_table(
    temperature: Float,
    log_g: Float,
    metallicity: Float,
    atlas: str,
) -> pd.DataFrame:
    """Load atmosphere table for given parameters from CSV file.

    Reads and returns a DataFrame containing flux and wavelength columns
    for the specified atmospheric parameters.

    :param temperature: Effective temperature in K.
    :type temperature: Float
    :param log_g: Surface gravity in log10(cm/s²).
    :type log_g: Float
    :param metallicity: Metallicity [Fe/H].
    :type metallicity: Float
    :param atlas: Atmosphere atlas name (e.g., 'castelli', 'ck04').
    :type atlas: str
    :returns: DataFrame with flux and wavelength columns.
    :rtype: pd.DataFrame
    :raises FileNotFoundError: If the atmosphere table file is not found.
    """
    source = settings.ATLAS_TO_BASE_DIR[atlas]
    directory = get_atm_directory(metallicity, atlas)
    filename = get_atm_table_filename(temperature, log_g, metallicity, atlas)
    path = Path(source) / directory / filename if directory is not None else Path(source) / filename

    if not path.is_file():
        msg = f"There is no file like {path}"
        raise FileNotFoundError(msg)

    # noinspection PyArgumentList
    return pd.read_csv(path, dtype=settings.ATM_MODEL_DATAFARME_DTYPES)


def get_list_of_all_atm_tables(atlas: str) -> list[str]:
    """Get list of all available atmosphere table files in configured location.

    Recursively searches the configured atlas directory for CSV files
    containing atmosphere table data.

    :param atlas: Atmosphere atlas name (e.g., 'castelli', 'ck04').
    :type atlas: str
    :returns: List of absolute file paths to atmosphere table CSV files.
    :rtype: list[str]
    """
    source = settings.ATLAS_TO_BASE_DIR[validated_atlas(atlas)]
    matches = []
    for root, _dirnames, filenames in Path(source).walk():
        for filename in filenames:
            if filename.endswith((".csv",)):
                matches.extend([str(root / filename)])
    return matches


def multithread_atm_tables_reader(
    path_queue: Queue,
    error_queue: Queue,
    result_queue: Queue,
) -> None:
    """Read atmosphere CSV files in a worker thread.

    Continuously reads file paths from path_queue, loads atmosphere data,
    and places results in result_queue. Terminates on "TERMINATOR" message
    or if an error occurs.

    :param path_queue: Queue supplying (index, file_path) tuples to read.
    :type path_queue: Queue;
    :param error_queue: Queue for error reporting from worker threads.
    :type error_queue: Queue;
    :param result_queue: Queue for (index, AtmDataContainer) results.
    :type result_queue: Queue;
    """
    while True:
        args = path_queue.get(timeout=1)
        if args == "TERMINATOR":
            break
        if not error_queue.empty():
            break
        index, file_path = args
        if file_path is None:
            result_queue.put((index, None))
            continue
        try:
            dtype_map = {"flux": FLOAT, "wave": FLOAT}
            t, l, m = parse_domain_quantities_from_atm_table_filename(Path(file_path).name)  # noqa: E741
            # noinspection PyArgumentList
            atm_container = AtmDataContainer(pd.read_csv(file_path, dtype=dtype_map), t, l, m, fpath=file_path)
            result_queue.put((index, atm_container))
        except Exception as we:  # noqa: BLE001
            error_queue.put(we)
            break


def multithread_atm_tables_reader_runner(fpaths: list[str] | tuple[str]) -> Queue:
    """Run multithread reader for atmosphere table CSV files.

    Spawns multiple worker threads to read atmosphere tables in parallel
    and collects results in a queue.

    :param fpaths: List of file paths to atmosphere tables.
    :type fpaths: Iterable[str]
    :returns: Queue containing (index, AtmDataContainer) tuples.
    :rtype: Queue
    :raises AtmosphereError: If any file path does not exist or
        if a worker thread encounters an error.
    """
    n_threads = settings.NUMBER_OF_THREADS

    path_queue: Queue = Queue(maxsize=len(fpaths) + n_threads)
    result_queue: Queue = Queue()
    error_queue: Queue = Queue()

    threads = []
    try:
        for index, fpath in enumerate(fpaths):
            if not Path(fpath).is_file():
                logger.debug("accessing atmosphere file %s", fpath)
                msg = (
                    f"file {fpath} doesn't exist. Your atmosphere tables are "
                    f"either not properly installed or atmosphere parameters "
                    f"of your stellar model are not covered by the currently "
                    f"used table."
                )
                raise AtmosphereError(msg)
            path_queue.put((index, fpath))

        for _ in range(n_threads):
            path_queue.put("TERMINATOR")

        logger.debug("initialising multithread atm table reader")
        for _ in range(n_threads):
            t = Thread(
                target=multithread_atm_tables_reader,
                args=(path_queue, error_queue, result_queue),
            )
            threads.append(t)
            t.daemon = True
            t.start()

        for t in threads:
            t.join()
        logger.debug("atm multithread reader finished all jobs")
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, terminating atmosphere table reader threads.")
        raise
    finally:
        if not error_queue.empty():
            raise error_queue.get()
    return result_queue


def compute_normal_radiances(
    matrices_dict: dict,
    *,
    flux_mult: Float = 1.0,
    wave_mult: Float = 1.0,
) -> dict[str, NDArray[Float]]:
    """Compute integrated normal radiance for each passband.

    The input mapping should provide, for every passband, a dictionary
    with keys given by :data:`settings.ATM_MODEL_DATAFRAME_FLUX` and
    :data:`settings.ATM_MODEL_DATAFRAME_WAVE`.

    :param matrices_dict: Mapping passband → {'flux': NDArray, 'wave': NDArray}.
    :type matrices_dict: dict[str, dict[str, NDArray]]
    :param flux_mult: Multiplicative factor to convert flux to desired
        units (typically to SI).
    :type flux_mult: Float
    :param wave_mult: Multiplicative factor to convert wavelength to
        desired units (typically meters).
    :type wave_mult: Float
    :returns: Mapping passband → integrated normal radiance arrays (one
        value per model/face) as numpy arrays of type FLOAT.
    :rtype: dict[str, numpy.typing.NDArray[elisa.types.Float]]
    """
    return {
        band: compute_normal_intensity(
            spectral_flux=dflux[settings.ATM_MODEL_DATAFRAME_FLUX],
            wavelength=dflux[settings.ATM_MODEL_DATAFRAME_WAVE],
            flux_mult=flux_mult,
            wave_mult=wave_mult,
        )
        for band, dflux in matrices_dict.items()
    }


def compute_normal_intensity(
    spectral_flux: NDArray,
    wavelength: NDArray,
    *,
    flux_mult: Float = 1.0,
    wave_mult: Float = 1.0,
) -> NDArray:
    """Integrate spectral flux over wavelength using Simpson's rule.

    The function integrates each row of ``spectral_flux`` over
    ``wavelength`` and applies ``flux_mult`` and ``wave_mult`` to convert
    to final units (for example to SI units). The returned array contains
    one integrated intensity per input row.

    :param spectral_flux: Array with shape (N_faces, N_wavelengths).
    :type spectral_flux: NDArray
    :param wavelength: Wavelength grid (1D array) used for integration.
    :type wavelength: NDArray
    :param flux_mult: Multiplicative factor applied to flux values.
    :type flux_mult: Float
    :param wave_mult: Multiplicative factor applied to wavelength.
    :type wave_mult: Float
    :returns: Integrated normal flux for each face.
    :rtype: NDArray
    """
    return flux_mult * wave_mult * up.simps(spectral_flux, wavelength, axis=1)


def compute_integral_si_intensity_from_passbanded_dict(
    passbaned_dict: dict,
) -> dict[str, list[IntensityContainer]]:
    """Compute integral SI intensity for each passband's atmosphere data.

    :param passbaned_dict: Dictionary mapping passband names to lists of
        atmosphere containers.
    :type passbaned_dict: dict
    :returns: Dictionary mapping passband names to lists of
        IntensityContainer objects.
    :rtype: dict[str, list[IntensityContainer]]
    """
    return {
        band: compute_integral_si_intensity_from_atm_data_containers(
            passbanded_atm,
        )
        for band, passbanded_atm in passbaned_dict.items()
    }


def compute_integral_si_intensity_from_atm_data_containers(
    atm_data_containers: Iterable[AtmDataContainer],
) -> list[IntensityContainer]:
    """Compute integrated SI intensity from atmosphere data containers.

    Integrates flux over wavelength for each container, converting to SI
    units using the container's flux and wavelength multiplicators.

    :param atm_data_containers: Atmosphere containers with flux and
        wavelength data.
    :type atm_data_containers: Iterable[AtmDataContainer]
    :returns: List of IntensityContainer objects with integrated flux and
        atmospheric parameters.
    :rtype: list[IntensityContainer]
    """
    return [
        IntensityContainer(
            intensity=const.PI
            * up.simps(
                adc.model.flux * adc.flux_to_si_mult,
                adc.model.wavelength * adc.wave_to_si_mult,
            ),
            temperature=adc.temperature,
            log_g=adc.log_g,
            metallicity=adc.metallicity,
        )
        for adc in atm_data_containers
    ]


def unique_atm_fpaths(fpaths: list[str]) -> tuple[set[str], dict[str, list[int]]]:
    """Identify unique atmosphere table paths and create index map.

    Returns a set of unique file paths and a mapping from each unique path
    to the indices where it appears in the input list.

    :param fpaths: List of file paths to atmosphere tables.
    :type fpaths: list[str]
    :returns: Tuple of (unique_paths_set, path_to_indices_map). The map
        allows remapping results indexed by unique path back to original
        indices.
    :rtype: tuple[set[str], dict[str, list[int]]]
    """
    fpaths_set = set(fpaths)
    fpaths_map = {key: [] for key in fpaths_set}
    for idx, key in enumerate(fpaths):
        fpaths_map[key].append(idx)
    return fpaths_set, fpaths_map


def remap_passbanded_unique_atms_to_origin(
    passbanded_containers: dict,
    fpaths_map: dict[str, list[int]],
) -> dict:
    """Remap passband-filtered atmosphere containers to original ordering.

    :param passbanded_containers: Dictionary mapping passband names to lists
        of atmosphere containers indexed by unique file path.
    :type passbanded_containers: dict
    :param fpaths_map: Mapping from unique file paths to original indices.
    :type fpaths_map: dict[str, list[int]]
    :returns: Dictionary mapping passband names to reordered container lists.
    :rtype: dict
    """
    return {band: remap_unique_atm_container_to_origin(atm, fpaths_map) for band, atm in passbanded_containers.items()}


def remap_unique_atm_container_to_origin(
    models: list[AtmDataContainer],
    fpaths_map: dict[str, list[int]],
) -> NDArray:
    """Remap atmosphere containers to original index ordering.

    Assigns containers to an object array indexed by original position.
    Note: Containers are mutable, so modifying any container affects
    all references to it.

    :param models: List of unique atmosphere containers.
    :type models: list[AtmDataContainer]
    :param fpaths_map: Mapping from file path to original indices.
    :type fpaths_map: dict[str, list[int]]
    :returns: Object array with containers assigned to original indices.
    :rtype: NDArray
    """
    models_arr = np.empty(
        max(list(itertools.chain.from_iterable(fpaths_map.values()))) + 1,
        dtype="O",
    )
    for model in models:
        models_arr[fpaths_map[model.fpath]] = model
    return models_arr


def read_unique_atm_tables(
    fpaths: list[str] | tuple[str],
) -> tuple[list[AtmDataContainer], dict[str, list[int]]]:
    """Load unique atmosphere table files from disk or buffer.

    Checks the atmosphere buffer cache first; loads uncached files using
    multithreading. Returns unique containers and a mapping to original
    file list indices.

    :param fpaths: List of file paths to atmosphere tables.
    :type fpaths: Iterable[str]
    :returns: Tuple of (unique_containers_list, index_map). Use the map to
        remap results back to original indices.
    :rtype: tuple[list[AtmDataContainer], dict[str, list[int]]]
    """
    fpaths, fpaths_map = unique_atm_fpaths(fpaths)

    # Check buffer for cached atmosphere tables
    models, load_fpaths = [], []
    for fpath in fpaths:
        if fpath in buffer.ATMOSPHERE_TABLES:
            models.append(buffer.ATMOSPHERE_TABLES[fpath])
        else:
            load_fpaths.append(fpath)

    # Load uncached tables using multithreading
    if len(load_fpaths) > 0:
        result_queue = multithread_atm_tables_reader_runner(load_fpaths)
        loaded_models = [qval[1] for qval in utils.IterableQueue(result_queue) if qval[1] is not None]
        # Cache loaded atmospheres
        for ii, fpath in enumerate(load_fpaths):
            buffer.ATMOSPHERE_TABLES[fpath] = loaded_models[ii]
        models += loaded_models

    # Reduce buffer size if needed
    buffer.reduce_buffer(buffer.ATMOSPHERE_TABLES)
    return models, fpaths_map


def find_atm_si_multiplicators(
    atm_containers: Iterable[AtmDataContainer],
) -> tuple[Float, Float]:
    """Extract flux and wavelength multiplicators from atmosphere containers.

    Assumes all containers have identical multiplicators and returns values
    from the first container.

    :param atm_containers: List of atmosphere containers.
    :type atm_containers: Iterable[AtmDataContainer]
    :returns: Tuple of (flux_multiplicator, wavelength_multiplicator).
    :rtype: tuple[Float, Float]
    :raises ValueError: If no valid container is supplied.
    """
    for atm_container in atm_containers:
        return atm_container.flux_to_si_mult, atm_container.wave_to_si_mult
    msg = "No valid atmospheric container has been supplied to method."
    raise ValueError(msg)


def find_atm_defined_wavelength(
    atm_containers: Iterable[AtmDataContainer],
) -> NDArray:
    """Extract wavelength array from the first atmosphere container.

    Assumes all containers have wavelengths already aligned.

    :param atm_containers: List of atmosphere containers.
    :type atm_containers: Iterable[AtmDataContainer]
    :returns: Wavelength array from the first container.
    :rtype: NDArray
    :raises AtmosphereError: If no valid container is supplied.
    """
    for atm_container in atm_containers:
        return atm_container.model.wavelength
    msg = "No valid atmospheric container has been supplied to method."
    raise AtmosphereError(msg)


def remap_passbanded_unique_atms_to_matrix(
    passbanded_containers: dict,
    fpaths_map: dict[str, list[int]],
) -> dict[str, NDArray]:
    """Create flux matrices for each passband from unique containers.

    Converts a dictionary of unique containers (indexed by passband and
    uniqueness) to a matrix form suitable for vectorized interpolation.

    :param passbanded_containers: Dictionary mapping passband names to lists
        of unique atmosphere containers.
    :type passbanded_containers: dict
    :param fpaths_map: Mapping from file paths to original indices.
    :type fpaths_map: dict[str, list[int]]
    :returns: Dictionary mapping passband names to flux matrices.
    :rtype: dict[str, NDArray]
    """
    return {band: remap_passbanded_unique_atm_to_matrix(atm, fpaths_map) for band, atm in passbanded_containers.items()}


def remap_passbanded_unique_atm_to_matrix(
    atm_containers: Iterable[AtmDataContainer],
    fpaths_map: dict[str, list[int]],
) -> NDArray:
    """Create a flux matrix from atmosphere containers for a passband.

    Builds a 2D matrix where each row contains flux values for a unique
    atmosphere, indexed by the original face/container position from fpaths_map.

    :param atm_containers: List of unique atmosphere containers.
    :type atm_containers: Iterable[AtmDataContainer]
    :param fpaths_map: Mapping from file paths to original indices.
    :type fpaths_map: dict[str, list[int]]
    :returns: Flux matrix with shape (total_faces, wavelengths).
    :rtype: NDArray
    """
    # Optimize total calculation with max of flattened indices
    all_indices = list(itertools.chain.from_iterable(fpaths_map.values()))
    total = max(all_indices) + 1

    wavelengths_defined = find_atm_defined_wavelength(atm_containers)
    wavelengths_length = len(wavelengths_defined)
    models_matrix = up.zeros((total, wavelengths_length))

    # Vectorized assignment: collect indices and values, then assign
    for atm_container in atm_containers:
        indices = fpaths_map[atm_container.fpath]
        models_matrix[indices] = atm_container.model.flux

    return models_matrix


def correct_normal_radiance_to_optical_depth(
    normal_radiances: dict,
    ld_cfs: dict,
) -> dict:
    """Correct normal radiance values for optical depth effects using limb darkening coefficients.

    Correcting normal radiance values by increment that will correct inacuracy caused by using too shallow optical depth
    for the middle of the disk. Correction was derived analytically from spherical model.

    :param normal_radiances: dict; dict(component: dict(filter: normal radiances for each face))
    :param ld_cfs: dict; dict(component: dict(filter: limb darkening coefficients for each face))
    :return: dict;
    """
    for star, component_normal_radiances in normal_radiances.items():
        ld_coefficients = ld_cfs[star]["bolometric"].T

        coeff = ld.calculate_integrated_limb_darkening_factor(limb_darkening_law=settings.LIMB_DARKENING_LAW,
                                                              coefficients=ld_coefficients)

        normal_radiances[star] = {
            band: normal_radiance / coeff for band, normal_radiance in component_normal_radiances.items()
        }

    return normal_radiances


def planck_function(
        wavelength: NDArray | Float,
        temperature: NDArray | Float,
) -> NDArray | Float:
    """Evaluate the Planck function for given wavelength(s) and temperature.

    Computes monochromatic spectral radiance (spectral flux) using the
    standard Planck formula. Input ``wavelength`` must be in meters and
    ``temperature`` in Kelvin. The function accepts scalars or numpy
    arrays and returns a scalar or array accordingly.

    :param wavelength: Wavelength(s) in meters.
    :type wavelength: NDArray | Float
    :param temperature: Temperature value(s) in Kelvin.
    :type temperature: NDArray | Float
    :returns: Spectral radiance corresponding to the inputs.
    :rtype: NDArray | Float
    """
    h = (2.0 * const.PLANCK_CONST * const.C ** 2) / np.power(wavelength, 5)
    e = (const.PLANCK_CONST * const.C) / (
            wavelength * const.BOLTZMAN_CONST * temperature
    )
    return h / (np.exp(e) - 1.0)


def get_standard_wavelengths() -> NDArray:
    """Load the standard wavelength grid used by atmosphere tables.

    Reads the JSON file located at ``settings.DATA_PATH / 'wavelength.json'``
    and returns a numpy array of wavelengths (in Angstrom).

    :returns: 1D array of wavelengths in Angstrom.
    :rtype: NDArray
    """
    data_file = Path(settings.DATA_PATH) / "wavelength.json"
    with Path(data_file).open("+r") as f:
        return np.array(json.loads(f.read()))
