"""Base classes for time series observational data storage and management."""

from __future__ import annotations

from abc import ABCMeta
from copy import copy, deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np

from elisa import logger as logger_module
from elisa import settings, utils
from elisa import units as u
from elisa.analytics.dataset import utils as dutils
from elisa.analytics.dataset.graphic import plot
from elisa.analytics.dataset.transform import LCDataProperties, RVDataProperties
from elisa.base.types import INT

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.types import TransformProperties

logger = logger_module.getLogger("analytics.dataset.base")


class DataSet(metaclass=ABCMeta):  # noqa: B024
    """Abstract base class for storing synthetic and observational time series data.

    This abstract class provides a framework for storing and managing time series
    data with their corresponding units and error information. Subclasses implement
    specific data handling for radial velocities (RVData) and light curves (LCData).

    :ivar name: Identifier name for the dataset instance.
    :ivar x_data: Independent variable data (times or phases).
    :ivar x_unit: Unit of the independent variable.
    :ivar y_data: Observable data (velocities, fluxes, magnitudes).
    :ivar y_unit: Unit of the observable.
    :ivar y_err: Errors or uncertainties in the observable.
    :ivar kwargs: Initial keyword arguments passed at initialization.
    :ivar plot: Plotting interface for the dataset.
    """

    TRANSFORM_PROPERTIES_CLS: TransformProperties | None = None
    ID: int = 1

    def __init__(self, name: str | None = None, **kwargs: Any) -> None:
        """Initialize a DataSet instance.

        :param name: Optional identifier for the dataset instance. If None or empty,
            auto-generates a unique ID.
        :type name: str | None
        :param kwargs: Additional keyword arguments for data initialization.
        :type kwargs: Any
        :raises ValueError: If data shapes or validity checks fail.
        """
        # initial kwargs
        self.kwargs: dict[str, Any] = copy(kwargs)
        self.plot: plot.Plot = plot.Plot(self)

        if utils.is_empty(name):
            self.name: str = str(DataSet.ID)
            logger.debug(
                "name of class instance %s set to %s",
                self.__class__.__name__,
                self.name,
            )
            DataSet.ID += 1
        else:
            self.name = str(name)

        # initializing params to default values
        self.x_data: NDArray[Any] = np.array([])
        self.x_unit: Any = None
        self.y_data: NDArray[Any] = np.array([])
        self.y_unit: Any = None
        self.y_err: NDArray[Any] | None = None

        self.check_data_validity(**kwargs)

    def transform_input(self, **kwargs: Any) -> dict[str, Any]:
        """Transform and validate input keyword arguments.

        :param kwargs: Input parameters to transform.
        :type kwargs: Any
        :returns: Transformed and validated parameters.
        :rtype: dict[str, Any]
        """
        return self.__class__.TRANSFORM_PROPERTIES_CLS.transform_input(**kwargs)

    def copy(self) -> DataSet:
        """Create a deep copy of the DataSet instance.

        :returns: A complete independent copy of this dataset.
        :rtype: DataSet
        """
        return deepcopy(self)

    @staticmethod
    def check_data_validity(**kwargs: Any) -> None:
        """Validate dataset integrity and consistency.

        Performs the following validations:

        - Checks that x_data and y_data have the same shape
        - Checks that y_err (if present) has the same shape as x_data and y_data
        - Checks for NaN values in all arrays

        :param kwargs: Dictionary containing data arrays (x_data, y_data, y_err).
        :type kwargs: Any
        :raises ValueError: If data shapes are incompatible or NaN values are found.
        """
        if np.shape(kwargs["x_data"]) != np.shape(kwargs["y_data"]):
            error_msg = "`x_data` and `y_data` are not of the same shape."
            raise ValueError(error_msg)
        if (
            "y_err" in kwargs
            and not utils.is_empty(kwargs["y_err"])
            and np.shape(kwargs["x_data"]) != np.shape(kwargs["y_err"])
        ):
            error_msg = "`y_err` are not of the same shape to `x_data` and `y_data`."
            raise ValueError(error_msg)

        # check for nans
        if np.isnan(kwargs["x_data"]).any():
            error_msg = "`x_data` contains NaN"
            raise ValueError(error_msg)
        if np.isnan(kwargs["y_data"]).any():
            error_msg = "`y_data` contains NaN"
            raise ValueError(error_msg)
        if "y_err" in kwargs and not utils.is_empty(kwargs["y_err"]) and np.isnan(kwargs["y_err"]).any():
            error_msg = "`y_err` contains NaN"
            raise ValueError(error_msg)

    @classmethod
    def load_from_file(
        cls,
        filename: str,
        x_unit: Any | None = None,
        y_unit: Any | None = None,
        data_columns: tuple[int, ...] | None = None,
        *,
        delimiter: str = settings.DELIM_WHITESPACE,
        downselect_ratio: float | None = None,
        **kwargs: Any,
    ) -> DataSet:
        """Load observational data from a file.

        Reads time series data from a delimited text file and creates a DataSet
        instance with the loaded data.

        :param filename: Path to the data file.
        :type filename: str
        :param x_unit: Unit of the independent variable (time or phase).
        :type x_unit: Any | None
        :param y_unit: Unit of the observable (velocity or flux).
        :type y_unit: Any | None
        :param data_columns: Tuple of column indices for (x_data, y_data, y_err).
            Defaults to (0, 1, 2).
        :type data_columns: tuple[int, ...] | None
        :param delimiter: Regex pattern defining column separator.
        :type delimiter: str
        :param downselect_ratio: Optional selection ratio in range (0, 1) to uniformly
            subsample the data. If provided, returns approximately 1/downselect_ratio
            of the original data points.
        :type downselect_ratio: float | None
        :param kwargs: Additional keyword arguments passed to the constructor
            (e.g., reference_magnitude for LCData).
        :type kwargs: Any
        :returns: DataSet instance with loaded data.
        :rtype: DataSet
        """
        data_columns = (0, 1, 2) if data_columns is None else data_columns
        data = dutils.read_data_file(filename, data_columns, delimiter=delimiter)

        if downselect_ratio is not None:
            idxs = np.arange(
                0,
                data.shape[0],
                step=int(1.0 / downselect_ratio),
                dtype=INT,
            )
            data = data[idxs]

        try:
            errs = data[:, 2]
        except IndexError:
            errs = None
        return cls(
            x_data=data[:, 0],
            y_data=data[:, 1],
            y_err=errs,
            x_unit=x_unit,
            y_unit=y_unit,
            **kwargs,
        )

    from_file = load_from_file

    def convert_to_phases(
        self,
        period: float,
        t0: float,
        *,
        centre: float = 0.0,
    ) -> None:
        """Convert independent variable from time to photometric phases.

        Transforms x_data from time units to dimensionless photometric phases
        according to the provided ephemeris parameters.

        :param period: Orbital period for phase folding.
        :type period: float
        :param t0: Reference time when phase equals zero (zero-epoch).
        :type t0: float
        :param centre: Phase value around which to center the data (default: 0.0).
        :type centre: float
        :returns: None (modifies dataset in-place).
        :rtype: None
        """
        self.x_data = utils.jd_to_phase(self.x_data, period, t0, centre=centre)
        self.x_unit = u.dimensionless_unscaled

    def convert_to_time(
        self,
        period: float,
        t0: float,
        *,
        to_unit: Any = u.DEFAULT_PERIOD_UNIT,
    ) -> None:
        """Convert independent variable from phases to time.

        Transforms x_data from dimensionless photometric phases to time units
        according to the provided ephemeris parameters.

        :param period: Orbital period used for phase folding.
        :type period: float
        :param t0: Reference time when phase equals zero (zero-epoch).
        :type t0: float
        :param to_unit: Target time unit for the conversion.
        :type to_unit: Any
        :returns: None (modifies dataset in-place).
        :rtype: None
        """
        self.x_data = self.x_data * period + t0
        self.x_unit = to_unit

    def smooth(self, *, method: str = "central_moving_average", **kwargs: Any) -> None:
        """Apply smoothing to the dataset using various methods.

        Smooths the observable data (y_data) using the specified method. Currently,
        supports central moving average binning. This function is intended for use
        with phased (dimensionless x_data) datasets.

        Available methods:

        - ``central_moving_average``: Bins data and computes average flux/RV within
          each bin, assigning result to bin center.

        :param method: Smoothing method identifier.
        :type method: str
        :param kwargs: Method-specific options. For ``central_moving_average``:

            - ``n_bins`` (int, default=100): Number of bins for data division
            - ``radius`` (int, default=2): Number of neighboring bins to include in average
            - ``cyclic_boundaries`` (bool, default=True): Treat data as periodic

        :type kwargs: Any
        :returns: None (modifies dataset in-place).
        :rtype: None
        :raises NotImplementedError: If specified method is not available.
        """
        available_methods = ["central_moving_average"]
        if method == "central_moving_average":
            n_bins: int = kwargs.get("n_bins", 100)
            radius: int = kwargs.get("radius", 2)
            cyclic_boundaries: bool = kwargs.get("cyclic_boundaries", True)
            dutils.central_moving_average(
                self,
                n_bins=n_bins,
                radius=radius,
                cyclic_boundaries=cyclic_boundaries,
            )
        else:
            error_msg = f"Method {method} is not implemented. Try one of these: {available_methods}"
            raise NotImplementedError(error_msg)


class RVData(DataSet):
    """Radial velocity time series data container.

    Stores and manages radial velocity measurements with corresponding time stamps,
    errors, and units. Automatically converts data to standard internal units
    (m/s for velocities, default period unit for time).

    Data can be initialized in two ways:

    1. **Direct initialization with arrays**:

       ::

           from elisa import RVData
           rv_data = RVData(
               x_data=times,
               x_unit=u.day,
               y_data=velocities,
               y_err=velocity_errors,
               y_unit=u.km / u.s
           )

    2. **Loading from file**:

       ::

           rv_data = RVData.load_from_file(
               filename,
               x_unit=u.day,
               y_unit=u.km / u.s,
               data_columns=(0, 1, 2)  # indices for time, RV, error
           )

    **Input Parameters:**

    :param x_data: Times of observation or photometric phases (numpy array).
    :param y_data: Radial velocity measurements (numpy array).
    :param y_err: Radial velocity uncertainties/errors (numpy array, optional).
    :param x_unit: Unit of x_data. If None or dimensionless_unscaled, treated as phases;
        otherwise must be convertible to time units (e.g., days).
    :param y_unit: Unit of y_data. Must be convertible to velocity units (e.g., m/s, km/s).
    :param name: Optional dataset identifier (auto-generated if not provided).
    """

    MANDATORY_KWARGS: tuple[str, ...] = settings.DATASET_MANDATORY_KWARGS
    OPTIONAL_KWARGS: tuple[str, ...] = settings.DATASET_OPTIONAL_KWARGS
    ALL_KWARGS: tuple[str, ...] = MANDATORY_KWARGS + OPTIONAL_KWARGS
    TRANSFORM_PROPERTIES_CLS: type = RVDataProperties

    __slots__ = ALL_KWARGS

    def __init__(self, name: str | None = None, **kwargs: Any) -> None:
        """Initialize an RVData instance.

        :param name: Optional identifier for the dataset instance.
        :type name: str | None
        :param kwargs: Keyword arguments including x_data, y_data, y_err, x_unit, y_unit.
        :type kwargs: Any
        :raises ValueError: If data validity checks fail or mandatory kwargs are missing.
        :raises TypeError: If invalid kwargs are provided.
        """
        utils.invalid_kwarg_checker(kwargs, self.__slots__, RVData)
        utils.check_missing_kwargs(self.MANDATORY_KWARGS, kwargs, instance_of=RVData)
        super().__init__(name, **kwargs)

        kwargs = self.transform_input(**kwargs)

        # conversion to base units
        kwargs = self.convert_arrays(**kwargs)
        self.check_data_validity(**kwargs)
        self.init_parameters(**kwargs)

    def init_parameters(self, **kwargs: Any) -> None:
        """Initialize instance attributes from validated parameters.

        :param kwargs: Validated parameter dictionary.
        :type kwargs: Any
        :returns: None.
        :rtype: None
        """
        logger.debug("initialising properties of class instance %s", self.__class__.__name__)
        for kwarg in RVData.ALL_KWARGS:
            if kwarg in kwargs:
                setattr(self, kwarg, kwargs[kwarg])

    @staticmethod
    def convert_arrays(**kwargs: Any) -> dict[str, Any]:
        """Convert data arrays and units to internal standard representations.

        Transforms input data and units to ELISa's internal units:

        - x_data: Converted to DEFAULT_PERIOD_UNIT (typically days)
        - y_data & y_err: Converted to VELOCITY_UNIT (typically m/s)
        - x_unit & y_unit: Updated to match converted units

        :param kwargs: Dictionary containing x_data, y_data, y_err, x_unit, y_unit.
        :type kwargs: Any
        :returns: Dictionary with converted data and units.
        :rtype: dict[str, Any]
        """
        # converting x-axis
        kwargs["x_data"] = dutils.convert_data(
            kwargs["x_data"],
            kwargs["x_unit"],
            u.DEFAULT_PERIOD_UNIT,
        )
        kwargs["x_unit"] = dutils.convert_unit(kwargs["x_unit"], u.DEFAULT_PERIOD_UNIT)

        # converting y-axis
        kwargs["y_data"] = dutils.convert_data(
            kwargs["y_data"],
            kwargs["y_unit"],
            u.VELOCITY_UNIT,
        )

        # convert errors
        if "y_err" in kwargs:
            kwargs["y_err"] = dutils.convert_data(
                kwargs["y_err"],
                kwargs["y_unit"],
                u.VELOCITY_UNIT,
            )
        kwargs["y_unit"] = dutils.convert_unit(kwargs["y_unit"], u.VELOCITY_UNIT)

        return kwargs


class LCData(DataSet):
    """Light curve time series data container.

    Stores and manages photometric light curve measurements with corresponding
    time stamps, errors, and units. Supports both flux and magnitude representations.
    Automatically converts data to standard internal units (dimensionless flux).

    Data can be initialized in two ways:

    1. **Direct initialization with arrays**:

       ::

           from elisa import LCData
           lc_data = LCData(
               x_data=times,
               x_unit=u.day,
               y_data=fluxes,
               y_err=flux_errors,
               y_unit=u.dimensionless_unscaled,
               reference_magnitude=0.0  # required if y_unit is magnitude
           )

    2. **Loading from file**:

       ::

           lc_data = LCData.load_from_file(
               filename,
               x_unit=u.day,
               y_unit=u.mag,
               reference_magnitude=0.0,
               data_columns=(0, 1, 2)  # indices for time, flux/mag, error
           )

    **Input Parameters:**

    :param x_data: Times of observation or photometric phases (numpy array).
    :param y_data: Flux measurements or magnitudes (numpy array).
    :param y_err: Flux/magnitude uncertainties/errors (numpy array, optional).
    :param x_unit: Unit of x_data. If None or dimensionless_unscaled, treated as phases;
        otherwise must be convertible to time units (e.g., days).
    :param y_unit: Unit of y_data. Either dimensionless (flux) or magnitude units.
        If magnitude is used, reference_magnitude must be provided.
    :param reference_magnitude: Reference magnitude for magnitude-to-flux conversion.
        Required if y_unit is magnitude-based.
    :param passband: Optional photometric passband identifier for the light curve.
    :param name: Optional dataset identifier (auto-generated if not provided).
    """

    MANDATORY_KWARGS: tuple[str, ...] = settings.DATASET_MANDATORY_KWARGS
    OPTIONAL_KWARGS: tuple[str, ...] = (
        *settings.DATASET_OPTIONAL_KWARGS,
        "reference_magnitude",
        "passband",
    )
    ALL_KWARGS: tuple[str, ...] = MANDATORY_KWARGS + OPTIONAL_KWARGS
    TRANSFORM_PROPERTIES_CLS: type = LCDataProperties

    __slots__ = ALL_KWARGS

    def __init__(self, name: str | None = None, **kwargs: Any) -> None:
        """Initialize an LCData instance.

        :param name: Optional identifier for the dataset instance.
        :type name: str | None
        :param kwargs: Keyword arguments including x_data, y_data, y_err, x_unit, y_unit,
            and optionally reference_magnitude and passband.
        :type kwargs: Any
        :raises ValueError: If data validity checks fail or mandatory kwargs are missing.
        :raises TypeError: If invalid kwargs are provided.
        """
        self.passband: str | None = None
        self.reference_magnitude: float | None = None

        utils.invalid_kwarg_checker(kwargs, self.__slots__, LCData)
        utils.check_missing_kwargs(self.MANDATORY_KWARGS, kwargs, instance_of=LCData)
        super().__init__(name, **kwargs)
        kwargs = self.transform_input(**kwargs)

        # conversion to base units
        kwargs = self.convert_arrays(**kwargs)
        self.check_data_validity(**kwargs)

        self.init_parameters(**kwargs)

    def init_parameters(self, **kwargs: Any) -> None:
        """Initialize instance attributes from validated parameters.

        :param kwargs: Validated parameter dictionary.
        :type kwargs: Any
        :returns: None.
        :rtype: None
        """
        logger.debug("initialising properties of class instance %s", self.__class__.__name__)
        for kwarg in LCData.ALL_KWARGS:
            if kwarg in kwargs:
                setattr(self, kwarg, kwargs[kwarg])

    @staticmethod
    def convert_arrays(**kwargs: Any) -> dict[str, Any]:
        """Convert data arrays and units to internal standard representations.

        Transforms input data and units to ELISa's internal units:

        - x_data: Converted to DEFAULT_PERIOD_UNIT (typically days)
        - y_data & y_err: Converted to dimensionless flux
        - x_unit & y_unit: Updated to match converted units

        Magnitudes are converted to normalized flux using the provided reference magnitude.

        :param kwargs: Dictionary containing x_data, y_data, y_err, x_unit, y_unit,
            and optionally reference_magnitude.
        :type kwargs: Any
        :returns: Dictionary with converted data and units.
        :rtype: dict[str, Any]
        """
        # converting x-axis
        kwargs["x_data"] = dutils.convert_data(
            kwargs["x_data"],
            kwargs["x_unit"],
            u.DEFAULT_PERIOD_UNIT,
        )
        kwargs["x_unit"] = dutils.convert_unit(kwargs["x_unit"], u.DEFAULT_PERIOD_UNIT)
        kwargs["reference_magnitude"] = kwargs.get("reference_magnitude")

        # convert errors
        if "y_err" in kwargs:
            kwargs["y_err"] = dutils.convert_flux_error(
                kwargs["y_err"],
                kwargs["y_unit"],
                zero_point=kwargs["reference_magnitude"],
            )

        # converting y-axis
        kwargs["y_data"] = dutils.convert_flux(
            kwargs["y_data"],
            kwargs["y_unit"],
            zero_point=kwargs["reference_magnitude"],
        )
        kwargs["y_unit"] = dutils.convert_unit(
            kwargs["y_unit"], u.dimensionless_unscaled,
        )

        return kwargs



