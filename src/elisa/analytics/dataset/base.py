import numpy as np
import astropy.units as au
import pandas as pd

from elisa.analytics.dataset.transform import (
    RVDataProperties,
    LCDataProperties
)
from elisa.logger import getLogger
from elisa import utils, units
from elisa.conf import config
from abc import ABCMeta
from elisa.utils import is_empty
from copy import copy, deepcopy
from elisa.analytics.dataset.graphic import plot

logger = getLogger('analytics.dataset.base')


def convert_data(data, unit, to_unit):
    """
    Converts data to desired format or leaves it dimensionless.

    :param data: numpy.array;
    :param unit: astropy.unit;
    :param to_unit: astropy.unit;
    :return: numpy.array;
    """
    return data if unit == au.dimensionless_unscaled else (data * unit).to(to_unit).value


def convert_flux(data, unit, zero_point=None):
    """
    If data are in magnitudes, they are converted to normalized flux.

    :param data: numpy.array;
    :param unit: astropy.unit.Unit;
    :param zero_point: float;
    :return: numpy.array;
    """
    if unit == au.mag:
        if zero_point is None:
            raise ValueError('You supplied your data in magnitudes. Please also specify '
                             'a zero point using keyword argument `reference_magnitude`.')
        else:
            data = utils.magnitude_to_flux(data, zero_point)

    return data


def convert_flux_error(error, unit, zero_point=None):
    """
    If data an its errors are in magnitudes, they are converted to normalized flux.

    :param error: numpy.array;
    :param unit: astropy.unit.Unit;
    :param zero_point: float;
    :return: numpy.array;
    """
    if unit == au.mag:
        if zero_point is None:
            raise ValueError('You supplied your data in magnitudes. Please also specify '
                             'a zero point using keyword argument `reference_magnitude`.')
        else:
            error = utils.magnitude_error_to_flux_error(error)
    return error


def convert_unit(unit, to_unit):
    """
    Converts to desired unit  or leaves it dimensionless.

    :param unit: astropy.unit;
    :param to_unit: astropy.unit;
    :return: astropy.unit;
    """
    return unit if unit == au.dimensionless_unscaled else to_unit


def read_data_file(filename, data_columns, delimiter=config.DELIM_WHITESPACE):
    """
    Function loads observation datafile. Rows with column names and comments should start with `#`.
    It deals with missing data by omitting given line

    :param delimiter: str; regex to define columns separtor
    :param filename: str;
    :param data_columns: Tuple;
    :return: numpy.array;
    """
    data = pd.read_csv(filename, header=None, comment='#', delimiter=delimiter,
                       error_bad_lines=False, engine='python')[list(data_columns)]
    data = data.apply(lambda s: pd.to_numeric(s, errors='coerce')).dropna()
    return data.to_numpy(dtype=float)


class DataSet(metaclass=ABCMeta):
    TRANSFORM_PROPERTIES_CLS = None
    ID = 1

    def __init__(self, name=None, **kwargs):
        # initial kwargs
        self.kwargs = copy(kwargs)
        self.plot = plot.Plot(self)

        if is_empty(name):
            self.name = str(DataSet.ID)
            logger.debug(f"name of class instance {self.__class__.__name__} set to {self.name}")
            DataSet.ID += 1
        else:
            self.name = str(name)

        # initializing parmas to default values
        self.x_data = np.array([])
        self.y_data = np.array([])
        self.y_err = None

        self.check_data_validity(**kwargs)

    def transform_input(self, **kwargs):
        return self.__class__.TRANSFORM_PROPERTIES_CLS.transform_input(**kwargs)

    def copy(self):
        return deepcopy(self)

    @staticmethod
    def check_data_validity(**kwargs):
        if not np.shape(kwargs['x_data']) == np.shape(kwargs['y_data']):
            raise ValueError('`x_data` and `y_data` are not of the same shape.')
        if 'y_err' in kwargs.keys():
            if not np.shape(kwargs['x_data']) == np.shape(kwargs['y_err']):
                raise ValueError('`y_err` are not of the same shape to `x_data` and `y_data`.')

        # check for nans
        if np.isnan(kwargs['x_data']).any():
            raise ValueError('`x_data` contains NaN')
        if np.isnan(kwargs['y_data']).any():
            raise ValueError('`y_data` contains NaN')
        if 'y_err' in kwargs.keys():
            if np.isnan(kwargs['y_err']).any():
                raise ValueError('`y_err` contains NaN')

    @classmethod
    def load_from_file(cls, filename, x_unit=None, y_unit=None, data_columns=None,
                       delimiter=config.DELIM_WHITESPACE, **kwargs):
        """
        Function loads a RV/LC measurements from text file.

        :param filename: str; name of the file
        :param x_unit: astropy.unit.Unit;
        :param y_unit: astropy.unit.Unit;
        :param data_columns: Tuple; ordered tuple with column indices of x_data, y_data, y_errors
        :param delimiter: str; regex to define columns separtor
        :param kwargs: Dict;
        :**kwargs options**:
            * **reference_magnitude** * -- float; zero point for magnitude conversion in case of LCData

        :return: Union[RVData, LCData];
        """
        data_columns = (0, 1, 2) if data_columns is None else data_columns
        data = read_data_file(filename, data_columns, delimiter=delimiter)

        try:
            errs = data[:, 2]
        except IndexError:
            errs = None
        return cls(x_data=data[:, 0],
                   y_data=data[:, 1],
                   y_err=errs,
                   x_unit=x_unit,
                   y_unit=y_unit,
                   **kwargs)

    from_file = load_from_file


class RVData(DataSet):
    """
    Child class of elisa.analytics.dataset.base.Dataset class storing radial velocity measurement.

    Input parameters:

    :param x_data: numpy.array; time or observed phases
    :param y_data: numpy.array; radial velocities
    :param y_err: numpy.array; radial velocity errors - optional
    :param x_unit: astropy.unit.Unit; if `None` or `astropy.unit.dimensionless_unscaled` is given,
                                      the `x_data are regarded as phases, otherwise if unit is convertible
                                      to days, the `x_data` are regarded to be in JD
    :param y_unit: astropy.unit.Unit; velocity unit of the observed radial velocities and its errors
    """

    MANDATORY_KWARGS = config.DATASET_MANDATORY_KWARGS
    OPTIONAL_KWARGS = config.DATASET_OPTIONAL_KWARGS
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS
    TRANSFORM_PROPERTIES_CLS = RVDataProperties

    __slots__ = ALL_KWARGS

    def __init__(self, name=None, **kwargs):
        utils.invalid_kwarg_checker(kwargs, self.__slots__, RVData)
        utils.check_missing_kwargs(self.MANDATORY_KWARGS, kwargs, instance_of=RVData)
        super().__init__(name, **kwargs)

        kwargs = self.transform_input(**kwargs)

        # conversion to base units
        kwargs = self.convert_arrays(**kwargs)
        self.check_data_validity(**kwargs)
        self.init_parameters(**kwargs)

    def init_parameters(self, **kwargs):
        logger.debug(f"initialising properties of class instance {self.__class__.__name__}")
        for kwarg in RVData.ALL_KWARGS:
            if kwarg in kwargs:
                setattr(self, kwarg, kwargs[kwarg])

    @staticmethod
    def convert_arrays(**kwargs):
        """
        Converting data and units to its base counterparts or keeping them dimensionless.

        :param kwargs: Dict;
        :return: Dict;
        """
        # converting x-axis
        kwargs['x_data'] = convert_data(kwargs['x_data'], kwargs['x_unit'], units.PERIOD_UNIT)
        kwargs['x_unit'] = convert_unit(kwargs['x_unit'], units.PERIOD_UNIT)

        # converting y-axis
        kwargs['y_data'] = convert_data(kwargs['y_data'], kwargs['y_unit'], units.VELOCITY_UNIT)

        # convert errors
        if 'y_err' in kwargs.keys():
            kwargs['y_err'] = convert_data(kwargs['y_err'], kwargs['y_unit'], units.VELOCITY_UNIT)
        kwargs['y_unit'] = convert_unit(kwargs['y_unit'], units.VELOCITY_UNIT)

        return kwargs


class LCData(DataSet):
    """
        Child class of elisa.analytics.dataset.base.Dataset class storing radial velocity measurement.

        Input parameters:

        :param x_data: numpy.array; time or observed phases
        :param y_data: numpy.array; light curves
        :param y_err: numpy.array; light curve errors - optional
        :param x_unit: astropy.unit.Unit; if `None` or `astropy.unit.dimensionless_unscaled` is given,
                                          the `x_data` are regarded as phases, otherwise if unit is convertible
                                          to days, the `x_data` are regarded to be in JD
        :param y_unit: astropy.unit.Unit; velocity unit of the observed flux and its errors
    """
    MANDATORY_KWARGS = config.DATASET_MANDATORY_KWARGS
    OPTIONAL_KWARGS = config.DATASET_OPTIONAL_KWARGS + ['reference_magnitude', 'passband']
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS
    TRANSFORM_PROPERTIES_CLS = LCDataProperties

    __slots__ = ALL_KWARGS

    def __init__(self, name=None, **kwargs):
        self.passband = None
        self.reference_magnitude = None

        utils.invalid_kwarg_checker(kwargs, self.__slots__, LCData)
        utils.check_missing_kwargs(self.MANDATORY_KWARGS, kwargs, instance_of=LCData)
        super().__init__(name, **kwargs)
        kwargs = self.transform_input(**kwargs)

        # conversion to base units
        kwargs = self.convert_arrays(**kwargs)
        self.check_data_validity(**kwargs)

        self.init_parameters(**kwargs)

    def init_parameters(self, **kwargs):
        logger.debug(f"initialising properties of class instance {self.__class__.__name__}")
        for kwarg in LCData.ALL_KWARGS:
            if kwarg in kwargs:
                setattr(self, kwarg, kwargs[kwarg])

    @staticmethod
    def convert_arrays(**kwargs):
        """
        Converting data and units to its base counterparts or keeping them dimensionless.

        :param kwargs: Dict;
        :return: Dict;
        """
        # converting x-axis
        kwargs['x_data'] = convert_data(kwargs['x_data'], kwargs['x_unit'], units.PERIOD_UNIT)
        kwargs['x_unit'] = convert_unit(kwargs['x_unit'], units.PERIOD_UNIT)
        kwargs['reference_magnitude'] = kwargs.get('reference_magnitude', None)

        # convert errors
        if 'y_err' in kwargs.keys():
            kwargs['y_err'] = convert_flux_error(kwargs['y_err'], kwargs['y_unit'],
                                                 zero_point=kwargs['reference_magnitude'])

        # converting y-axis
        kwargs['y_data'] = convert_flux(kwargs['y_data'], kwargs['y_unit'], zero_point=kwargs['reference_magnitude'])
        kwargs['y_unit'] = convert_unit(kwargs['y_unit'], units.dimensionless_unscaled)

        return kwargs
