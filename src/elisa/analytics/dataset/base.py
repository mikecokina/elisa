import numpy as np
import astropy.units as u

from elisa.analytics.dataset.transform import RVDataProperties, LCDataProperties
from elisa.logger import getLogger
from elisa import utils, units
from elisa.conf import config
from abc import (
    ABCMeta,
    abstractmethod
)
from elisa.utils import is_empty
from copy import copy

logger = getLogger('analytics.dataset.base')


def convert_data(data, unit, to_unit):
    """
    Converts data to desired format or leaves it dimensionless.

    :param data: numpy.array;
    :param unit: astropy.unit;
    :param to_unit: astropy.unit;
    :return: numpy.array;
    """
    return data if unit == u.dimensionless_unscaled else (data * unit).to(to_unit).value


def convert_flux(data, unit, zero_point=None):
    """
    If data are in magnitudes, they are converted to normalized flux.

    :param data: numpy.array;
    :param unit: astropy.unit.Unit;
    :param zero_point: float;
    :return: numpy.array;
    """
    if unit == u.mag:
        if zero_point is None:
            raise ValueError('You supplied your data in magnitudes. Please also specify a zero point using keyword '
                             'argument `reference_magnitude`.')
        else:
            data = utils.magnitude_to_flux(data, zero_point)

    return data


def convert_flux_error(error, unit, zero_point=None):
    """
    If data an its errors are in magnitudes, they are converted to normalized flux.

    :param data: numpy.array;
    :param unit: astropy.unit.Unit;
    :param zero_point: float;
    :return: numpy.array;
    """
    if unit == u.mag:
        if zero_point is None:
            raise ValueError('You supplied your data in magnitudes. Please also specify a zero point using keyword '
                             'argument `reference_magnitude`.')
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
    return unit if unit == u.dimensionless_unscaled else to_unit


def read_data_file(filename, data_columns):
    """
    Function loads observation datafile. Rows with column names and comments should start with `#`.
    It deals with missing data by omitting given line

    :param filename: str;
    :param data_columns: tuple;
    :return: numpy.array;
    """
    data = [[] for _ in data_columns]
    with open(filename) as f:
        for line in f:
            if line.startswith('#'):
                continue

            items = [xx.strip() for xx in line.split()]
            try:
                data_to_append = [items[ii] for ii in data_columns]
            except IndexError:
                continue

            for ii in range(len(data_columns)):
                data[ii].append(float(data_to_append[ii]))

    return np.array(data).T


class DataSet(metaclass=ABCMeta):

    ID = 1

    def __init__(self, name=None, **kwargs):
        # initial kwargs
        self.kwargs = copy(kwargs)

        if is_empty(name):
            self.name = str(DataSet.ID)
            logger.debug(f"name of class instance {self.__class__.__name__} set to {self.name}")
            DataSet.ID += 1
        else:
            self.name = str(name)

        # initializing parmas to default values
        self.x_data = np.ndarray(0)
        self.y_data = np.ndarray(0)
        self.yerr = None

        self.check_data_validity(**kwargs)

    @abstractmethod
    def transform_input(self, *args, **kwargs):
        pass

    @staticmethod
    def check_data_validity(**kwargs):
        if not np.shape(kwargs['x_data']) == np.shape(kwargs['y_data']):
            raise ValueError('`x_data` and `y_data` are not of the same shape.')
        if 'yerr' in kwargs.keys():
            if not np.shape(kwargs['x_data']) == np.shape(kwargs['yerr']):
                raise ValueError('`yerr` are not of the same shape to `x_data` and `y_data`.')

        # check for nans
        if np.isnan(kwargs['x_data']).any():
            raise ValueError('`x_data` contains NaN')
        if np.isnan(kwargs['y_data']).any():
            raise ValueError('`y_data` contains NaN')
        if 'yerr' in kwargs.keys():
            if np.isnan(kwargs['yerr']).any():
                raise ValueError('`yerr` contains NaN')


class RVData(DataSet):
    """
    Child class of elisa.analytics.dataset.base.Dataset class storing radial velocity measurement.

    Input parameters:

    :param x_data: numpy.array; time or observed phases
    :param y_data: numpy.array; radial velocities
    :param yerr: numpy.array; radial velocity errors - optional
    :param x_unit: astropy.unit.Unit; if `None` or `astropy.unit.dimensionless_unscaled` is given,
                                      the `x_data are regarded as phases, otherwise if unit is convertible
                                      to days, the `x_data` are regarded to be in JD
    :param y_unit: astropy.unit.Unit; velocity unit of the observed radial velocities and its errors
    """

    MANDATORY_KWARGS = config.DATASET_MANDATORY_KWARGS
    OPTIONAL_KWARGS = config.DATASET_OPTIONAL_KWARGS
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    def __init__(self, name=None, **kwargs):
        utils.invalid_kwarg_checker(kwargs, RVData.ALL_KWARGS, RVData)
        utils.check_missing_kwargs(RVData.MANDATORY_KWARGS, kwargs, instance_of=RVData)
        super(RVData, self).__init__(name, **kwargs)

        kwargs = self.transform_input(**kwargs)

        # conversion to base units
        kwargs = self.convert_arrays(**kwargs)
        self.check_data_validity(**kwargs)

        self.init_parameters(**kwargs)

    def transform_input(self, **kwargs):
        """
        Transform and validate input kwargs.

        :param kwargs: Dict;
        :return: Dict;
        """
        return RVDataProperties.transform_input(**kwargs)

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
        if 'yerr' in kwargs.keys():
            kwargs['yerr'] = convert_data(kwargs['yerr'], kwargs['y_unit'], units.VELOCITY_UNIT)
        kwargs['y_unit'] = convert_unit(kwargs['y_unit'], units.VELOCITY_UNIT)

        return kwargs

    @staticmethod
    def load_from_file(filename, x_unit=None, y_unit=None, data_columns=None):
        """
        Function loads a RV measurements from text file.

        :param filename: str; name of the file
        :param x_unit: astropy.unit.Unit;
        :param y_unit: astropy.unit.Unit;
        :param data_columns: Tuple, ordered tuple with column indices of x_data, y_data, y_errors
        :return: RVData;
        """
        data_columns = (0, 1, 2) if data_columns is None else data_columns

        data = read_data_file(filename, data_columns)
        try:
            errs = data[:, 2]
        except IndexError:
            errs = None
        return RVData(x_data=data[:, 0],
                      y_data=data[:, 1],
                      yerr=errs,
                      x_unit=x_unit,
                      y_unit=y_unit)


class LCData(DataSet):
    """
        Child class of elisa.analytics.dataset.base.Dataset class storing radial velocity measurement.

        Input parameters:

        :param x_data: numpy.array; time or observed phases
        :param y_data: numpy.array; light curves
        :param yerr: numpy.array; light curve errors - optional
        :param x_unit: astropy.unit.Unit; if `None` or `astropy.unit.dimensionless_unscaled` is given,
                                          the `x_data are regarded as phases, otherwise if unit is convertible
                                          to days, the `x_data` are regarded to be in JD
        :param y_unit: astropy.unit.Unit; velocity unit of the observed flux and its errors
    """
    MANDATORY_KWARGS = config.DATASET_MANDATORY_KWARGS
    OPTIONAL_KWARGS = config.DATASET_OPTIONAL_KWARGS + ['reference_magnitude']
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    def __init__(self, name=None, **kwargs):
        utils.invalid_kwarg_checker(kwargs, LCData.ALL_KWARGS, LCData)
        utils.check_missing_kwargs(LCData.MANDATORY_KWARGS, kwargs, instance_of=LCData)
        super(LCData, self).__init__(name, **kwargs)
        kwargs = self.transform_input(**kwargs)

        self.zero_magnitude = None

        # conversion to base units
        kwargs = self.convert_arrays(**kwargs)
        self.check_data_validity(**kwargs)

        self.init_parameters(**kwargs)

    def transform_input(self, **kwargs):
        """
        Transform and validate input kwargs.

        :param kwargs: Dict;
        :return: Dict;
        """
        return LCDataProperties.transform_input(**kwargs)

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

        kwargs['reference_magnitude'] = None if 'reference_magnitude' not in kwargs.keys() else \
            kwargs['reference_magnitude']

        # convert errors
        if 'yerr' in kwargs.keys():
            kwargs['yerr'] = convert_flux_error(kwargs['yerr'], kwargs['y_unit'],
                                                zero_point=kwargs['reference_magnitude'])

        # converting y-axis
        kwargs['y_data'] = convert_flux(kwargs['y_data'], kwargs['y_unit'], zero_point=kwargs['reference_magnitude'])

        kwargs['y_unit'] = convert_unit(kwargs['y_unit'], units.dimensionless_unscaled)

        return kwargs

    @staticmethod
    def load_from_file(filename, x_unit=None, y_unit=None, data_columns=None, reference_magnitude=None):
        """
        Function loads a RV measurements from text file.

        :param filename: str;
        :param x_unit: astropy.unit.Unit;
        :param y_unit: astropy.unit.Unit;
        :param data_columns: Tuple, ordered tuple with column indices of x_data, y_data, y_errors
        :param reference_magnitude: float; zero point for magnitude conversion
        :return: LCData;
        """
        data_columns = (0, 1, 2) if data_columns is None else data_columns

        data = read_data_file(filename, data_columns)
        try:
            errs = data[:, 2]
        except IndexError:
            errs = None
        return LCData(x_data=data[:, 0],
                      y_data=data[:, 1],
                      yerr=errs,
                      x_unit=x_unit,
                      y_unit=y_unit,
                      reference_magnitude=reference_magnitude)
