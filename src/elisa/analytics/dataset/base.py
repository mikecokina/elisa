import numpy as np
import astropy.units as u

from elisa.analytics.dataset.transform import RVdataProperties
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
    converts data to desired format or leaves it dimensionless

    :param data: np.ndarray;
    :param unit: astropy.unit;
    :param to_unit: astropy.unit;
    :return: np.ndarray;
    """
    return data if unit == u.dimensionless_unscaled else (data * unit).to(to_unit).value


def convert_unit(unit, to_unit):
    """
    converts to desired unit  or leaves it dimensionless

    :param unit: astropy.unit;
    :param to_unit: astropy.unit;
    :return: astropy.unit;
    """
    return unit if unit == u.dimensionless_unscaled else to_unit


class Dataset(metaclass=ABCMeta):

    ID = 1

    def __init__(self, name=None, **kwargs):
        # initial kwargs
        self.kwargs = copy(kwargs)

        if is_empty(name):
            self.name = str(Dataset.ID)
            logger.debug(f"name of class instance {self.__class__.__name__} set to {self.name}")
            Dataset.ID += 1
        else:
            self.name = str(name)

        # initializing parmas to default values
        self.x_data = np.ndarray(0)
        self.y_data = np.ndarray(0)
        self.yerr = np.ndarray(0)
        self.x_unit = None
        self.y_unit = None

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


class RVdata(Dataset):
    """
    Child class of elisa.analytics.dataset.base.Dataset class storing radial velocity measurement.

    Input parameters:

    :param x_data: numpy.ndarray; time or observed phases
    :param y_data: numpy.ndarray; radial velocities
    :param yerr: numpy.ndarray; radial velocity errors - optional
    :param x_unit: astropy.unit.Unit; if `None` or `astropy.unit.dimensionless_unscaled` is given, the `x_data are regarded
    as phases, otherwise if unit is convertible to days, the `x_data` are regarded to be in JD
    :param y_unit: astropy.unit.Unit; velocity unit of the observed radial velocities and its errors
    """

    MANDATORY_KWARGS = config.DATASET_MANDATORY_KWARGS
    OPTIONAL_KWARGS = config.DATASET_OPTIONAL_KWARGS
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    def __init__(self, name=None, **kwargs):
        utils.invalid_kwarg_checker(kwargs, RVdata.ALL_KWARGS, RVdata)
        super(RVdata, self).__init__(name, **kwargs)
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
        return RVdataProperties.transform_input(**kwargs)

    def init_parameters(self, **kwargs):
        logger.debug(f"initialising properties of class instance {self.__class__.__name__}")
        for kwarg in RVdata.ALL_KWARGS:
            if kwarg in kwargs:
                setattr(self, kwarg, kwargs[kwarg])

    @staticmethod
    def convert_arrays(**kwargs):
        """
        converting data and units to its base counterparts or keeping them dimensionless
        :param kwargs:
        :return:
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
