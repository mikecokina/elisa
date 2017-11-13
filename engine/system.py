from abc import ABCMeta, abstractmethod
from astropy import units as u
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s : [%(levelname)s] : %(name)s : %(message)s')


class System(object):
    """
    Abstract class defining System
    see https://docs.python.org/3.5/library/abc.html for more infromations
    """

    __metaclass__ = ABCMeta

    ID = 1
    KWARGS = []

    # Units
    __DISTANCE_UNIT = u.m
    __TIME_UNIT = u.s
    __PERIOD_UNIT = u.d
    __VELOCITY_UNIT = __DISTANCE_UNIT / __TIME_UNIT
    __ARC_UNIT = u.rad

    def __init__(self, name=None, **kwargs):
        self._logger = logging.getLogger(System.__name__)

        # default params
        self._gamma = None

        if name is None:
            self._name = str(System.ID)
            self._logger.debug("Name of class instance {} set to {}".format(System.__name__, self._name))
            System.ID += 1
        else:
            self._name = str(name)

    @property
    def name(self):
        """
        name of object initialized on base of this abstract class

        :return: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """
        setter for name of system

        :param name: str
        :return:
        """
        self._name = str(name)

    @property
    def gamma(self):
        """
        system center of mass radial velocity in default velocity unit

        :return: numpy.float
        """
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        """
        system center of mass velocity
        expected type is astropy.units.quantity.Quantity, numpy.float or numpy.int othervise TypeError will be raised
        if unit is not specified, default velocity unit is assumed

        :param gamma: numpy.float/numpy.int/astropy.units.quantity.Quantity
        :return: None
        """
        if isinstance(gamma, u.quantity.Quantity):
            self._gamma = np.float64(gamma.to(self.__VELOCITY_UNIT))
        elif isinstance(gamma, (int, np.int, float, np.float)):
            self._gamma = np.float64(gamma)
        else:
            raise TypeError('Value of variable `gamma` is not (np.)int or (np.)float '
                            'nor astropy.unit.quantity.Quantity instance.')

    # getters for protected variables
    @classmethod
    def get_distance_unit(cls):
        """
        returns default distance unit in astropy.units.quantity.Quantity format

        :return: astropy.units.quantity.Quantity
        """
        return cls.__DISTANCE_UNIT

    @classmethod
    def get_time_unit(cls):
        """
        returns default time unit in astropy.units.quantity.Quantity format

        :return: astropy.units.quantity.Quantity
        """
        return cls.__TIME_UNIT

    @classmethod
    def get_arc_unit(cls):
        """
        returns default arc unit in astropy.units.quantity.Quantity format

        :return: astropy.units.quantity.Quantity
        """
        return cls.__ARC_UNIT

    @classmethod
    def get_velocity_unit(cls):
        """
        returns default velocity unit in astropy.units.quantity.Quantity format

        :return: astropy.units.quantity.Quantity
        """
        return cls.__VELOCITY_UNIT

    @classmethod
    def get_period_unit(cls):
        """
        returns default period unit in astropy.units.quantity.Quantity format

        :return: astropy.units.quantity.Quantity
        """
        return cls.__PERIOD_UNIT

    @abstractmethod
    def compute_lc(self):
        pass

    def get_info(self):
        pass
