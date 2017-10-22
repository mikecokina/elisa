from abc import ABCMeta, abstractmethod
from astropy import units as u
import numpy as np


class System(object, metaclass=ABCMeta):
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
    __VELOCITY_UNIT = __DISTANCE_UNIT / __TIME_UNIT
    __ARCH_UNIT = u.rad

    def __init__(self, name=None, **kwargs):
        if name is None:
            self._name = str(System.ID)
            System.ID += 1
        else:
            self._name = str(name)

        # values of properties
        for kwarg in self.KWARGS:
            setattr(self, kwarg, kwargs[kwarg])

    @property
    def name(self):
        """
        name of object initialized on base of this abstract class
        :return: str
        """
        return self._name

    @name.setter
    def name(self, name):
        self._name = str(name)

    @property
    def gamma(self):
        """
        system center of mass radial velocity
        :return:
        """
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        """
        system center of mass velocity
        expected type is astropy.units.quantity.Quantity, numpy.float or numpy.int othervise TypeError will be raised

        :param gamma: numpy.float/numpy.int/astropy.units.quantity.Quantity
        :return: None
        """
        if isinstance(gamma, u.quantity.Quantity):
            self._gamma = np.float64(gamma.to(self.__VELOCITY_UNIT))
        elif isinstance(gamma, (int, np.int, float, np.float)):
            self._gamma = np.float64(gamma * self.__VELOCITY_UNIT)
        else:
            raise TypeError('Value of variable `gamma` is not (np.)int or (np.)float '
                            'nor astropy.unit.quantity.Quantity instance.')

    # getters for protected variables
    @classmethod
    def get_distance_unit(cls):
        return cls.__DISTANCE_UNIT

    @classmethod
    def get_time_unit(cls):
        return cls.__TIME_UNIT

    @classmethod
    def get_arch_unit(cls):
        return cls.__ARCH_UNIT

    @classmethod
    def get_velocity_unit(cls):
        return cls.__VELOCITY_UNIT

    @abstractmethod
    def compute_lc(self):
        pass

    @abstractmethod
    def get_info(self):
        pass

