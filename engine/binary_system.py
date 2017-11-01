from engine.system import System
from astropy import units as u
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s : [%(levelname)s] : %(name)s : %(message)s')


class BinarySystem(System):

    KWARGS = ['gamma', 'inclination', 'period']

    def __init__(self, name=None, **kwargs):
        self.is_property(kwargs)
        super(BinarySystem, self).__init__(name=name, **kwargs)

        # default values of properties
        self._inclination = None
        self._period = None

        # values of properties
        for kwarg in BinarySystem.KWARGS:
            if kwarg in kwargs:
                self._logger.debug("Setting property {} "
                                   "of class instance {} to {}".format(kwarg, BinarySystem.__name__, kwargs[kwarg]))
                setattr(self, kwarg, kwargs[kwarg])

    @property
    def period(self):
        """
        binarys system period

        :return:
        """
        return self._period

    @period.setter
    def period(self, period):
        if isinstance(period, u.quantity.Quantity):
            self._period = np.float64(period.to(self.get_period_unit()))
        elif isinstance(period, (int, np.int, float, np.float)):
            self._inclination = np.float64(period * self.get_period_unit())
        else:
            raise TypeError('Input of variable `period` is not (np.)int or (np.)float '
                            'nor astropy.unit.quantity.Quantity instance.')

    @property
    def inclination(self):
        """
        inclination of binary star system

        :return:
        """
        return self._inclination

    @inclination.setter
    def inclination(self, inclination):
        if isinstance(inclination, u.quantity.Quantity):
            self._inclination = np.float64(inclination.to(self.get_arc_unit()))
        elif isinstance(inclination, (int, np.int, float, np.float)):
            self._inclination = np.float64(inclination * self.get_arc_unit())
        else:
            raise TypeError('Input of variable `inclination` is not (np.)int or (np.)float '
                            'nor astropy.unit.quantity.Quantity instance.')

    def compute_lc(self):
        pass

    def get_info(self):
        pass

    @classmethod
    def is_property(cls, kwargs):
        is_not = ['`{}`'.format(k) for k in kwargs if k not in cls.KWARGS]
        if is_not:
            raise AttributeError('Arguments {} are not valid {} properties.'.format(', '.join(is_not), cls.__name__))
