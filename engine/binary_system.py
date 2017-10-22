from engine.system import System
from astropy import units as u
import numpy as np


class BinarySystem(System):

    KWARGS = ['gamma', 'inclination']

    def __init__(self, name=None, **kwargs):
        self.is_property(kwargs)
        super(BinarySystem, self).__init__(name=name, **kwargs)

        # default values of properties
        self._inclination = None

        # values of properties
        for kwarg in BinarySystem.KWARGS:
            setattr(self, kwarg, kwargs[kwarg])

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
            self._inclination = np.float64(inclination.to(self.get_arch_unit()))
        elif isinstance(inclination, (int, np.int, float, np.float)):
            self._inclination = np.float64(inclination * self.get_arch_unit())
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
