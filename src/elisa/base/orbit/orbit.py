import numpy as np

from abc import ABCMeta, abstractmethod
from copy import copy


class AbstractOrbit(metaclass=ABCMeta):
    """
     Abstract class that defines framework for (orbital) motion of the System instances.
    """
    MANDATORY_KWARGS = []
    OPTIONAL_KWARGS = []
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    def __init__(self, **kwargs):
        self.kwargs = copy(kwargs)

        self.period = np.nan
        self.inclination = np.nan
        self.phase_shift = 0.0

    @classmethod
    def true_phase(cls, phase, phase_shift):
        """
        Returns shifted phase of the orbit by the amount `phase_shift`.

        :param phase: Union[numpy.array, float];
        :param phase_shift: float;
        :return: Union[numpy.array, float];
        """
        return phase + phase_shift

    @staticmethod
    def phase(true_phase, phase_shift):
        """
        reverts the phase shift introduced in function true phase

        :param true_phase: Union[numpy.array, float];
        :param phase_shift: numpy.float;
        :return: Union[numpy.array, float];
        """
        return true_phase - phase_shift

    @abstractmethod
    def orbital_motion(self, phase):
        pass
