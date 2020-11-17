import numpy as np

from . transform import OrbitProperties
from ... import (
    const as c,
    units as u,
    utils
)
from ... logger import getLogger

logger = getLogger('single_system.orbit.orbit')


def angular_velocity(rotation_period):
    """
    Rotational angular velocity of the star.

    :param rotation_period: float;
    :return: float;
    """
    return c.FULL_ARC / (rotation_period * u.PERIOD_UNIT).to(u.s).value


def true_phase_to_azimuth(phase):
    """
    Calculates observer azimuths for single star system.

    :param phase: Union[numpy.array, float];
    :return: Union[numpy.array, float];
    """
    return c.FULL_ARC * phase


def azimuth_to_true_phase(azimuth):
    """
    :param azimuth: Union[numpy.array, float];
    :return: Union[numpy.array, float];
    """
    return azimuth / c.FULL_ARC


class Orbit(object):
    """
    Model which represents rotational motion of the single system as apparent orbital motion of the observer.

    :param rotational_period: Union[(numpy.)float, (numpy.)int, astropy.units.quantity.Quantity];
    :param inclination: Union[float, astropy.units.Quantity]; If unitless values is supplied, default unit
                          suppose to be radians.
    """
    MANDATORY_KWARGS = ['rotation_period', 'inclination']
    OPTIONAL_KWARGS = ['phase_shift']
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    def __init__(self, **kwargs):
        utils.invalid_kwarg_checker(kwargs, Orbit.ALL_KWARGS, Orbit)
        utils.check_missing_kwargs(self.__class__.MANDATORY_KWARGS, kwargs, instance_of=self.__class__)
        kwargs = OrbitProperties.transform_input(**kwargs)

        # default valeus of properties
        self.rotational_period = np.nan
        self.inclination = np.nan
        self.phase_shift = 0.0

        # values of properties
        logger.debug(f"setting properties of orbit")
        for kwarg in kwargs:
            setattr(self, kwarg, kwargs[kwarg])

    @staticmethod
    def phase(true_phase, phase_shift):
        """
        correction in phase

        :param true_phase: Union[numpy.array, float];
        :param phase_shift: numpy.float;
        :return: Union[numpy.array, float];
        """
        return true_phase - phase_shift

    @staticmethod
    def rotational_motion(phase):
        """
        Function takes photometric phase of the single system as input and calculates azimuths for the observer

        :param phase: Union[numpy.array, float];
        :return: numpy.array; matrix consisting of column stacked vectors azimut angle and phase

        ::

            numpy.array((az1, phs1),
                        (az2, phs2),
                         ...
                        (azN, phsN))
        """
        # ability to accept scalar as input
        if isinstance(phase, (int, np.int, float, np.float)):
            phase = np.array([np.float(phase)])

        azimuth_angle = true_phase_to_azimuth(phase=phase)

        return np.column_stack((azimuth_angle, phase))

    @staticmethod
    def rotational_motion_from_azimuths(azimuth):
        """
        return rotational motion derived from known azimuths

        :param azimuth: Union[numpy.array, float];
        :return: numpy.array; matrix consisting of column stacked vectors distance,
                                azimut angle, true anomaly and phase

        ::

               numpy.array((az1, phs1),
                           (az2, phs2),
                            ...
                           (azN, phsN))

        """
        true_phase = azimuth_to_true_phase(azimuth)
        return np.column_stack((azimuth, true_phase))
