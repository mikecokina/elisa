from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from elisa import const as c
from elisa import units as u
from elisa import utils
from elisa.base.orbit.orbit import AbstractOrbit
from elisa.base.types import FLOAT, INT
from elisa.logger import getLogger
from elisa.single_system.orbit.transform import OrbitProperties

logger = getLogger("single_system.orbit.orbit")

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.types import Float


def angular_velocity(rotation_period: Float) -> Float:
    """Return rotational angular velocity of the star.

    The rotational angular velocity is computed from the supplied
    ``rotation_period`` and default single-system unit definition.

    :param rotation_period: Rotation period of the star in project time units.
    :type rotation_period: elisa.types.Float
    :returns: Angular velocity in radians per project time unit.
    :rtype: elisa.types.Float
    """
    seconds = (rotation_period * u.DefaultSingleSystemUnits.system.rotation_period).to(
        u.TIME_UNIT,
    ).value
    return c.FULL_ARC / seconds


def true_phase_to_azimuth(phase: NDArray | Float) -> NDArray[np.floating] | Float:
    """Convert photometric phase(s) to observer azimuth(s).

    The mapping is a simple linear scaling by a full orbital angle.

    :param phase: Photometric phase scalar or array-like.
    :type phase: numpy.typing.NDArray | elisa.types.Float
    :returns: Azimuth(s) in radians or array of radians.
    :rtype: numpy.ndarray | elisa.types.Float
    """
    return c.FULL_ARC * phase


def azimuth_to_true_phase(azimuth: NDArray | Float) -> NDArray[np.floating] | Float:
    """Convert observer azimuth(s) to photometric phase(s).

    :param azimuth: Azimuth scalar or array-like in radians.
    :type azimuth: numpy.typing.NDArray | elisa.types.Float
    :returns: Photometric phase(s).
    :rtype: numpy.ndarray | elisa.types.Float
    """
    return azimuth / c.FULL_ARC


class Orbit(AbstractOrbit):
    """Represent single-system rotational motion as apparent orbital motion.

    The class accepts ``rotation_period`` and ``inclination`` among other
    optional parameters supported by :class:`OrbitProperties`.

    :cvar MANDATORY_KWARGS: Names of mandatory keyword arguments.
    :cvar OPTIONAL_KWARGS: Names of optional keyword arguments.
    :cvar ALL_KWARGS: All accepted keyword names.
    """

    MANDATORY_KWARGS: ClassVar[tuple[str, ...]] = ("rotation_period", "inclination")
    OPTIONAL_KWARGS: ClassVar[tuple[str, ...]] = ("phase_shift",)
    ALL_KWARGS: ClassVar[tuple[str, ...]] = MANDATORY_KWARGS + OPTIONAL_KWARGS

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the single-system orbit model.

        Keyword arguments are validated against :pyattr:`~Orbit.ALL_KWARGS` and
        transformed using :class:`OrbitProperties`.

        :param kwargs: Orbit parameters forwarded to the model.
        :type kwargs: dict
        """
        utils.invalid_kwarg_checker(kwargs, list(Orbit.ALL_KWARGS), Orbit)
        utils.check_missing_kwargs(list(self.__class__.MANDATORY_KWARGS), kwargs, instance_of=self.__class__)
        kwargs = OrbitProperties.transform_input(**kwargs)

        super().__init__(**kwargs)

        # default values of properties
        self.rotational_period: Float = np.nan

        # set supplied properties without using f-strings in logging
        logger.debug("setting properties of orbit instance %s", self.__class__.__name__)
        for kw in kwargs:
            setattr(self, kw, kwargs[kw])

        # keep period consistent with rotational_period
        self.period = self.rotational_period

    def orbital_motion(self, phase: NDArray | Float) -> NDArray[np.floating]:
        """Alias to :meth:`rotational_motion`.

        :param phase: Photometric phase(s).
        :type phase: numpy.typing.NDArray | elisa.types.Float
        :returns: 2-D array with columns (azimuth, placeholder, phase).
        :rtype: numpy.ndarray
        """
        return self.rotational_motion(phase)

    @staticmethod
    def rotational_motion(phase: NDArray | Float) -> NDArray[np.floating]:
        """Compute rotational motion (azimuths) for given photometric phase(s).

        The function accepts a phase scalar or array-like and returns an
        ``(N, 3)`` array with columns: ``(azimuth, nan, phase)``.

        :param phase: Photometric phase(s) as scalar or array-like.
        :type phase: numpy.typing.NDArray | elisa.types.Float
        :returns: 2-D array where each row is (azimuth, nan, phase).
        :rtype: numpy.ndarray
        """
        # allow scalar inputs
        if isinstance(phase, (int, INT, float, FLOAT)):
            phase = np.array([FLOAT(phase)])

        azimuth_angle = true_phase_to_azimuth(phase)

        return np.column_stack((azimuth_angle, np.full(np.shape(phase), np.nan), phase))

    @staticmethod
    def rotational_motion_from_azimuths(azimuth: NDArray | Float) -> NDArray[np.floating]:
        """Compute rotational motion when azimuth(s) are provided.

        Returns an ``(N, 3)`` array with columns ``(azimuth, nan, phase)`` where
        ``phase`` is derived from the provided azimuth(s).

        :param azimuth: Azimuth scalar or array-like in radians.
        :type azimuth: numpy.typing.NDArray | elisa.types.Float
        :returns: 2-D array with rows (azimuth, nan, phase).
        :rtype: numpy.ndarray
        """
        true_phase = azimuth_to_true_phase(azimuth)
        return np.column_stack((azimuth, np.full(np.shape(azimuth), np.nan), true_phase))
