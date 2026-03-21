from __future__ import annotations

from abc import ABCMeta, abstractmethod
from copy import copy
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.types import Float


class AbstractOrbit(metaclass=ABCMeta):
    """Define the interface for orbital motion.

    The class stores configuration ``kwargs`` and basic orbital parameters
    that concrete orbit implementations may use. Class-level keyword lists
    are immutable tuples to avoid mutable class attribute warnings.

    :cvar MANDATORY_KWARGS: Tuple of names of mandatory keyword arguments.
    :cvar OPTIONAL_KWARGS: Tuple of names of optional keyword arguments.
    :cvar ALL_KWARGS: Tuple of all keyword argument names.
    """

    MANDATORY_KWARGS: ClassVar[tuple[str, ...]] = ()
    OPTIONAL_KWARGS: ClassVar[tuple[str, ...]] = ()
    ALL_KWARGS: ClassVar[tuple[str, ...]] = MANDATORY_KWARGS + OPTIONAL_KWARGS

    def __init__(self, **kwargs: Any) -> None:
        """Initialize orbit parameters.

        :param kwargs: Arbitrary keyword configuration forwarded to concrete
            orbit implementations and stored on the instance.
        :type kwargs: dict[str, Any]
        """
        self.kwargs: dict[str, Any] = copy(kwargs)

        # Orbital parameters. Use elisa.types.Float where appropriate.
        # We import Float under TYPE_CHECKING; runtime uses floats but annotations
        # remain correct due to postponed evaluation.
        self.period: Float = np.nan  # type: ignore[assignment]
        self.inclination: Float = np.nan  # type: ignore[assignment]
        self.phase_shift: Float = 0.0

    @classmethod
    def true_phase(cls, phase: NDArray | Float, phase_shift: Float) -> NDArray[np.floating] | Float:
        """Return the phase shifted by ``phase_shift``.

        If ``phase`` is array-like an :class:`numpy.ndarray` is returned.
        Otherwise, a scalar :class:`elisa.types.Float` is returned.

        :param phase: Phase or array of phases to shift.
        :type phase: NDArray | elisa.types.Float
        :param phase_shift: Amount to shift the phase by.
        :type phase_shift: elisa.types.Float
        :returns: Shifted phase(s).
        :rtype: numpy.ndarray | elisa.types.Float
        """
        if np.isscalar(phase):
            return phase + phase_shift

        return np.asarray(phase) + phase_shift

    @staticmethod
    def phase(true_phase: NDArray | Float, phase_shift: Float) -> NDArray[np.floating] | Float:
        """Revert a previously applied phase shift.

        :param true_phase: Shifted phase(s).
        :type true_phase: NDArray | elisa.types.Float
        :param phase_shift: Applied phase shift to revert.
        :type phase_shift: elisa.types.Float
        :returns: Original phase(s) before the shift.
        :rtype: numpy.ndarray | elisa.types.Float
        """
        if np.isscalar(true_phase):
            return true_phase - phase_shift

        return np.asarray(true_phase) - phase_shift

    @abstractmethod
    def orbital_motion(self, phase: NDArray | Float) -> NDArray[np.floating] | Float:
        """Compute orbital motion for the provided phase(s).

        Concrete implementations should compute positions, velocities or other
        motion-related quantities for the provided phase array or scalar.

        :param phase: Phase(s) at which to compute the orbital motion.
        :type phase: NDArray | elisa.types.Float
        :returns: Result of the orbital motion computation.
        :rtype: numpy.ndarray | elisa.types.Float
        """
        raise NotImplementedError
