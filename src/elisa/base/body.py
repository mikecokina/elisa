from __future__ import annotations

from abc import ABCMeta, abstractmethod
from copy import copy
from typing import TYPE_CHECKING, Any

import numpy as np

from elisa import settings
from elisa import umpy as up
from elisa.base.spot import Spot
from elisa.logger import getLogger
from elisa.utils import is_empty

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = getLogger("base.body")


class Body(metaclass=ABCMeta):
    """Abstract base class for physical bodies modelled by ELISa.

    The :class:`Body` class defines the minimal interface and shared
    attributes used by concrete bodies (for example :class:`Star`). It
    stores common parameters such as mass, temperature and spot
    definitions and provides helpers to manage and normalise spot
    instances.
    """

    ID = 1
    MANDATORY_KWARGS = ()
    OPTIONAL_KWARGS = ()
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    def __init__(self, name: str | None, **kwargs: Any) -> None:
        """Initialise common body properties.

        :param name: Optional instance name; when ``None`` a numeric id is
            assigned automatically.
        :param kwargs: Arbitrary keyword arguments saved in ``self.kwargs``.
        :type kwargs: dict
        :returns: None
        :rtype: None
        """
        # initial kwargs
        self.kwargs: dict[str, Any] = copy(kwargs)

        if is_empty(name):
            self.name = str(Body.ID)
            logger.debug("name of class instance %s set to %s", self.__class__.__name__, self.name)
            Body.ID += 1
        else:
            self.name = str(name)

        # initializing parameters to default values
        self.synchronicity: float = np.nan
        self.mass: float = np.nan
        self.albedo: float = np.nan
        self.discretization_factor: float = float(up.radians(settings.DEFAULT_DISCRETIZATION_FACTOR))
        self.t_eff: float = np.nan
        self.polar_radius: float = np.nan
        self._spots: dict[int, Spot] = {}
        self.equatorial_radius: float = np.nan
        self.atmosphere: str = ""
        self.limb_darkening_coefficients: dict[str, dict[str, Any]] | None = None

    @abstractmethod
    def init(self) -> None:
        """Perform any post-construction initialisation required by the body."""

    @abstractmethod
    def transform_input(self, *args, **kwargs) -> dict:
        """Transform and validate input keyword arguments.

        Implementations should return a mapping with transformed values.
        """

    @property
    def spots(self) -> dict[int, Spot]:
        """Return the spot collection attached to this body.

        :returns: Mapping spot_index → :class:`Spot` instances.
        :rtype: dict[int, elisa.base.spot.Spot]
        """
        return self._spots

    @spots.setter
    def spots(self, spots: Iterable[dict]) -> None:
        """Set the spots collection from an iterable of spot definitions.

        The order of definitions determines layering: the first spot in
        the iterable will be drawn below subsequently defined overlapping
        spots.

            [
                 {"longitude": 90,
                  "latitude": 58,
                  "angular_radius": 15,
                  "temperature_factor": 0.9},
                 {"longitude": 85,
                  "latitude": 80,
                  "angular_radius": 30,
                  "temperature_factor": 1.05},
                 {"longitude": 45,
                  "latitude": 90,
                  "angular_radius": 30,
                  "temperature_factor": 0.95},
             ]

        :param spots: Iterable[Dict]; definition of spots for given object
        """
        self._spots = {idx: Spot(**spot_meta) for idx, spot_meta in enumerate(spots)} if not is_empty(spots) else {}
        for spot_idx, spot_instance in self.spots.items():
            self.setup_spot_instance_discretization_factor(spot_instance, spot_idx)

    def has_spots(self) -> bool:
        """Return True when at least one spot is defined for this body."""
        return len(self._spots) > 0

    @abstractmethod
    def has_pulsations(self) -> bool:
        ...

    def remove_spot(self, spot_index: int) -> None:
        """Remove the spot with the given index.

        :param spot_index: Index of the spot to remove.
        :type spot_index: int
        :returns: None
        :rtype: None
        """
        del self._spots[spot_index]

    def setup_spot_instance_discretization_factor(self, spot_instance: Spot, spot_index: int) -> Spot:
        """Apply discretization-factor rules to a spot instance.

        Rules applied:

        - If the spot does not define its own discretization factor, the
          parent's (body) discretization factor is used (scaled by 0.9).
        - If the spot's discretization factor exceeds its angular radius,
          it is clamped to the angular radius.

        :param spot_instance: Spot instance to adjust.
        :type spot_instance: elisa.base.spot.Spot.
        :param spot_index: Spot index (used for logging only).
        :type spot_index: int
        :returns: The adjusted :class:`Spot` instance.
        :rtype: elisa.base.spot.Spot
        """
        if is_empty(spot_instance.discretization_factor):
            logger.debug(
                "angular density of the spot %s on %s component was not supplied "
                "and discretization factor of body %s was used.",
                spot_index,
                self.name,
                self.discretization_factor,
            )
            spot_instance.discretization_factor = 0.9 * self.discretization_factor
        if spot_instance.discretization_factor > spot_instance.angular_radius:
            logger.debug(
                "angular density %s of the spot %s on %s component was larger than "
                "its angular radius; setting to angular radius",
                self.discretization_factor,
                spot_index,
                self.name,
            )
            spot_instance.discretization_factor = spot_instance.angular_radius

        return spot_instance
