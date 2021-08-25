import numpy as np

from typing import Dict
from copy import copy
from abc import (
    ABCMeta,
    abstractmethod
)

from .. utils import is_empty
from .. base.spot import Spot
from .. logger import getLogger
from .. import (
    units as u,
    umpy as up
)

logger = getLogger('base.body')


class Body(metaclass=ABCMeta):
    """
    Abstract class that defines bodies modelled by this package.
    """

    ID = 1
    MANDATORY_KWARGS = []
    OPTIONAL_KWARGS = []
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    def __init__(self, name: str, **kwargs):
        """
        Properties of abstract class Body.
        """
        # initial kwargs
        self.kwargs: Dict = copy(kwargs)

        if is_empty(name):
            self.name = str(Body.ID)
            logger.debug(f"name of class instance {self.__class__.__name__} set to {self.name}")
            Body.ID += 1
        else:
            self.name = str(name)

        # initializing paramas to default values
        self.synchronicity: float = np.nan
        self.mass: float = np.nan
        self.albedo: float = np.nan
        self.discretization_factor: float = np.float64(up.radians(5))
        self.t_eff: float = np.nan
        self.polar_radius: float = np.nan
        self._spots: Dict = dict()
        self.equatorial_radius: float = np.nan
        self.atmosphere: str = ""

    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def transform_input(self, *args, **kwargs):
        pass

    @property
    def spots(self):
        """
        :return: Dict[int, elisa.base.spot.Spot]
        """
        return self._spots

    @spots.setter
    def spots(self, spots):
        """
        Order in which the spots are defined will determine the layering of the spots (spot defined as first will lay
        bellow any subsequently defined overlapping spot). Example of defined spots

        ::

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
        self._spots = {idx: Spot(**spot_meta) for idx, spot_meta in enumerate(spots)} if not is_empty(spots) else dict()
        for spot_idx, spot_instance in self.spots.items():
            self.setup_spot_instance_discretization_factor(spot_instance, spot_idx)

    def has_spots(self):
        """
        Find whether object has defined spots.

        :return: bool;
        """
        return len(self._spots) > 0

    def remove_spot(self, spot_index: int):
        """
        Remove n-th spot index of object.

        :param spot_index: int;
        """
        del self._spots[spot_index]

    def setup_spot_instance_discretization_factor(self, spot_instance, spot_index):
        """
        Setup discretization factor for given spot instance based on defined rules.

        - use value of the parent star if the spot discretization factor is not defined
        - if spot_instance.discretization_factor > 0.5 * spot_instance.angular_diameter then factor is set to
                      0.5 * spot_instance.angular_diameter

        :param spot_instance: elisa.base.spot.Spot;
        :param spot_index: int; spot index (has no affect on process, used for logging)
        :return: elisa.base.spot.Spot;
        """
        if is_empty(spot_instance.discretization_factor):
            logger.debug(f'angular density of the spot {spot_index} on {self.name} component was not supplied '
                         f'and discretization factor of star {self.discretization_factor} was used.')
            spot_instance.discretization_factor = (0.9 * self.discretization_factor * u.ARC_UNIT).value
        if spot_instance.discretization_factor > spot_instance.angular_radius:
            logger.debug(f'angular density {self.discretization_factor} of the spot {spot_index} on {self.name} '
                         f'component was larger than its angular radius. Therefore value of angular density was '
                         f'set to be equal to 0.5 * angular diameter')
            spot_instance.discretization_factor = spot_instance.angular_radius

        return spot_instance
