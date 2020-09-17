import numpy as np

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
    Abstract class that defines bodies that can be modelled by this software.
    Units are imported from astropy.units module::

        see documentation http://docs.astropy.org/en/stable/units/

    It implements following input arguments (body properties) which can be set on input of child instance.

    :param name: str; arbitrary name of instance
    :param synchronicity: float; Object synchronicity (F = omega_rot/omega_orb) setter.
                                 Expects number input convertible to numpy float64 / float.
    :param mass: float; If mass is int, np.int, float, np.float, program assumes solar mass as it's unit.
                        If mass astropy.unit.quantity.Quantity instance, program converts it to default units.
    :param albeo: float; Bolometric albedo (reradiated energy/ irradiance energy).
                         Accepts value of albedo in range (0, 1).
    :param discretization_factor: float;
    :param t_eff: float; Accepts value in Any temperature unit. If your input is without unit,
                         function assumes that supplied value is in Kelvins.
    :param polar_radius: Expected type is astropy.units.quantity.Quantity, numpy.float or numpy.int othervise
                         TypeError will be raised. If quantity is not specified, default distance unit is assumed.
    :param spots: List[Dict[str, float]]; Spots definitions. Example of defined spots

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
    :param equatorial_radius: float
    """

    ID = 1

    def __init__(self, name, **kwargs):
        """
        Properties of abstract class Body.
        """
        # initial kwargs
        self.kwargs = copy(kwargs)

        if is_empty(name):
            self.name = str(Body.ID)
            logger.debug(f"name of class instance {self.__class__.__name__} set to {self.name}")
            Body.ID += 1
        else:
            self.name = str(name)

        # initializing parmas to default values
        self.synchronicity = np.nan
        self.mass = np.nan
        self.albedo = np.nan
        self.discretization_factor = up.radians(3)
        self.t_eff = np.nan
        self.polar_radius = np.nan
        self._spots = dict()
        self.equatorial_radius = np.nan

    @abstractmethod
    def transform_input(self, *args, **kwargs):
        pass

    @property
    def spots(self):
        """
        :return: Dict[int, Spot]
        """
        return self._spots

    @spots.setter
    def spots(self, spots):
        # todo: update example
        """
        example of defined spots

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

        :return: bool
        """
        return len(self._spots) > 0

    def remove_spot(self, spot_index: int):
        """
        Remove n-th spot index of object.

        :param spot_index: int
        """
        del (self._spots[spot_index])

    def setup_spot_instance_discretization_factor(self, spot_instance, spot_index):
        """
        Setup discretization factor for given spot instance based on defined rules::

            - used Star discretization factor if not specified in spot
            - if spot_instance.discretization_factor > 0.5 * spot_instance.angular_diameter then factor is set to
              0.5 * spot_instance.angular_diameter

        :param spot_instance: Spot
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
