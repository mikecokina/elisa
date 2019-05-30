import numpy as np

from typing import Tuple
from abc import ABCMeta, abstractmethod
from astropy import units as u
from scipy.optimize import fsolve

from elisa.engine import logger, units
from elisa.engine import const as c
from elisa.engine.utils import is_empty


class System(metaclass=ABCMeta):
    """
    Abstract class defining System
    see https://docs.python.org/3.5/library/abc.html for more informations
    """

    ID = 1
    KWARGS = []
    OPTIONAL_KWARGS = []
    ALL_KWARGS = KWARGS + OPTIONAL_KWARGS

    def __init__(self, name: str = None, suppress_logger: bool = False, **kwargs) -> None:

        self._logger = logger.getLogger(self.__class__.__name__, suppress=suppress_logger)
        self.initial_kwargs = kwargs.copy()

        # default params
        self._gamma: float = np.nan
        self._period: float = np.nan
        self._inclination: float = np.nan

        if is_empty(name):
            self._name = str(System.ID)
            self._logger.debug(f"name of class instance {self.__class__.__name__} set to {self._name}")
            System.ID += 1
        else:
            self._name = str(name)

        self._inlination = None

    @property
    def name(self) -> str:
        """
        name of object initialized on base of this abstract class

        :return: str
        """
        return self._name

    @name.setter
    def name(self, name: any):
        """
        setter for name of system

        :param name: str
        :return:
        """
        self._name = str(name)

    @property
    def gamma(self) -> float:
        """
        system center of mass radial velocity in default velocity unit

        :return: numpy.float
        """
        return self._gamma

    @gamma.setter
    def gamma(self, gamma: any):
        """
        system center of mass velocity
        expected type is astropy.units.quantity.Quantity, numpy.float or numpy.int othervise TypeError will be raised
        if unit is not specified, default velocity unit is assumed

        :param gamma: numpy.float/numpy.int/astropy.units.quantity.Quantity
        :return: None
        """
        if isinstance(gamma, u.quantity.Quantity):
            self._gamma = np.float64(gamma.to(units.VELOCITY_UNIT))
        elif isinstance(gamma, (int, np.int, float, np.float)):
            self._gamma = np.float64(gamma)
        else:
            raise TypeError('Value of variable `gamma` is not (np.)int or (np.)float '
                            'nor astropy.unit.quantity.Quantity instance.')

    @property
    def period(self) -> float:
        """
        returns orbital period of binary system

        :return: (np.)int, (np.)float, astropy.unit.quantity.Quantity
        """
        return self._period

    @period.setter
    def period(self, period: any):
        """
        set orbital period of binary star system, if unit is not specified, default period unit is assumed

        :param period: (np.)int, (np.)float, astropy.unit.quantity.Quantity
        :return:
        """
        if isinstance(period, u.quantity.Quantity):
            self._period = np.float64(period.to(units.PERIOD_UNIT))
        elif isinstance(period, (int, np.int, float, np.float)):
            self._period = np.float64(period)
        else:
            raise TypeError('Input of variable `period` is not (np.)int or (np.)float '
                            'nor astropy.unit.quantity.Quantity instance.')
        self._logger.debug(f"Setting property period "
                           f"of class instance {self.__class__.__name__} to {self._period}")

    @property
    def inclination(self) -> float:
        """
        inclination of system, angle between z axis and line of sight

        :return: (np.)int, (np.)float, astropy.unit.quantity.Quantity
        """
        return self._inclination

    @inclination.setter
    def inclination(self, inclination: any):
        """
        set orbit inclination of system, if unit is not specified, default unit is assumed

        :param inclination: (np.)int, (np.)float, astropy.unit.quantity.Quantity
        :return:
        """

        if isinstance(inclination, u.quantity.Quantity):
            self._inclination = np.float64(inclination.to(units.ARC_UNIT))
        elif isinstance(inclination, (int, np.int, float, np.float)):
            self._inclination = np.float64(inclination)
        else:
            raise TypeError('Input of variable `inclination` is not (np.)int or (np.)float '
                            'nor astropy.unit.quantity.Quantity instance.')

        if not 0 <= self.inclination <= c.PI:
            raise ValueError(f'Inclination value of {self.inclination} is out of bounds (0, pi).')

        self._logger.debug(f"setting property inclination "
                           f"of class instance {self.__class__.__name__} to {self._inclination}")

    @abstractmethod
    def compute_lightcurve(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_info(self):
        pass

    @abstractmethod
    def init(self):
        pass

    def _solver(self, fn, condition, *args, **kwargs) -> Tuple:
        """
        will solve fn implicit function taking args by using scipy.optimize.fsolve method and return
        solution if satisfy conditional function

        # final spot containers contain their own points and simplex remaped from zero
        # final object contain only points w/o spots points and simplex (triangulation)
        # je tato poznamka vhodna tu? trosku metie

        :param fn: function
        :param condition: function
        :param args: tuple
        :return: float (np.nan), bool
        """
        # precalculation of auxiliary values
        solution, use = np.nan, False
        scipy_solver_init_value = np.array([1. / 10000.])
        try:
            solution, _, ier, mesg = fsolve(fn, scipy_solver_init_value, full_output=True, args=args, xtol=1e-10)
            if ier == 1 and not np.isnan(solution[0]):
                solution = solution[0]
                use = True if 1e15 > solution > 0 else False
            else:
                self._logger.warning('Solution in implicit solver was not found, cause: {}'.format(mesg))
        except Exception as e:
            self._logger.debug("Attempt to solve function {} finished w/ exception: {}".format(fn.__name__, str(e)))
            use = False

        args_to_use = kwargs.get('original_kwargs', args)
        return (solution, use) if condition(solution, *args_to_use) else (np.nan, False)

    @abstractmethod
    def build_mesh(self, *args, **kwargs):
        """
        abstract method for creating surface points

        :param args:
        :param kwargs:
        :return:
        """
        pass

    @abstractmethod
    def build_faces(self, *args, **kwargs):
        """
        abstract method for building body surface from given set of points in already calculated and stored in
        object.points

        :param args:
        :param kwargs:
        :return:
        """
        pass

    @abstractmethod
    def build_surface(self, *args, **kwargs):
        """
        abstract method which builds surface from ground up including points and faces of surface and spots

        :param args:
        :param kwargs:
        :return:
        """
        pass

    @abstractmethod
    def build_surface_map(self, *args, **kwargs):
        """
        abstract method which calculates surface maps for surface faces of given body (eg. temperature or gravity
        acceleration map)

        :param args:
        :param kwargs:
        :return:
        """
        pass
