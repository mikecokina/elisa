import numpy as np

from abc import ABCMeta, abstractmethod
from astropy import units as u
from scipy.optimize import fsolve

from elisa import logger, const as c, units
from elisa import utils
from elisa.utils import is_empty

from elisa.base.body import Body


class System(metaclass=ABCMeta):
    """
    Abstract class defining System
    """

    ID = 1
    MANDATORY_KWARGS = []
    OPTIONAL_KWARGS = []
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    def __init__(self, name=None, logger_name=None, suppress_logger=False, **kwargs):

        self._suppress_logger = suppress_logger
        self._logger = logger.getLogger(logger_name or self.__class__.__name__, suppress=self._suppress_logger)

        # default params
        self._gamma = np.nan
        self._period = np.nan
        self._inclination = np.nan
        self._additional_light = 0.0

        if is_empty(name):
            self._name = str(System.ID)
            self._logger.debug(f"name of class instance {self.__class__.__name__} set to {self._name}")
            System.ID += 1
        else:
            self._name = str(name)

    @property
    def name(self):
        """
        Name of object initialized on base of this abstract class.

        :return: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """
        Setter for name of system.

        :param name: str
        :return:
        """
        self._name = str(name)

    @property
    def gamma(self):
        """
        System center of mass radial velocity in default velocity unit.

        :return: float
        """
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        """
        Set system center of mass velocity.
        Expected type is astropy.units.quantity.Quantity, numpy.float or numpy.int othervise TypeError will be raised.
        If unit is not specified, default velocity unit is assumed.

        :param gamma: (numpy.)float, (numpy.)int, astropy.units.quantity.Quantity
        :return:
        """
        if isinstance(gamma, u.quantity.Quantity):
            self._gamma = np.float64(gamma.to(units.VELOCITY_UNIT))
        elif isinstance(gamma, (int, np.int, float, np.float)):
            self._gamma = np.float64(gamma)
        else:
            raise TypeError('Value of variable `gamma` is not (numpy.)int or (numpy.)float '
                            'nor astropy.unit.quantity.Quantity instance.')

    @property
    def period(self):
        """
        Returns orbital period of binary system.

        :return: float
        """
        return self._period

    @period.setter
    def period(self, period: any):
        """
        Set orbital period of binary star system.
        If unit is not specified, default period unit is assumed.

        :param period: (numpy.)int, (numpy.)float, astropy.unit.quantity.Quantity
        :return:
        """
        if isinstance(period, u.quantity.Quantity):
            self._period = np.float64(period.to(units.PERIOD_UNIT))
        elif isinstance(period, (int, np.int, float, np.float)):
            self._period = np.float64(period)
        else:
            raise TypeError('Input of variable `period` is not (numpy.)int or (numpy.)float '
                            'nor astropy.unit.quantity.Quantity instance.')
        self._logger.debug(f"Setting property period "
                           f"of class instance {self.__class__.__name__} to {self._period}")

    @property
    def inclination(self):
        """
        Returns inclination of system.

        :return: float
        """
        return self._inclination

    @inclination.setter
    def inclination(self, inclination):
        """
        Setter for inclination of the system.
        If unit is not supplied, value in degrees is assumed.

        :param inclination: float
        :return:
        """
        if isinstance(inclination, u.quantity.Quantity):
            self._inclination = np.float64(inclination.to(units.ARC_UNIT))
        elif isinstance(inclination, (int, np.int, float, np.float)):
            self._inclination = np.float64((inclination * u.deg).to(units.ARC_UNIT))
        else:
            raise TypeError('Input of variable `inclination` is not (numpy.)int or (numpy.)float '
                            'nor astropy.unit.quantity.Quantity instance.')

        if not 0 <= self.inclination <= c.PI:
            raise ValueError(f'Inclination value of {self.inclination} is out of bounds (0, pi).')

        self._logger.debug(f"setting property inclination of class instance "
                           f"{self.__class__.__name__} to {self._inclination}")

    @property
    def additional_light(self):
        """
        Returns additional light - light that does not originate from any member of system.

        :return: float (0,1)
        """
        return self._additional_light

    @additional_light.setter
    def additional_light(self, add_light):
        """
        setter for additional light - light that does not originate from any member of system.

        :param add_light: float (0, 1)
        :return:
        """
        if not 0.0 <= add_light <= 1.0:
            raise ValueError('Invalid value of additional light. Valid values are between 0 and 1.')
        self._additional_light = add_light

    @abstractmethod
    def compute_lightcurve(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_info(self):
        pass

    @abstractmethod
    def init(self):
        pass

    def _solver(self, fn, condition, *args, **kwargs):
        """
        Will solve `fn` implicit function taking args by using scipy.optimize.fsolve method and return
        solution if satisfy conditional function.

        :param fn: function
        :param condition: function
        :param args: tuple
        :return: float, bool
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
                self._logger.warning(f'solution in implicit solver was not found, cause: {mesg}')
        except Exception as e:
            self._logger.debug(f"attempt to solve function {fn.__name__} finished w/ exception: {str(e)}")
            use = False

        args_to_use = kwargs.get('original_kwargs', args)
        return (solution, use) if condition(solution, *args_to_use) else (np.nan, False)

    @abstractmethod
    def build_mesh(self, *args, **kwargs):
        """
        Abstract method for creating surface points.

        :param args:
        :param kwargs:
        :return:
        """
        pass

    @abstractmethod
    def build_faces(self, *args, **kwargs):
        """
        Abstract method for building body surface (faces) from given set of points in already calculated and stored in
        Object.points.

        :param args:
        :param kwargs:
        :return:
        """
        pass

    @abstractmethod
    def build_surface(self, *args, **kwargs):
        """
        Abstract method which builds surface from ground up including points and faces of surface and spots.

        :param args:
        :param kwargs:
        :return:
        """
        pass

    @abstractmethod
    def build_surface_map(self, *args, **kwargs):
        """
        Abstract method which calculates surface maps for surface faces of given body (eg. temperature or gravity
        acceleration map).

        :param args:
        :param kwargs:
        :return:
        """
        pass

    @abstractmethod
    def build(self, *args, **kwargs):
        pass

    @staticmethod
    def _object_params_validity_check(components, mandatory_kwargs):
        """
        Checking if star instances have all additional atributes set properly.

        :param mandatory_kwargs: list
        :return:
        """
        for component, component_instance in components.items():
            if not isinstance(component_instance, Body):
                raise TypeError(f"Component `{component}` is not instance of class {Body.__name__}")

        # checking if system components have all mandatory parameters initialised
        missing_kwargs = []
        for component, component_instance in components.items():
            for kwarg in mandatory_kwargs:
                if utils.is_empty(getattr(component_instance, kwarg)):
                    missing_kwargs.append(f"`{kwarg}`")

            if len(missing_kwargs) != 0:
                raise ValueError(f'Mising argument(s): {", ".join(missing_kwargs)} '
                                 f'in {component} component Star class')

    def init_properties(self, **kwargs):
        """
        Setup system properties from input
        :param kwargs: Dict; all supplied input properties
        :return:
        """
        self._logger.debug(f"initialising properties of system {self.name}, values: {kwargs}")
        for kwarg in kwargs:
            setattr(self, kwarg, kwargs[kwarg])

    def has_pulsations(self):
        """
        Resolve whether any of components has pulsation

        :return: bool
        """
        retval = False
        for component_instance in self.stars.values():
            retval = retval | component_instance.has_pulsations()
        return retval

    def has_spots(self):
        """
        Resolve whether any of components has spots

        :return: bool
        """
        retval = False
        for component_instance in self.stars.values():
            retval = retval | component_instance.has_spots()
        return retval
