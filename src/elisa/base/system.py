import numpy as np

from abc import ABCMeta, abstractmethod
from scipy.optimize import fsolve

from elisa import logger, utils, umpy as up
from elisa.utils import is_empty
from elisa.base.body import Body


class System(metaclass=ABCMeta):
    """
    Abstract class defining System

    :param inclination: float or astropy.quantity.Quantity; Inclination of binary system.
    :param period: float or astropy.quantity.Quantity; Period of binary system.
    :param gamma: float or astropy.quantity.Quantity; Center of mass velocity of binary system.
    :param additional_light: float;
    """

    ID = 1
    MANDATORY_KWARGS = []
    OPTIONAL_KWARGS = []
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    def __init__(self, name=None, logger_name=None, suppress_logger=False, **kwargs):

        self._suppress_logger = suppress_logger
        self._logger = logger.getLogger(logger_name or self.__class__.__name__, suppress=self._suppress_logger)

        # default params
        self.inclination = np.nan
        self.period = np.nan
        self.gamma = np.nan
        self.additional_light = 0.0

        if is_empty(name):
            self.name = str(System.ID)
            self._logger.debug(f"name of class instance {self.__class__.__name__} set to {self.name}")
            self.__class__.ID += 1
        else:
            self.name = str(name)

    @abstractmethod
    def compute_lightcurve(self, *args, **kwargs):
        pass

    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def build_mesh(self, *args, **kwargs):
        pass

    @abstractmethod
    def build_faces(self, *args, **kwargs):
        pass

    @abstractmethod
    def build_surface(self, *args, **kwargs):
        pass

    @abstractmethod
    def build_surface_map(self, *args, **kwargs):
        pass

    @abstractmethod
    def build(self, *args, **kwargs):
        pass

    @abstractmethod
    def transform_input(self, *args, **kwargs):
        pass

    def has_pulsations(self):
        return False

    def has_spots(self):
        return False

    # fixme: reimplement, since star is not defined in general (currenlty not defined at all)
    # def has_pulsations(self):
    #     """
    #     Resolve whether any of components has pulsation
    #
    #     :return: bool
    #     """
    #     retval = False
    #     for component_instance in self.stars.values():
    #         retval = retval | component_instance.has_pulsations()
    #     return retval
    #
    # def has_spots(self):
    #     """
    #     Resolve whether any of components has spots
    #
    #     :return: bool
    #     """
    #     retval = False
    #     for component_instance in self.stars.values():
    #         retval = retval | component_instance.has_spots()
    #     return retval

    # fixme: use fsolver
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
            if ier == 1 and not up.isnan(solution[0]):
                solution = solution[0]
                use = True if 1e15 > solution > 0 else False
            else:
                self._logger.warning(f'solution in implicit solver was not found, cause: {mesg}')
        except Exception as e:
            self._logger.debug(f"attempt to solve function {fn.__name__} finished w/ exception: {str(e)}")
            use = False

        args_to_use = kwargs.get('original_kwargs', args)
        return (solution, use) if condition(solution, *args_to_use) else (np.nan, False)

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
