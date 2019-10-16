import numpy as np

from abc import ABCMeta, abstractmethod
from elisa import logger, utils
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

    def init_properties(self, **kwargs):
        """
        Setup system properties from input.
        :param kwargs: Dict; all supplied input properties
        :return:
        """
        self._logger.debug(f"initialising properties of system {self.name}, values: {kwargs}")
        for kwarg in kwargs:
            setattr(self, kwarg, kwargs[kwarg])

    @staticmethod
    def object_params_validity_check(components, mandatory_kwargs):
        """
        Checking if star instances have all additional atributes set properly.

        :param components: str
        :param mandatory_kwargs: List
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

    @property
    @abstractmethod
    def components(self):
        pass

    def has_pulsations(self):
        """
        Resolve whether any of components has pulsation
        :return: bool
        """
        retval = False
        for component_instance in self.components.values():
            retval = retval | component_instance.has_pulsations()
        return retval

    def has_spots(self):
        """
        Resolve whether any of components has spots

        :return: bool
        """
        retval = False
        for component_instance in self.components.values():
            retval = retval | component_instance.has_spots()
        return retval
