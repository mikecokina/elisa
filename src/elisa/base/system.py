import numpy as np

from typing import Dict, Union
from abc import ABCMeta, abstractmethod
from .. base.body import Body
from .. logger import getLogger
from .. import utils
from .. pulse import pulsations
from elisa import units as u
from elisa.base.surface.temperature import interpolate_bolometric_gravity_darkening

logger = getLogger('base.system')


class System(metaclass=ABCMeta):
    """
    Abstract class defining System.
    Following arguments are implemented as common any of child instances.
    """

    ID = 1
    MANDATORY_KWARGS = []
    OPTIONAL_KWARGS = []
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    def __init__(self, name=None, **kwargs):
        # default params
        self.inclination: float = np.nan
        self.period: float = np.nan
        self.t0: float = np.nan
        self.gamma: float = np.nan
        self.additional_light: float = 0.0

        self._components: Union[None, Dict] = None

        if utils.is_empty(name):
            self.name = str(System.ID)
            logger.debug(f"name of class instance {self.__class__.__name__} autoset to {self.name}")
            self.__class__.ID += 1
        else:
            self.name = str(name)

    @property
    @abstractmethod
    def components(self):
        pass

    @abstractmethod
    def compute_lightcurve(self, *args, **kwargs):
        pass

    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def transform_input(self, *args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def from_json(cls, data, _verify, _kind_of):
        pass

    def assign_pulsations_amplitudes(self, normalisation_constant=1.0):
        """
        Function assigns amplitudes of displacement to each mode based on radial velocity amplitude.

        :param normalisation_constant: float;
        """
        for component, component_instance in self._components.items():
            if component_instance.has_pulsations():
                pulsations.assign_amplitudes(component_instance, normalisation_constant)

    def init_properties(self, **kwargs):
        """
        Setup system properties from input.

        :param kwargs: Dict; all supplied input properties
        """
        logger.debug(f"initialising properties of system {self.name}, values: {kwargs}")
        for kwarg in kwargs:
            setattr(self, kwarg, kwargs[kwarg])

    def has_pulsations(self):
        """
        Resolve whether any of components has pulsation.

        :return: bool;
        """
        return any([instance.has_pulsations() for instance in self.components.values()])

    def has_spots(self):
        """
        Resolve whether any of components has spots.

        :return: bool;
        """
        return any([instance.has_spots() for instance in self.components.values()])

    @staticmethod
    def object_params_validity_check(components, mandatory_kwargs):
        """
        Checking if star instances have all additional attributes set properly.

        :param components: str;
        :param mandatory_kwargs: List;
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

    def kwargs_serializer(self):
        """
        Creating dictionary of keyword arguments of System class in order to be able to reinitialize the class
        instance in init().

        :return: Dict;
        """
        serialized_kwargs = dict()
        for kwarg in self.ALL_KWARGS:
            if kwarg in ['argument_of_periastron', 'inclination']:
                value = getattr(self, kwarg)
                if not isinstance(value, u.Quantity):
                    value = value * u.ARC_UNIT
                serialized_kwargs[kwarg] = value
            else:
                serialized_kwargs[kwarg] = getattr(self, kwarg)
        return serialized_kwargs

    def setup_betas(self):
        """
        Setup of default gravity darkening components.
        """
        for component, instance in self.components.items():
            if utils.is_empty(instance.gravity_darkening):
                instance.gravity_darkening = interpolate_bolometric_gravity_darkening(instance.t_eff)

    setup_gravity_darkening = setup_betas
