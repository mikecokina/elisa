from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from elisa import const, utils
from elisa import units as u
from elisa.base.body import Body
from elisa.base.surface.temperature import interpolate_bolometric_gravity_darkening
from elisa.logger import getLogger
from elisa.pulse import pulsations

if TYPE_CHECKING:
    from elisa.types import Float


logger = getLogger("base.system")


class System(metaclass=ABCMeta):
    """Abstract base class representing a celestial system.

    Concrete system implementations (binary, single star) extend this class
    and provide component-specific behaviour.
    """

    ID = 1
    MANDATORY_KWARGS: tuple[str, ...] = ()
    OPTIONAL_KWARGS: tuple[str, ...] = ()
    ALL_KWARGS: tuple[str, ...] = MANDATORY_KWARGS + OPTIONAL_KWARGS

    STAR_MANDATORY_KWARGS: tuple[str, ...] = ()
    STAR_OPTIONAL_KWARGS: tuple[str, ...] = ()
    STAR_ALL_KWARGS: tuple[str, ...] = STAR_MANDATORY_KWARGS + STAR_OPTIONAL_KWARGS

    def __init__(self, name: str | None = None, **kwargs: Any) -> None:
        """Initialise the base System.

        :param name: Optional name of the system; a numeric ID is assigned if
            omitted.
        :type name: str | None
        :param kwargs: Additional keyword arguments to set as attributes.
        :type kwargs: Any
        :returns: None
        :rtype: None
        """
        # default params
        self.inclination: Float = np.nan
        self.period: Float = np.nan
        self.t0: Float = np.nan
        self.gamma: Float = np.nan
        self.additional_light: Float = 0.0
        self.distance: Float = 10 * const.PC

        self._components: dict[str, Body] | None = None

        if utils.is_empty(name):
            self.name = str(System.ID)
            logger.debug(
                "name of class instance %s autoset to %s",
                self.__class__.__name__,
                self.name,
            )
            self.__class__.ID += 1
        else:
            self.name = str(name)

        # apply any provided kwargs to the instance
        if kwargs:
            self.init_properties(**kwargs)

    @property
    @abstractmethod
    def components(self) -> dict[str, Body]:
        """Mapping of component name to component instance.

        :returns: Dictionary of component instances.
        :rtype: dict[str, elisa.base.body.Body]
        """
        ...

    @abstractmethod
    def compute_lightcurve(self, *args: Any, **kwargs: Any) -> Any:
        """Compute light curve for the system.

        Subclasses must implement the integration over the surface and
        return a mapping of passband/component to flux arrays.

        :param args: Positional arguments forwarded to concrete implementation.
        :type args: Any
        :param kwargs: Keyword arguments forwarded to concrete implementation.
        :type kwargs: Any
        :returns: Implementation-defined return value (typically phases and
            fluxes mapping).
        :rtype: Any
        """
        ...

    @abstractmethod
    def init(self) -> None:
        """Perform any expensive initialisation required before computations.

        :returns: None
        :rtype: None
        """
        ...

    @abstractmethod
    def transform_input(self, *args: Any, **kwargs: Any) -> Any:
        """Transform and validate input parameters for the system.

        :param args: Positional inputs.
        :type args: Any
        :param kwargs: Keyword inputs.
        :type kwargs: Any
        :returns: Transformed input structure.
        :rtype: Any
        """
        ...

    @classmethod
    @abstractmethod
    def from_json(cls, data: dict, *, _verify: bool, _kind_of: str) -> System:
        """Create a system instance from a JSON-like data structure.

        :param data: Parsed JSON data.
        :type data: dict
        :param _verify: Whether to verify input completeness.
        :type _verify: bool
        :param _kind_of: Kind of system (used by concrete implementations).
        :type _kind_of: str
        :returns: New System instance.
        :rtype: System
        """
        ...

    @property
    @abstractmethod
    def default_input_units(self) -> Any:
        """Return mapping of expected input units for this system.

        :returns: Unit mapping for input parameters.
        :rtype: Any
        """
        ...

    @property
    @abstractmethod
    def default_internal_units(self) -> Any:
        """Return internal unit mapping used by this system implementation.

        :returns: Internal units mapping.
        :rtype: Any
        """
        ...

    def to_json(self) -> dict[str, Any]:
        """Serialize System instance to JSON.

        :returns: Dict; JSON serializable
        :rtype: dict[str, Any]
        """
        sys_units: Any = self.default_internal_units
        sys_input = self.default_input_units

        spot_units = u.DefaultSpotUnits
        spot_input = u.DefaultSpotInputUnits

        mode_input = u.DefaultPulsationsInputUnits
        mode_units = u.DefaultPulsationsUnits

        json_data: dict[str, Any] = {
            "system": {
                attr: (getattr(self, attr) * sys_units.system[attr]).to(sys_input["system"][attr]).value
                for attr in self.ALL_KWARGS
            },
        }

        for component, instance in self.components.items():
            comp_dict: dict[str, Any] = {
                attr: (getattr(instance, attr) * sys_units[component][attr]).to(sys_input[component][attr]).value
                for attr in self.STAR_ALL_KWARGS
                if getattr(instance, attr) is not None
            }

            if getattr(instance, "atmosphere", None):
                comp_dict["atmosphere"] = instance.atmosphere
            if getattr(instance, "limb_darkening_coefficients", None):
                comp_dict["limb_darkening_coefficients"] = instance.limb_darkening_coefficients

            if instance.has_spots():
                spot_list: list[dict[str, Any]] = [
                    {
                        attr: (getattr(spot, attr) * spot_units[attr]).to(spot_input[attr]).value
                        for attr in spot.ALL_KWARGS
                    }
                    for spot in instance.spots.values()
                ]
                comp_dict["spots"] = spot_list

            # this is just to avoid warning about instance not having pulsations in static type checker
            if hasattr(instance, "pulsations") and instance.has_pulsations():
                mode_list: list[dict[str, Any]] = [
                    {
                        attr: (getattr(mode, attr) * mode_units[attr]).to(mode_input[attr]).value
                        if attr != "tidally_locked"
                        else getattr(mode, attr)
                        for attr in mode.ALL_KWARGS
                    }
                    for mode in instance.pulsations.values()
                ]
                comp_dict["pulsations"] = mode_list

            json_data[component] = comp_dict

        return json_data

    def assign_pulsations_amplitudes(self, normalisation_constant: Float = 1.0) -> None:
        """Assign pulsation amplitudes to component modes using RV amplitude.

        :param normalisation_constant: Multiplicative normalisation constant.
        :type normalisation_constant: elisa.types.Float
        :returns: None
        :rtype: None
        """
        if self._components is None:
            return

        for component_instance in self._components.values():
            if component_instance.has_pulsations():
                pulsations.assign_amplitudes(component_instance, normalisation_constant)

    def init_properties(self, **kwargs: Any) -> None:
        """Initialise system attributes from keyword arguments.

        :param kwargs: Mapping of attribute names to values.
        :type kwargs: Any
        :returns: None
        :rtype: None
        """
        logger.debug("initialising properties of system %s, values: %s", self.name, kwargs)
        for kwarg, value in kwargs.items():
            setattr(self, kwarg, value)

    def has_pulsations(self) -> bool:
        """Return True if any component has pulsations.

        :returns: True when at least one component has pulsations.
        :rtype: bool
        """
        return any(instance.has_pulsations() for instance in self.components.values())

    def has_spots(self) -> bool:
        """Return True if any component has spots.

        :returns: True when at least one component has spots.
        :rtype: bool
        """
        return any(instance.has_spots() for instance in self.components.values())

    @staticmethod
    def object_params_validity_check(components: dict[str, Body], mandatory_kwargs: list[str] | tuple[str]) -> None:
        """Validate that provided component objects are correctly initialised.

        :param components: Mapping of component name to component instance.
        :type components: dict[str, elisa.base.body.Body]
        :param mandatory_kwargs: List of attributes required on each component.
        :type mandatory_kwargs: list[str] | tuple[str]
        :raises TypeError: When a component is not an instance of Body.
        :raises ValueError: When mandatory attributes are missing.
        :returns: None
        :rtype: None
        """
        for component, component_instance in components.items():
            if not isinstance(component_instance, Body):
                message = f"Component `{component}` is not instance of class {Body.__name__}"
                raise TypeError(message)

        # checking if system components have all mandatory parameters initialised
        missing_kwargs: list[str] = []
        for component, component_instance in components.items():
            for kwarg in mandatory_kwargs:
                if utils.is_empty(getattr(component_instance, kwarg)):
                    missing_kwargs.extend([f"`{kwarg}`"])

            if len(missing_kwargs) != 0:
                message = (
                    f"Mising argument(s): {', '.join(missing_kwargs)} in {component} component Star class"
                )
                raise ValueError(message)

    def kwargs_serializer(self) -> dict[str, Any]:
        """Serialize system keyword arguments for re-initialisation.

        :returns: Mapping of argument names to serializable values.
        :rtype: dict[str, Any]
        """
        cls_name = type(self).__name__
        serialized_kwargs: dict[str, Any] = {}
        for kwarg in self.ALL_KWARGS:
            value = getattr(self, kwarg)
            if isinstance(value, u.Quantity):
                serialized_kwargs[kwarg] = value
            else:
                def_unit = getattr(u.default_unit_map[cls_name], kwarg)
                serialized_kwargs[kwarg] = value if def_unit == u.dimensionless_unscaled else value * def_unit
        return serialized_kwargs

    def setup_betas(self) -> None:
        """Set up default gravity darkening parameters for components.

        :returns: None
        :rtype: None
        """
        for instance in self.components.values():
            if utils.is_empty(instance.gravity_darkening):
                instance.gravity_darkening = interpolate_bolometric_gravity_darkening(instance.t_eff)

    setup_gravity_darkening = setup_betas

    @abstractmethod
    def get_positions_method(self, *args: Any, **kwargs: Any) -> Any:
        """Return a callable that computes orbital positions for this system.

        :param args: Positional arguments forwarded to concrete implementation.
        :type args: Any
        :param kwargs: Keyword arguments forwarded to concrete implementation.
        :type kwargs: Any
        :returns: Callable that computes positions or an appropriate descriptor.
        :rtype: Any
        """
        ...
