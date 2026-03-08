from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

import numpy as np
from scipy import optimize

from elisa import const, settings, utils
from elisa import umpy as up
from elisa import units as u
from elisa.base.container import SystemPropertiesContainer
from elisa.base.curves import utils as rv_utils
from elisa.base.error import MorphologyError
from elisa.base.star import Star
from elisa.base.system import System
from elisa.base.types import FLOAT
from elisa.binary_system import graphic, model
from elisa.binary_system import radius as bsradius
from elisa.binary_system import utils as bsutils
from elisa.binary_system.container import OrbitalPositionContainer
from elisa.binary_system.curves import c_router, lc, rv
from elisa.binary_system.orbit import orbit
from elisa.binary_system.surface import mesh
from elisa.binary_system.surface.temperature import interpolate_albedo
from elisa.binary_system.transform import BinarySystemProperties
from elisa.logger import getLogger
from elisa.opt.fsolver import fsolve

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any

    from numpy.typing import NDArray

    from elisa.types import ComponentName, Float
    from elisa.units import _DefaultBinarySystemInputUnits, _DefaultBinarySystemUnits

logger = getLogger("binary_system.system")

ComponentSelection: TypeAlias = Literal["primary", "secondary", "all", "both"]

NUM_LAGRANGE_POINTS = 5
MAXIMAL_RADIUS_BOUNDARY = 30.0


class BinarySystem(System):
    """Store and calculate properties of a binary system.

    Child class of :class:`elisa.base.system.System`.

    The class can be imported directly::

        >>> from elisa import BinarySystem

    After initialization, apart from the attributes already defined by the
    user with the arguments, the user has access to the following attributes:

        :mass_ratio: float; secondary mass / primary mass
        :semi_major_axis: float; semi major axis of system in physical units
        :morphology: str; morphology of the system:

                      :`detached`: both components are not filling their respective Roche lobes,
                      :`semi-detached`: one of the components is filling its Roche lobe,
                      :`double-contact`: both components fill their Roche lobes,
                      :`over-contact`: components are physically connected with a ``neck``

    ``BinarySystem`` requires instances of :class:`elisa.base.star.Star` in
    ``primary`` and ``secondary`` argument with the following mandatory
    arguments:

        :param mass: float; If mass is int, np.int, float, np.float, program assumes solar mass as its unit.
                            If mass is astropy.unit.quantity.Quantity instance, program converts it to default units.
        :param t_eff: float; Accepts value in any temperature unit. If your input is without unit,
                             function assumes that supplied value is in K.
        :param surface_potential: float; generalized surface potential (Wilson 79)
        :param synchronicity: float; synchronicity F (omega_rot / omega_orb), equals 1 for synchronous rotation

    The following optional arguments are also available:

        :param metallicity: float; log[M/H] default value is 0.0
        :param gravity_darkening: float; gravity darkening factor, if not supplied, it is interpolated
                                         from Claret 2003 based on t_eff
        :param albedo: float; surface albedo, value from <0, 1> interval, if not supplied,
                              Claret 2001 will be used for interpolation
        :param limb_darkening_coefficients: Union[float, dict]; optional limb darkening coefficients
                                            used for the whole star useful in case the modeled star is outside the
                                            supported range of atmospheric parameters. Limb darkening coefficients can
                                            be supplied as dict {passband: ld_coefs}. If unused, elisa will
                                            interpolate the values from supplied limb-darkening tables.

    Each component instance will after initialization contain following
    attributes:

        :critical_surface_potential: float; potential of the star required to fill its Roche lobe
        :equivalent_radius: float; radius of a sphere with the same volume as a component (in SMA units)
        :filling_factor: float: calculated as (Omega_{inner} - Omega) / (Omega_{inner} - Omega_{outter})

                            :filling factor < 0: component does not fill its Roche lobe
                            :filling factor = 0: component fills preciselly its Roche lobe
                            :1 > filling factor > 0: component overflows its Roche lobe
                            :filling factor = 1: upper boundary of the filling factor, higher value would lead to
                                                 the mass loss trough Lagrange point L2

        Radii at periastron (in SMA units)
            :polar_radius: float; radius of a star towards the pole of the star
            :side_radius: float; radius of a star in the direction perpendicular to the pole
                                 and direction of a companion
            :backward_radius: float; radius of a star in the opposite direction as the binary companion
            :forward_radius: float; radius of a star towards the binary companion,
                                    returns numpy.NaN if the system is over-contact

    The ``BinarySystem`` can be initialized either by using valid class
    arguments, e.g.::

        >>> from elisa import BinarySystem
        >>> from elisa import Star
        >>> # noinspection PyShadowingNames
        >>> from astropy import units as u
        >>>
        >>> primary = Star(
        >>>     mass=2.15 * u.solMass,
        >>>     surface_potential=3.6,
        >>>     synchronicity=1.0,
        >>>     t_eff=10000 * u.K,
        >>>     gravity_darkening=1.0,
        >>>     discretization_factor=5,
        >>>     albedo=0.6,
        >>>     metallicity=0.0,
        >>> )
        >>>
        >>> secondary = Star(
        >>>     mass=0.45 * u.solMass,
        >>>     surface_potential=5.39,
        >>>     synchronicity=1.0,
        >>>     t_eff=8000 * u.K,
        >>>     gravity_darkening=1.0,
        >>>     albedo=0.6,
        >>>     metallicity=0,
        >>> )
        >>>
        >>> bs = BinarySystem(
        >>>     primary=primary,
        >>>     secondary=secondary,
        >>>     argument_of_periastron=58 * u.deg,
        >>>     gamma=-30.7 * u.km / u.s,
        >>>     period=2.5 * u.d,
        >>>     eccentricity=0.0,
        >>>     inclination=85 * u.deg,
        >>>     primary_minimum_time=2440000.00000 * u.d,
        >>>     phase_shift=0.0,
        >>>     distance=162 * u.pc,
        >>> )

    It can also be initialized by using the
    :meth:`BinarySystem.from_json` method that accepts various parameter
    combinations. See the docstring of :meth:`from_json` for details.

    The orbit of the binary system can be modeled using
    :meth:`calculate_orbital_motion`, e.g.::
        >>> from elisa import get_default_binary
        >>>> binary = get_default_binary()
        >>> binary.calculate_orbital_motion(np.linspace(0, 1))

    The class contains substantial plotting capability in the
    ``BinarySystem.plot`` module. Plot functions can be called as functions of
    the plot module, e.g.::

        >>> binary.plot.surface(phase=0.1, colormap='temperature')

    Similarly, an animation of the orbital motion can be produced using
    ``BinarySystem.animation`` module and its function
    ``orbital_motion(*args)``.

    List of valid input system arguments:

    :param primary: instance of primary component
    :type primary: Star
    :param secondary: instance of secondary component
    :type secondary: Star
    :param inclination: Inclination of the system. If unit is not supplied,
        value in degrees is assumed.
    :type inclination: Float | Any
    :param period: Orbital period of binary star system. If unit is not
        specified, default period unit is assumed (days).
    :type period: Float | Any
    :param eccentricity: Value from <0, 1> interval.
    :type eccentricity: Float
    :param argument_of_periastron: Argument of periastron.
    :type argument_of_periastron: Float | Any
    :param gamma: Center of mass velocity. Expected type is
        astropy.units.quantity.Quantity, numpy.float or numpy.int, otherwise
        TypeError will be raised. If unit is not specified, default velocity
        unit is assumed (m/s).
    :type gamma: Float | Any
    :param phase_shift: Phase shift of the primary eclipse with respect to the
        ephemeris. ``true_phase`` is used during calculations, where
        ``true_phase = phase + phase_shift``.
    :type phase_shift: Float
    :param primary_minimum_time: Reference primary minimum time.
    :type primary_minimum_time: Float | Any
    :param additional_light: Fraction of light that does not originate from
        the binary system.
    :type additional_light: Float
    :param distance: Distance between system and the observer.
    :type distance: Float | Any
    """

    MANDATORY_KWARGS = ("inclination", "period", "eccentricity", "argument_of_periastron")
    OPTIONAL_KWARGS = ("gamma", "phase_shift", "additional_light", "primary_minimum_time", "distance")
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    STAR_MANDATORY_KWARGS = ("mass", "t_eff", "surface_potential", "synchronicity")
    STAR_OPTIONAL_KWARGS = ("metallicity", "gravity_darkening", "albedo")
    STAR_ALL_KWARGS = STAR_MANDATORY_KWARGS + STAR_OPTIONAL_KWARGS

    def __init__(self, primary: Star, secondary: Star, name: str | None = None, **kwargs: Any) -> None:
        """Initialize the binary system.

        :param primary: Primary component.
        :type primary: Star
        :param secondary: Secondary component.
        :type secondary: Star
        :param name: Optional system name.
        :type name: str | None
        :param kwargs: Binary-system keyword arguments.
        :type kwargs: Any
        :return: ``None``.
        :rtype: None
        """
        utils.invalid_kwarg_checker(kwargs, BinarySystem.ALL_KWARGS, self.__class__)
        utils.check_missing_kwargs(BinarySystem.MANDATORY_KWARGS, kwargs, instance_of=BinarySystem)
        self.object_params_validity_check(
            {"primary": primary, "secondary": secondary},
            self.STAR_MANDATORY_KWARGS,
        )
        transformed_kwargs = self.transform_input(**kwargs)

        super().__init__(name, **transformed_kwargs)

        logger.info("initialising object %s", self.__class__.__name__)
        logger.debug("setting properties of components of class instance %s", self.__class__.__name__)

        self.plot = graphic.plot.Plot(self)
        self.animation = graphic.animation.Animation(self)

        self.primary = primary
        self.secondary = secondary
        self._components: dict[str, Star] = {"primary": self.primary, "secondary": self.secondary}

        self.orbit: orbit.Orbit | None = None
        self.period: Float = np.nan
        self.eccentricity: Float = np.nan
        self.argument_of_periastron: Float = np.nan
        self.primary_minimum_time: Float = 0.0
        self.phase_shift: Float = 0.0
        self.gamma: Float = 0.0
        self.mass_ratio: Float = self.secondary.mass / self.primary.mass

        self.init_properties(**transformed_kwargs)

        logger.debug("computing semi-major axis")
        self.semi_major_axis: Float = self.calculate_semi_major_axis()

        self.init_orbit()

        logger.debug("setting up critical surface potentials of components in periastron")
        self.setup_periastron_critical_potential()

        logger.debug("setting up morphological classification of binary system")
        self.morphology: str = self.compute_morphology()

        self.setup_components_radii(
            components_distance=self.orbit.periastron_distance,
            calculate_equivalent_radius=True,
        )
        self.setup_betas()
        self.setup_albedos()
        self.assign_pulsations_amplitudes(normalisation_constant=self.semi_major_axis)
        self.setup_discretisation_factor()

        self.t0: Float = self.primary_minimum_time

    @property
    def default_input_units(self) -> _DefaultBinarySystemInputUnits:
        """Return default units of initialization parameters.

        These units are used when values are provided without explicit units.

        :return: Default input units.
        :rtype: elisa.units.DefaultBinarySystemInputUnits
        """
        return u.DefaultBinarySystemInputUnits

    @property
    def default_internal_units(self) -> _DefaultBinarySystemUnits:
        """Return internal units of system parameters.

        :return: Default internal units.
        :rtype: elisa.units.DefaultBinarySystemUnits
        """
        return u.DefaultBinarySystemUnits

    @classmethod
    def from_json(
        cls,
        data: dict[str, Any],
        *,
        _verify: bool = True,
        _kind_of: str | None = None,
    ) -> BinarySystem:
        """Create a :class:`BinarySystem` instance from JSON-like input.

        Accepted input contains either standard parameters with component
        masses, or community-style parameters with ``semi_major_axis`` and
        ``mass_ratio``. Examples::

            {
              "system": {
                "inclination": 90.0,
                "period": 10.1,
                "argument_of_periastron": "90.0 deg",
                "gamma": 0.0,
                "eccentricity": 0.3,
                "primary_minimum_time": 0.0,
                "phase_shift": 0.0,
                "distance": 155,
              },
              "primary": {
                "mass": 2.0,
                "surface_potential": 7.1,
                "synchronicity": 1.0,
                "t_eff": 6500.0,
                "gravity_darkening": 1.0,
                "discretization_factor": 5,
                "albedo": 1.0,
                "metallicity": 0.0,
                "atmosphere": "ck04",
              },
              "secondary": {
                "mass": 2.0,
                "surface_potential": 7.1,
                "synchronicity": 1.0,
                "t_eff": 6500.0,
                "gravity_darkening": 1.0,
                "discretization_factor": 5,
                "albedo": 1.0,
                "metallicity": 0.0,
                "atmosphere": "black_body",
              },
            }

        or::

            {
              "system": {
                "inclination": 90.0,
                "period": 10.1,
                "argument_of_periastron": 90.0,
                "gamma": 0.0,
                "eccentricity": 0.3,
                "primary_minimum_time": 0.0,
                "phase_shift": 0.0,
                "semi_major_axis": 10.5,
                "mass_ratio": 0.5,
                "distance": "125 pc",
              },
              "primary": {
                "surface_potential": 7.1,
                "synchronicity": 1.0,
                "t_eff": 6500.0,
                "gravity_darkening": 1.0,
                "discretization_factor": 5,
                "albedo": 1.0,
                "metallicity": 0.0,
                "atmosphere": "black_body",
              },
              "secondary": {
                "surface_potential": 7.1,
                "synchronicity": 1.0,
                "t_eff": 6500.0,
                "gravity_darkening": 1.0,
                "discretization_factor": 5,
                "albedo": 1.0,
                "metallicity": 0.0,
                "atmosphere": "black_body",
              },
            }

        Default units when unit is not specified as string::

            {
                "inclination": [degrees],
                "period": [days],
                "argument_of_periastron": [degrees],
                "gamma": [m/s],
                "eccentricity": [dimensionless],
                "primary_minimum_time": [d],
                "phase_shift": [dimensionless],
                "distance": [pc],
                "mass": [solMass],
                "surface_potential": [dimensionless],
                "synchronicity": [dimensionless],
                "t_eff": [K],
                "gravity_darkening": [dimensionless],
                "discretization_factor": [degrees],
                "albedo": [dimensionless],
                "metallicity": [dimensionless],
                "semi_major_axis": [solRad],
                "mass_ratio": [dimensionless],
                "limb_darkening_coefficients": [dimensionless],
            }

        :param data: Input mapping describing the system and both components.
        :type data: dict[str, Any]
        :param _verify: If ``True``, validate the input schema before object
            creation.
        :type _verify: bool
        :param _kind_of: Optional explicit input kind override.
        :type _kind_of: str | None
        :return: Binary system created from the supplied data.
        :rtype: BinarySystem
        """
        data_cp = deepcopy(data)
        if _verify:
            bsutils.validate_binary_json(data_cp)

        kind_of = _kind_of or bsutils.resolve_json_kind(data_cp)
        if kind_of == "community":
            data_cp = bsutils.transform_json_community_to_std(data_cp)

        primary = Star(**data_cp["primary"])
        secondary = Star(**data_cp["secondary"])
        return cls(primary=primary, secondary=secondary, **data_cp["system"])

    @classmethod
    def from_fit_results(
        cls,
        results: dict[str, Any],
        atmosphere: dict[str, str] | None = None,
        limb_darkening_coefficients: dict[str, dict[str, Any]] | None = None,
    ) -> BinarySystem:
        """Build a binary system from standard fit-results format.

        :param results: Fit results in the form
            ``{'component': {'param_name': {'value': value, 'fixed': ...}}}``.
        :type results: dict[str, Any]
        :param atmosphere: Atmosphere model for each component, for example
            ``'ck04'`` or ``'bb'``.
        :type atmosphere: dict[str, str] | None
        :param limb_darkening_coefficients: Custom limb-darkening coefficients
            for each component and passband.
        :type limb_darkening_coefficients: dict[str, dict[str, Any]] | None
        :return: Binary system created from fit results.
        :rtype: BinarySystem
        """
        extra_parameters = {
            "atmosphere": atmosphere,
            "limb_darkening_coefficients": limb_darkening_coefficients,
        }

        data: dict[str, Any] = {}
        for key, component in results.items():
            if key == "r_squared":
                continue

            data[key] = {}
            for param, content in component.items():
                if param in {"spots", "pulsations"}:
                    features: list[dict[str, Any]] = []
                    for feature in content:
                        feature_data: dict[str, Any] = {}
                        for f_param, f_content in feature.items():
                            if f_param == "label":
                                continue
                            feature_data[f_param] = f_content["value"]
                        features.append(feature_data)
                    data[key][param] = features
                else:
                    data[key][param] = content["value"]

            for extra_param, value in extra_parameters.items():
                if value is not None and value.get(key) is not None:
                    data[key][extra_param] = value[key]

        return cls.from_json(data=data)

    def build_container(
        self,
        phase: Float | None = None,
        time: Float | None = None,
        *,
        build_pulsations: bool = True,
    ) -> OrbitalPositionContainer:
        """Build an :class:`OrbitalPositionContainer` at a given phase or time.

        The method returns a fully built model binary system at user-defined
        photometric phase or time of observation.

        :param phase: Photometric phase.
        :type phase: Float | None
        :param time: Observation time in Julian Date.
        :type time: Float | None
        :param build_pulsations: Whether to build pulsations.
        :type build_pulsations: bool
        :return: Fully built orbital-position container.
        :rtype: OrbitalPositionContainer
        :raises ValueError: If both ``phase`` and ``time`` are supplied.
        """
        if phase is not None and time is not None:
            message = (
                "Please specify whether you want to build your container either at given "
                "photometric `phase` or at given `time`."
            )
            raise ValueError(message)

        resolved_phase = phase if time is None else utils.jd_to_phase(time, period=self.period, t0=self.t0)
        position = self.calculate_orbital_motion(
            input_argument=resolved_phase,
            return_nparray=False,
            calculate_from="phase",
        )[0]
        orbital_position_container = OrbitalPositionContainer.from_binary_system(self, position)
        orbital_position_container.build(build_pulsations=build_pulsations)

        logger.info(
            "Orbital position container was successfully built at photometric phase %.2f.",
            resolved_phase,
        )
        return orbital_position_container

    def init(self) -> None:
        """Reinitialize the system after changing binary-system parameters.

        This also reinitializes both components in case values stored inside
        component instances were changed.

        :return: ``None``.
        :rtype: None
        """
        for component in settings.BINARY_COUNTERPARTS:
            getattr(self, component).init()

        self.__init__(primary=self.primary, secondary=self.secondary, **self.kwargs_serializer())

    @property
    def components(self) -> dict[str, Star]:
        """Return component objects.

        :return: Mapping of component names to :class:`Star` instances.
        :rtype: dict[str, Star]
        """
        return self._components

    def properties_serializer(self) -> dict[str, Any]:
        """Serialize binary-system properties to a JSON-compatible mapping.

        :return: Serialized properties in the form
            ``{'primary': {}, 'secondary': {}, 'system': {}}``.
        :rtype: dict[str, Any]
        """
        props = BinarySystemProperties.transform_input(**self.kwargs_serializer())
        props.update(
            {
                "semi_major_axis": self.semi_major_axis,
                "morphology": self.morphology,
                "mass_ratio": self.mass_ratio,
            },
        )
        return props

    def to_properties_container(self) -> SystemPropertiesContainer:
        """Convert system properties to :class:`SystemPropertiesContainer`.

        :return: Properties container.
        :rtype: SystemPropertiesContainer
        """
        return SystemPropertiesContainer(**self.properties_serializer())

    def init_orbit(self) -> None:
        """Initialize the orbit object for the current binary system.

        :return: ``None``.
        :rtype: None
        """
        logger.debug("re/initializing orbit in class instance %s / %s", self.__class__.__name__, self.name)
        orbit_kwargs = {key: getattr(self, key) for key in orbit.Orbit.ALL_KWARGS}
        self.orbit = orbit.Orbit(**orbit_kwargs)

    def is_eccentric(self) -> bool:
        """Resolve whether the system is eccentric.

        :return: ``True`` if the eccentricity is greater than zero.
        :rtype: bool
        """
        return self.eccentricity > 0

    def is_synchronous(self) -> bool:
        """Resolve whether the system is synchronous.

        The system is considered synchronous if synchronicity of both
        components is equal to ``1``.

        :return: ``True`` if both components are synchronous.
        :rtype: bool
        """
        return (self.primary.synchronicity == 1) and (self.secondary.synchronicity == 1)

    def calculate_semi_major_axis(self) -> Float:
        """Calculate the semi-major axis using Kepler's third law.

        :return: Semi-major axis in internal units.
        :rtype: Float
        """
        period = np.float64((self.period * u.DefaultBinarySystemUnits.system.period).to(u.TIME_UNIT))
        return (const.G * (self.primary.mass + self.secondary.mass) * period**2 / (4 * const.PI**2)) ** (1.0 / 3)

    def compute_morphology(self) -> str:  # noqa: C901
        """Determine and return system morphology.

        The morphology is determined from the current system parameters and the
        computed critical potentials.

        :return: Morphology of the system. One of ``detached``,
            ``semi-detached``, ``double-contact``, or ``over-contact``.
        :rtype: str
        :raises MorphologyError: If the system configuration is non-physical or
            inconsistent with the supplied potentials.
        """
        precs = 1e-8  # precision threshold for potential comparison
        morphology: str | None = None

        if self.primary.synchronicity == 1 and self.secondary.synchronicity == 1 and self.eccentricity == 0.0:
            lp = self.libration_potentials()
            self.primary.filling_factor = self.compute_filling_factor(self.primary.surface_potential, lp)
            self.secondary.filling_factor = self.compute_filling_factor(self.secondary.surface_potential, lp)

            if ((1 > self.secondary.filling_factor > 0) or (1 > self.primary.filling_factor > 0)) and abs(
                self.primary.filling_factor - self.secondary.filling_factor,
            ) > precs:
                message = "Detected over-contact binary system, but potentials of components are not the same."
                raise MorphologyError(message)

            if self.primary.filling_factor > 1 or self.secondary.filling_factor > 1:
                message = (
                    "Non-Physical system: primary_filling_factor or secondary_filling_factor is greater than 1. "
                    "Filling factor is obtained as following: "
                    "(Omega_{inner} - Omega) / (Omega_{inner} - Omega_{outter})."
                )
                raise MorphologyError(message)

            if (
                (abs(self.primary.filling_factor) < precs and self.secondary.filling_factor < 0)
                or (self.primary.filling_factor < 0 and abs(self.secondary.filling_factor) < precs)
                or (abs(self.primary.filling_factor) < precs and abs(self.secondary.filling_factor) < precs)
            ):
                morphology = "semi-detached"
            elif self.primary.filling_factor < 0 and self.secondary.filling_factor < 0:
                morphology = "detached"
            elif 1 >= self.primary.filling_factor > 0:
                morphology = "over-contact"
            elif self.primary.filling_factor > 1 or self.secondary.filling_factor > 1:
                message = "Non-Physical system: potential of components is too low."
                raise MorphologyError(message)
        else:
            self.primary.filling_factor = None
            self.secondary.filling_factor = None

            if (
                abs(self.primary.surface_potential - self.primary.critical_surface_potential) < precs
                and abs(self.secondary.surface_potential - self.secondary.critical_surface_potential) < precs
            ):
                morphology = "double-contact"
            elif (
                abs(self.primary.surface_potential - self.primary.critical_surface_potential) < precs
                and self.secondary.surface_potential > self.secondary.critical_surface_potential
            ) or (
                abs(self.secondary.surface_potential - self.secondary.critical_surface_potential) < precs
                and self.primary.surface_potential > self.primary.critical_surface_potential
            ):
                morphology = "semi-detached"
            elif (
                self.primary.surface_potential > self.primary.critical_surface_potential
                and self.secondary.surface_potential > self.secondary.critical_surface_potential
            ):
                morphology = "detached"
            else:
                message = "Non-Physical system. Change stellar parameters."
                raise MorphologyError(message)

        return morphology

    def setup_discretisation_factor(self) -> None:  # noqa: C901
        """Adjust discretization factors of both components to similar sizes.

        If neither component has its discretization factor set, the smaller
        component is adjusted according to the bigger one. If only one factor is
        supplied, the second one is adjusted with respect to the companion.

        Spot discretization factors are also adjusted when necessary.

        :return: ``None``.
        :rtype: None
        """

        def _adjust_alpha(adj_component: Star, ref_comp: Star) -> Float:
            return (
                ref_comp.discretization_factor
                * (ref_comp.equivalent_radius / adj_component.equivalent_radius)
                * (ref_comp.t_eff / adj_component.t_eff) ** 2
            )

        adj_comp: str | None = None
        adj: Star | None = None
        ref: Star | None = None

        if (
            self.primary.kwargs.get("discretization_factor") is None
            and self.secondary.kwargs.get("discretization_factor") is None
        ):
            cond_a = self.secondary.equivalent_radius * self.secondary.t_eff**2
            cond_b = self.primary.equivalent_radius * self.primary.t_eff**2
            if cond_a < cond_b:
                adj = self.secondary
                ref = self.primary
                adj_comp = "secondary"
            else:
                adj = self.primary
                ref = self.secondary
                adj_comp = "primary"
        elif self.secondary.kwargs.get("discretization_factor") is None:
            adj = self.secondary
            ref = self.primary
            adj_comp = "secondary"
        elif self.primary.kwargs.get("discretization_factor") is None:
            adj = self.primary
            ref = self.secondary
            adj_comp = "primary"

        if adj_comp is not None and adj is not None and ref is not None:
            adj.discretization_factor = _adjust_alpha(adj, ref)
            max_discretization = np.radians(settings.MAX_DISCRETIZATION_FACTOR)
            min_discretization = np.radians(settings.MIN_DISCRETIZATION_FACTOR)

            adj.discretization_factor = min(adj.discretization_factor, max_discretization)
            adj.discretization_factor = max(adj.discretization_factor, min_discretization)

            logger.info(
                "setting discretization factor of %s component to %.2f"
                "according to discretization factor of the companion.",
                adj_comp,
                up.degrees(adj.discretization_factor),
            )

        for component in settings.BINARY_COUNTERPARTS:
            instance = getattr(self, component)
            if instance.has_spots():
                for spot in instance.spots.values():
                    if not spot.kwargs.get("discretization_factor"):
                        spot.discretization_factor = instance.discretization_factor

    def transform_input(self, **kwargs: Any) -> dict[str, Any]:
        """Transform and validate input keyword arguments.

        :param kwargs: Keyword arguments for binary-system initialization.
        :type kwargs: Any
        :return: Transformed keyword-argument mapping.
        :rtype: dict[str, Any]
        """
        return BinarySystemProperties.transform_input(**kwargs)

    def setup_periastron_critical_potential(self) -> None:
        """Compute and set critical surface potentials for both components.

        Critical surface potential is defined as the potential at which a
        component fills its Roche lobe.

        :return: ``None``.
        :rtype: None
        """
        for component, instance in self.components.items():
            if component not in ("primary", "secondary"):
                message = f"Invalid component name: {component}"
                raise ValueError(message)

            critical_potential = self.critical_potential(
                component=component,
                components_distance=1.0 - self.eccentricity,
            )
            instance.critical_surface_potential = critical_potential

    def critical_potential(self, component: ComponentName, components_distance: Float) -> Float:
        """Return critical potential for the selected component.

        :param component: Target component, either ``primary`` or ``secondary``.
        :type component: Literal["primary", "secondary"]
        :param components_distance: Distance between components.
        :type components_distance: Float
        :return: Critical potential.
        :rtype: Float
        """
        synchronicity = self.primary.synchronicity if component == "primary" else self.secondary.synchronicity
        return self.critical_potential_static(component, components_distance, self.mass_ratio, synchronicity)

    @staticmethod
    def critical_potential_static(
        component: ComponentName,
        components_distance: Float,
        mass_ratio: Float,
        synchronicity: Float,
    ) -> Float:
        """Calculate critical potential for a binary-system component.

        :param component: Target component, either ``primary`` or ``secondary``.
        :type component: Literal["primary", "secondary"]
        :param components_distance: Distance between components.
        :type components_distance: Float
        :param mass_ratio: Mass ratio of the system.
        :type mass_ratio: Float
        :param synchronicity: Component synchronicity.
        :type synchronicity: Float
        :return: Critical potential.
        :rtype: Float
        :raises ValueError: If the solver fails or ``component`` is invalid.
        """
        args1 = (synchronicity, mass_ratio, components_distance)
        args2 = (*args1, 0.0, const.HALF_PI)
        solver_message = (
            "Iteration process to solve critical potential seems to lead nowhere "
            "(critical potential solver has failed)."
        )
        solver_err = ValueError(solver_message)

        if component == "primary":
            solution = optimize.newton(model.primary_potential_derivative_x, x0=1e-6, args=args1, tol=1e-12)
            if not up.isnan(solution):
                precalc_args = model.pre_calculate_for_potential_value_primary(*args2)
                args = (mass_ratio, *precalc_args)
                return abs(model.potential_value_primary(solution, *args))
            raise solver_err

        if component == "secondary":
            solution = optimize.newton(model.secondary_potential_derivative_x, x0=1e-6, args=args1, tol=1e-12)
            if not up.isnan(solution):
                precalc_args = model.pre_calculate_for_potential_value_secondary(*args2)
                args = (mass_ratio, *precalc_args)
                return abs(model.potential_value_secondary(components_distance - solution, *args))
            raise solver_err

        message = "Parameter `component` has incorrect value. Use `primary` or `secondary`."
        raise ValueError(message)

    def libration_potentials(self) -> list[Float]:
        """Return potentials in L3, L1, and L2 respectively.

        :return: Potentials ``[Omega(L3), Omega(L1), Omega(L2)]``.
        :rtype: list[Float]
        """
        return self.libration_potentials_static(self.orbit.periastron_distance, self.mass_ratio)

    @staticmethod
    def libration_potentials_static(periastron_distance: Float, mass_ratio: Float) -> list[Float]:
        """Return potentials in L3, L1, and L2 for supplied parameters.

        This is the static version of :meth:`libration_potentials`.

        :param periastron_distance: Periastron distance.
        :type periastron_distance: Float
        :param mass_ratio: Mass ratio of the system.
        :type mass_ratio: Float
        :return: Potential values at L3, L1, and L2.
        :rtype: list[Float]
        """

        def _potential(radius: Float | Sequence[Float]) -> list[Float]:
            theta = const.HALF_PI
            distance = periastron_distance

            if isinstance(radius, (float, int, np.floating, np.integer)):
                radii = [radius]
            elif isinstance(radius, (list, tuple, np.ndarray)):
                radii = list(radius)
            else:
                message = "Incorrect value of variable `radius`."
                raise TypeError(message)

            p_values: list[Float] = []
            for r_value in radii:
                phi, radius_value = (0.0, r_value) if r_value >= 0 else (const.PI, abs(r_value))
                block_a = 1.0 / radius_value
                block_b = mass_ratio / up.sqrt(
                    up.power(distance, 2)
                    + up.power(radius_value, 2)
                    - 2.0 * radius_value * up.cos(phi) * up.sin(theta) * distance,
                )
                block_c = (mass_ratio * radius_value * up.cos(phi) * up.sin(theta)) / up.power(distance, 2)
                block_d = 0.5 * (1 + mass_ratio) * up.power(radius_value, 2) * (1 - up.power(up.cos(theta), 2))
                p_values.append(block_a + block_b - block_c + block_d)
            return p_values

        lagrangian_points = BinarySystem.lagrangian_points_static(periastron_distance, mass_ratio)
        return _potential(lagrangian_points)

    def lagrangian_points(self) -> list[Float]:
        """Compute Lagrangian points for current system parameters.

        :return: X values of libration points ``[L3, L1, L2]`` respectively.
        :rtype: list[Float]
        """
        return self.lagrangian_points_static(self.orbit.periastron_distance, self.mass_ratio)

    @staticmethod
    def lagrangian_points_static(periastron_distance: Float, mass_ratio: Float) -> list[Float]:
        """Return Lagrangian points for supplied system parameters.

        This is the static version of :meth:`lagrangian_points`.

        :param periastron_distance: Periastron distance.
        :type periastron_distance: Float
        :param mass_ratio: Mass ratio of the system.
        :type mass_ratio: Float
        :return: X values of libration points ``[L3, L1, L2]`` respectively.
        :rtype: list[Float]
        """

        def _potential_dx(x: Float, distance: Float) -> Float:
            """Evaluate derivative of the general potential along the x-axis.

            Assumptions::

                primary.synchronicity = secondary.synchronicity = 1.0
                eccentricity = 0.0

            :param x: Coordinate along the x-axis.
            :type x: Float
            :param distance: Periastron distance of components.
            :type distance: Float
            :return: Potential derivative value.
            :rtype: Float
            """
            r_sqr = x**2
            rw_sqr = (distance - x) ** 2
            return (
                -(x / r_sqr ** (3.0 / 2.0))
                + ((mass_ratio * (distance - x)) / rw_sqr ** (3.0 / 2.0))
                + (mass_ratio + 1) * x
                - mass_ratio / distance**2
            )

        xs = np.linspace(-periastron_distance * 3.0, periastron_distance * 3.0, 100)
        round_to = 10
        unique_points: list[Float] = []
        lagrange: list[Float] = []

        for x_val in xs:
            try:
                old_settings = np.seterr(divide="raise", invalid="raise")
                _potential_dx(round(x_val, round_to), periastron_distance)
                np.seterr(**old_settings)
            except Exception as exc:  # noqa: BLE001
                logger.debug("invalid value passed to potential, exception: %s", str(exc))
                continue

            try:
                solution, _, ier, _ = fsolve(
                    _potential_dx,
                    x_val,
                    full_output=True,
                    args=(periastron_distance,),
                    xtol=1e-12,
                )
                if ier == 1:
                    rounded_solution = round(solution[0], 5)
                    if rounded_solution not in unique_points:
                        try:
                            value_dx = abs(round(_potential_dx(solution[0], periastron_distance), 4))
                            use_solution = value_dx == 0
                        except Exception as exc:  # noqa: BLE001
                            logger.debug("skipping solution for x: %s due to exception: %s", x_val, str(exc))
                            use_solution = False

                        if use_solution:
                            unique_points.append(rounded_solution)
                            lagrange.append(solution[0])
                            if len(lagrange) == NUM_LAGRANGE_POINTS - 2:
                                break
            except Exception as exc:  # noqa: BLE001
                logger.debug("solution for x: %s lead to nowhere, exception: %s", x_val, str(exc))
                continue

        return sorted(lagrange) if mass_ratio < 1.0 else sorted(lagrange, reverse=True)

    def compute_equipotential_boundary(  # noqa: C901, PLR0912
        self,
        components_distance: Float,
        plane: str,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute equipotential boundary cross-sections of both components.

        The method computes cross-sections of the Hill surface in the selected
        plane.

        :param components_distance: Distance between components.
        :type components_distance: Float
        :param plane: Cross-section plane, one of ``xy``, ``yz``, or ``zx``.
        :type plane: str
        :return: Tuple ``(points_primary, points_secondary)`` with 2-D Cartesian
            coordinates.
        :rtype: tuple[NDArray[numpy.float64], NDArray[numpy.float64]]
        :raises ValueError: If ``plane`` is invalid.
        """
        components = ["primary", "secondary"]
        points_primary: list[list[Float]] = []
        points_secondary: list[list[Float]] = []
        fn_map = {
            "primary": (model.potential_primary_fn, model.pre_calculate_for_potential_value_primary),
            "secondary": (model.potential_secondary_fn, model.pre_calculate_for_potential_value_secondary),
        }

        angles = np.linspace(0, const.FULL_ARC, 500, endpoint=True)
        for component in components:
            component_instance = getattr(self, component)
            synchronicity = component_instance.synchronicity

            for angle in angles:
                if utils.is_plane(plane, "xy"):
                    args = (synchronicity, self.mass_ratio, components_distance, angle, const.HALF_PI)
                elif utils.is_plane(plane, "yz"):
                    args = (synchronicity, self.mass_ratio, components_distance, const.HALF_PI, angle)
                elif utils.is_plane(plane, "zx"):
                    args = (synchronicity, self.mass_ratio, components_distance, 0.0, angle)
                else:
                    message = "Invalid choice of crossection plane, use only: `xy`, `yz`, `zx`."
                    raise ValueError(message)

                scipy_solver_init_value = np.array([components_distance / 10000.0])
                aux_args = (self.mass_ratio, *fn_map[component][1](*args))
                solver_args = (aux_args, component_instance.surface_potential)
                solution, _, ier, _ = fsolve(
                    fn_map[component][0],
                    scipy_solver_init_value,
                    full_output=True,
                    args=solver_args,
                    xtol=1e-12,
                )

                if ier != 1 or up.isnan(solution[0]):
                    continue

                radius_solution = solution[0]
                if not MAXIMAL_RADIUS_BOUNDARY >= radius_solution >= 0:
                    continue

                if utils.is_plane(plane, "yz"):
                    point = [radius_solution * up.sin(angle), radius_solution * up.cos(angle)]
                    if component == "primary":
                        points_primary.append(point)
                    else:
                        points_secondary.append(point)
                elif utils.is_plane(plane, "xz"):
                    if component == "primary":
                        points_primary.append([radius_solution * up.sin(angle), radius_solution * up.cos(angle)])
                    else:
                        points_secondary.append(
                            [-(radius_solution * up.sin(angle) - components_distance), radius_solution * up.cos(angle)],
                        )
                elif component == "primary":
                    points_primary.append([radius_solution * up.cos(angle), radius_solution * up.sin(angle)])
                else:
                    points_secondary.append(
                        [-(radius_solution * up.cos(angle) - components_distance), radius_solution * up.sin(angle)],
                    )

        return np.array(points_primary), np.array(points_secondary)

    def get_positions_method(self) -> Callable[..., Any]:
        """Return orbital-motion method used for position computation.

        :return: Callable used for orbital-motion computation.
        :rtype: Callable[..., Any]
        """
        return self.calculate_orbital_motion

    def calculate_orbital_motion(
        self,
        input_argument: Float | NDArray | None = None,
        *,
        return_nparray: bool = False,
        calculate_from: Literal["phase", "azimuths"] = "phase",
    ) -> NDArray[np.float64] | list[const.Position]:
        """Calculate orbital motion for supplied phases or azimuths.

        :param input_argument: Input phases or azimuths.
        :type input_argument: Float | NDArray | None
        :param return_nparray: If ``True``, return positions as NumPy array.
        :type return_nparray: bool
        :param calculate_from: Input mode, either ``phase`` or ``azimuths``.
        :type calculate_from: Literal["phase", "azimuths"]
        :return: Either a NumPy array of positions or a list of
            :class:`elisa.const.Position` instances.
        :rtype: NDArray[numpy.float64] | list[const.Position]
        """
        input_array = np.array([input_argument]) if np.isscalar(input_argument) else input_argument
        orbital_motion = (
            self.orbit.orbital_motion(phase=input_array)
            if calculate_from == "phase"
            else self.orbit.orbital_motion_from_azimuths(azimuth=input_array)
        )
        idx = up.arange(np.shape(input_array)[0], dtype=np.int64)
        positions = np.hstack((idx[:, np.newaxis], orbital_motion))
        return positions if return_nparray else [const.Position(*position) for position in positions]

    def calculate_components_radii(self, components_distance: Float) -> dict[str, dict[str, Float]]:
        """Calculate component radii.

        The method calculates polar, side, backward, and, if applicable,
        forward radii and returns them for both components.

        :param components_distance: Distance between components in SMA units.
        :type components_distance: Float
        :return: Mapping of component radii.
        :rtype: dict[str, dict[str, Float]]
        """
        radius_functions = [
            bsradius.calculate_polar_radius,
            bsradius.calculate_side_radius,
            bsradius.calculate_backward_radius,
        ]
        components = settings.BINARY_COUNTERPARTS

        if self.eccentricity == 0.0:
            corrected_potential = {component: getattr(self, component).surface_potential for component in components}
        else:
            corrected_potential_values = self.correct_potentials(distances=np.array([components_distance]))
            corrected_potential = {component: corrected_potential_values[component][0] for component in components}

        radii: dict[str, dict[str, Float]] = {"primary": {}, "secondary": {}}
        for component in components:
            instance = getattr(self, component)
            kwargs = {
                "synchronicity": instance.synchronicity,
                "mass_ratio": self.mass_ratio,
                "components_distance": components_distance,
                "surface_potential": corrected_potential[component],
                "component": component,
            }

            for fn in radius_functions:
                logger.debug(
                    "initialising %s for %s component",
                    " ".join(str(fn.__name__).split("_")[1:]),
                    component,
                )
                param_name = "_".join(str(fn.__name__).split("_")[1:])
                radii[component][param_name] = fn(**kwargs)

            if self.morphology != "over-contact":
                radii[component]["forward_radius"] = bsradius.calculate_forward_radius(**kwargs)

        return radii

    def setup_components_radii(
        self,
        components_distance: Float,
        *,
        calculate_equivalent_radius: bool = True,
    ) -> None:
        """Set up component radii and optionally equivalent radii.

        The method calculates equivalent, polar, side, backward and, if not W
        UMa, also forward radii and assigns them to component instances.

        :param components_distance: Distance between components in SMA units.
        :type components_distance: Float
        :param calculate_equivalent_radius: Some applications do not require
            equivalent-radius calculation.
        :type calculate_equivalent_radius: bool
        :return: ``None``.
        :rtype: None
        """
        radii = self.calculate_components_radii(components_distance)

        for component, radius_values in radii.items():
            instance = getattr(self, component)
            for key, value in radius_values.items():
                setattr(instance, key, value)

            if calculate_equivalent_radius:
                narrowed_component = cast("ComponentSelection", component)
                instance.equivalent_radius = self.calculate_equivalent_radius(narrowed_component)[component]

    def setup_albedos(self) -> None:
        """Set up default component albedos.

        Missing albedo values are interpolated from effective temperature.

        :return: ``None``.
        :rtype: None
        """
        for instance in self.components.values():
            instance.albedo = interpolate_albedo(instance.t_eff) if utils.is_empty(instance.albedo) else instance.albedo

    @staticmethod
    def compute_filling_factor(surface_potential: Float, lagrangian_points: Sequence[Float]) -> Float:
        """Compute filling factor of a binary star system.

        Filling factor is computed as::

            (Omega_{inner} - Omega) / (Omega_{inner} - Omega_{outter})

        where ``Omega_X`` denotes potential value and ``Omega`` is the
        potential of the given star. ``inner`` and ``outter`` are critical
        inner and outer potentials for the given binary star system.

        :param surface_potential: Surface potential of the component.
        :type surface_potential: Float
        :param lagrangian_points: Lagrangian-point potentials in order
            ``[L3, L1, L2]``.
        :type lagrangian_points: Sequence[Float]
        :return: Filling factor.
        :rtype: Float
        """
        return (lagrangian_points[1] - surface_potential) / (lagrangian_points[1] - lagrangian_points[2])

    def correct_potentials(
        self,
        phases: NDArray | list | None = None,
        component: ComponentSelection | None = "all",
        iterations: int = 2,
        distances: NDArray | None = None,
    ) -> dict[str, NDArray[np.float64]]:
        """Correct potentials so that component volume is conserved.

        The function calculates potential for each phase or distance in a way
        that conserves the volume of the component.

        :param phases: Orbital phases. Ignored when ``distances`` is supplied.
        :type phases: NDArray | None
        :param component: Target component, ``primary``, ``secondary``, or
            ``all``.
        :type component: Literal["primary", "secondary", "all", "both"] | None
        :param iterations: Number of correction iterations.
        :type iterations: int
        :param distances: Component distances. If not ``None``, corrected
            potentials are calculated for these distances.
        :type distances: NDArray | None
        :return: Corrected potentials for requested components.
        :rtype: dict[str, NDArray[numpy.float64]]
        :raises ValueError: If neither ``phases`` nor ``distances`` is
            supplied.
        """
        if distances is None:
            if phases is None:
                message = "Either `phases` or component `distances` have to be supplied."
                raise ValueError(message)
            data = self.orbit.orbital_motion(phases)
            distances_array = np.asarray(data[:, 0], dtype=FLOAT)
        else:
            distances_array = np.asarray(distances, dtype=FLOAT)

        component_ = cast("ComponentSelection", component)
        components = bsutils.component_to_list(component_)
        potentials: dict[str, NDArray[np.float64]] = {}

        for component_name in components:
            star = getattr(self, component_name)
            new_potentials = star.surface_potential * np.ones(distances_array.shape, dtype=FLOAT)

            points_equator, points_meridian = self.generate_equator_and_meridian_points(
                components_distance=1.0,
                component=component_name,
                surface_potential=star.surface_potential,
            )
            reference_volume = utils.calculate_volume_ellipse_approx(points_equator, points_meridian)
            equiv_r_mean = utils.calculate_equiv_radius(reference_volume)

            side_radii = np.empty(distances_array.shape, dtype=FLOAT)
            volume = np.empty(distances_array.shape, dtype=FLOAT)
            for _ in range(iterations):
                for idx, _potential in enumerate(new_potentials):
                    radii_args = (
                        star.synchronicity,
                        self.mass_ratio,
                        distances_array[idx],
                        _potential,
                        component_name,
                    )
                    side_radii[idx] = bsradius.calculate_side_radius(*radii_args)

                    points_equator, points_meridian = self.generate_equator_and_meridian_points(
                        components_distance=distances_array[idx],
                        component=component_name,
                        surface_potential=_potential,
                    )
                    volume[idx] = utils.calculate_volume_ellipse_approx(points_equator, points_meridian)

                equiv_r = utils.calculate_equiv_radius(volume)
                coeff = equiv_r_mean / equiv_r
                corrected_side_radii = coeff * side_radii

                new_potentials = np.array(
                    [
                        bsutils.potential_from_radius(
                            component_name,
                            corrected_side_radii[idx],
                            const.HALF_PI,
                            const.HALF_PI,
                            distances_array[idx],
                            self.mass_ratio,
                            star.synchronicity,
                        )
                        for idx in range(len(distances_array))
                    ],
                    dtype=FLOAT,
                )

            potentials[component_name] = np.asarray(new_potentials, dtype=FLOAT)

        return potentials

    def calculate_equivalent_radius(self, component: ComponentSelection) -> dict[str, Float]:
        """Return equivalent radius of the given component or components.

        Equivalent radius is the radius of the sphere with the same volume as
        the given component.

        :param component: Target component, ``primary``, ``secondary``, or
            ``all``.
        :type component: Literal["primary", "secondary", "all", "both"]
        :return: Equivalent radii mapping, for example ``{'primary': r_equiv}``.
        :rtype: dict[str, Float]
        """
        components = bsutils.component_to_list(component)
        r_equiv: dict[str, Float] = {}
        for component_name in components:
            star = getattr(self, component_name)
            points_equator, points_meridian = self.generate_equator_and_meridian_points(
                components_distance=1.0,
                component=component_name,
                surface_potential=star.surface_potential,
            )
            volume = utils.calculate_volume_ellipse_approx(points_equator, points_meridian)
            r_equiv[component_name] = utils.calculate_equiv_radius(volume)
        return r_equiv

    def calculate_bolometric_luminosity(self, components: ComponentSelection | None) -> dict[str, Float]:
        """Calculate bolometric luminosity of the given component or components.

        Bolometric luminosity is calculated from effective temperature and
        equivalent radius using black-body approximation.

        :param components: Target component, ``primary``, ``secondary``, or
            ``all``.
        :type components: Literal["primary", "secondary", "all", "both"] | None
        :return: Luminosity mapping, for example ``{'primary': L_bol}``.
        :rtype: dict[str, Float]
        """
        component_names = bsutils.component_to_list(components)
        r_equiv = {component: getattr(self, str(component)).equivalent_radius for component in component_names}

        luminosity: dict[str, Float] = {}
        for component_name in component_names:
            star = getattr(self, component_name)
            luminosity[component_name] = (
                4.0
                * const.PI
                * np.power(r_equiv[component_name] * self.semi_major_axis, 2)
                * const.STEFAN_BOLTZMAN_CONST
                * np.power(star.t_eff, 4)
            )
        return luminosity

    def generate_equator_and_meridian_points(
        self,
        components_distance: Float | NDArray[Float],
        component: ComponentName,
        surface_potential: Float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Generate contour points for the equator and meridian sections.

        The function calculates two arrays of points contouring equator and
        meridian while solving for the same x values.

        :param components_distance: Distance between components.
        :type components_distance: Float | NDArray[Float]
        :param component: Target component, either ``primary`` or ``secondary``.
        :type component: Literal["primary", "secondary"]
        :param surface_potential: Surface potential to use.
        :type surface_potential: Float
        :return: Tuple ``(equator_points, meridian_points)``.
        :rtype: tuple[NDArray[numpy.float64], NDArray[numpy.float64]]
        """
        x: NDArray[np.float64]

        star = getattr(self, component)
        discretization_factor = star.discretization_factor

        rad_args = (
            star.synchronicity,
            self.mass_ratio,
            components_distance,
            surface_potential,
            component,
        )
        backward_radius = bsradius.calculate_backward_radius(*rad_args)

        if self.morphology == "detached":
            num = int(const.PI // discretization_factor)
            theta = np.linspace(
                discretization_factor,
                const.PI - discretization_factor,
                num=num + 1,
                endpoint=True,
            )
            forward_radius = bsradius.calculate_forward_radius(*rad_args)
            a = 0.5 * (forward_radius + backward_radius)
            c = forward_radius - a
            x = a * up.cos(theta) + c
        elif self.morphology == "over-contact":
            num = int(const.HALF_PI // discretization_factor)
            theta = np.linspace(
                const.HALF_PI + discretization_factor,
                const.PI - discretization_factor,
                num=num + 1,
                endpoint=True,
            )
            forward_radius = (
                mesh.calculate_neck_position(self, return_polynomial=False)
                if component == "primary"
                else 1 - mesh.calculate_neck_position(self, return_polynomial=False)
            )
            a = 0.5 * (forward_radius + backward_radius)
            c = forward_radius - a
            x_back = a * up.cos(theta) + c
            x_front = np.linspace(forward_radius, c, num=num + 1, endpoint=True)
            x = np.concatenate((x_front, x_back))
        elif self.morphology in ["semi-detached", "double-contact"]:
            num = int(const.HALF_PI // discretization_factor)
            theta = np.linspace(
                const.HALF_PI + discretization_factor,
                const.PI - discretization_factor,
                num=num,
                endpoint=True,
            )
            forward_radius = bsradius.calculate_forward_radius(*rad_args)
            a = 0.5 * (forward_radius + backward_radius)
            c = forward_radius - a
            x_front = np.linspace(forward_radius - 0.05 * a, c, num=num + 1, endpoint=True)
            x_back = a * up.cos(theta) + c
            x = np.concatenate((x_front, x_back))
        else:
            message = f"Unsupported morphology `{self.morphology}`."
            raise ValueError(message)

        fn_cylindrical = getattr(model, f"potential_{component}_cylindrical_fn")
        precal_cylindrical = getattr(model, f"pre_calculate_for_potential_value_{component}_cylindrical")
        cylindrical_potential_derivative_fn = getattr(model, f"radial_{component}_potential_derivative_cylindrical")

        phi1 = const.HALF_PI * np.ones(x.shape)
        phi2 = up.zeros(x.shape)
        phi = up.concatenate((phi1, phi2))
        z = up.concatenate((x, x))

        args = (
            phi,
            z,
            components_distance,
            a / 2,
            precal_cylindrical,
            fn_cylindrical,
            cylindrical_potential_derivative_fn,
            surface_potential,
            self.mass_ratio,
            star.synchronicity,
        )
        points = mesh.get_surface_points_cylindrical(*args)

        if self.morphology != "over-contact":
            equator_points = np.vstack(
                ([0, 0, forward_radius], points[: points.shape[0] // 2, :], [0, 0, -backward_radius]),
            )
            meridian_points = np.vstack(
                ([0, 0, forward_radius], points[points.shape[0] // 2 :, :], [0, 0, -backward_radius]),
            )
            return equator_points, meridian_points

        equator_points = np.vstack((points[: points.shape[0] // 2, :], [0, 0, -backward_radius]))
        meridian_points = np.vstack((points[points.shape[0] // 2 :, :], [0, 0, -backward_radius]))
        return equator_points, meridian_points

    def compute_lightcurve(self, **kwargs: Any) -> dict[str, NDArray[np.float64]]:
        """Decide which light-curve generator function should be used.

        The selected generator depends on the binary system's basic properties.

        ``kwargs`` are passed to light-curve generator functions. Supported
        options include passbands, bandwidths, phases, and position method.

        :param kwargs: Arguments passed into light-curve generator functions.
        :type kwargs: Any
        :return: Generated light curve data by passband.
        :rtype: dict[str, NDArray[numpy.float64]]
        """
        curve_fn = c_router.resolve_curve_method(self, curve="lc")
        return curve_fn(**kwargs)

    def _compute_circular_synchronous_lightcurve(self, **kwargs: Any) -> dict[str, NDArray[np.float64]]:
        return lc.compute_circular_synchronous_lightcurve(self, **kwargs)

    def _compute_circular_spotty_asynchronous_lightcurve(self, **kwargs: Any) -> dict[str, NDArray[np.float64]]:
        return lc.compute_circular_spotty_asynchronous_lightcurve(self, **kwargs)

    def _compute_circular_pulsating_lightcurve(self, **kwargs: Any) -> dict[str, NDArray[np.float64]]:
        return lc.compute_circular_pulsating_lightcurve(self, **kwargs)

    def _compute_eccentric_spotty_lightcurve(self, **kwargs: Any) -> dict[str, NDArray[np.float64]]:
        return lc.compute_eccentric_spotty_lightcurve(self, **kwargs)

    def _compute_eccentric_lightcurve(self, **kwargs: Any) -> dict[str, NDArray[np.float64]]:
        return lc.compute_eccentric_lightcurve_no_spots(self, **kwargs)

    def compute_rv(self, **kwargs: Any) -> dict[str, NDArray[np.float64]]:
        """Compute radial-velocity curves using the requested method.

        ``kwargs`` may include:

        - ``method``: ``kinematic`` or ``radiometric``
        - ``position_method``: callable returning orbital positions
        - ``phases``: photometric phases

        :param kwargs: Radial-velocity computation arguments.
        :type kwargs: Any
        :return: Radial-velocity curves for primary and secondary component.
        :rtype: dict[str, NDArray[numpy.float64]]
        :raises ValueError: If ``method`` is unknown.
        """
        if kwargs["method"] == "kinematic":
            return rv.kinematic_radial_velocity(self, **kwargs)

        if kwargs["method"] == "radiometric":
            curve_fn = c_router.resolve_curve_method(self, curve="rv")
            rv_kwargs = rv_utils.include_passband_data_to_kwargs(**kwargs)
            return curve_fn(**rv_kwargs)

        message = (
            f"Unknown RV computing method `{kwargs['method']}`.\n"
            "List of available methods: [`kinematic`, `radiometric`]."
        )
        raise ValueError(message)

    def _compute_circular_synchronous_rv_curve(self, **kwargs: Any) -> dict[str, NDArray[np.float64]]:
        return rv.compute_circular_synchronous_rv_curve(self, **kwargs)

    def _compute_circular_spotty_asynchronous_rv_curve(self, **kwargs: Any) -> dict[str, NDArray[np.float64]]:
        return rv.compute_circular_spotty_asynchronous_rv_curve(self, **kwargs)

    def _compute_circular_pulsating_rv_curve(self, **kwargs: Any) -> dict[str, NDArray[np.float64]]:
        return rv.compute_circular_pulsating_rv_curve(self, **kwargs)

    def _compute_eccentric_spotty_rv_curve(self, **kwargs: Any) -> dict[str, NDArray[np.float64]]:
        return rv.compute_eccentric_spotty_rv_curve(self, **kwargs)

    def _compute_eccentric_rv_curve_no_spots(self, **kwargs: Any) -> dict[str, NDArray[np.float64]]:
        return rv.compute_eccentric_rv_curve_no_spots(self, **kwargs)
