from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import optimize

from elisa import const, units, utils
from elisa import umpy as up
from elisa.base.curves import utils as rv_utils
from elisa.base.star import Star
from elisa.base.system import System
from elisa.base.types import INT
from elisa.logger import getLogger
from elisa.opt.fsolver import fsolve
from elisa.single_system import graphic, model
from elisa.single_system import radius as sradius
from elisa.single_system import utils as sys_utils
from elisa.single_system.container import SinglePositionContainer
from elisa.single_system.curves import c_router, lc, rv
from elisa.single_system.orbit import orbit
from elisa.single_system.transform import SingleSystemProperties

# Backwards-compatible alias expected by many functions in this module
c = const

logger = getLogger("single_system.system")

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypedDict

    from numpy.typing import NDArray

    from elisa.types import Float, Int

    # TypedDicts for single-star JSON input shapes
    class SingleSystemSystem(TypedDict, total=False):
        inclination: Float
        rotation_period: Float
        gamma: Float | Int
        reference_time: Float
        phase_shift: Float
        distance: Float | str

    class SingleStarStandard(TypedDict, total=False):
        mass: Float | Int
        t_eff: Float
        gravity_darkening: Float
        discretization_factor: Int
        metallicity: Float
        polar_log_g: Float | str

    class SingleStarRadius(TypedDict, total=False):
        mass: Float | Int
        t_eff: Float
        gravity_darkening: Float
        discretization_factor: Int
        metallicity: Float
        equivalent_radius: Float | str

    class SingleStandardParams(TypedDict):
        system: SingleSystemSystem
        star: SingleStarStandard

    class SingleRadiusParams(TypedDict):
        system: SingleSystemSystem
        star: SingleStarRadius

    SingleSystemParams = SingleStandardParams | SingleRadiusParams


class SingleSystem(System):
    r"""Represent a rotating single-star system.

    Child class of :class:`elisa.base.system.System`.

    The class can be imported directly:

    ::

        from elisa import SingleSystem

    After initialization, apart from the attributes already defined by the user
    with the arguments, the user has access to the following attributes:

        :angular_velocity: float; angular velocity of the stellar rotation

    ``SingleSystem`` requires an instance of :class:`elisa.base.star.Star` in
    the ``star`` argument with the following mandatory arguments:

        :param mass: float; If mass is int, np.int, float, np.float, program
            assumes solar mass as its unit. If mass is an
            astropy.unit.quantity.Quantity instance, the program converts it to
            default units.
        :param t_eff: float; Accepts value in any temperature unit. If input is
            without unit, the supplied value is assumed to be in K.
        :param polar_log_g: float; :math:`\log_{10}` of the polar surface gravity

    The following optional arguments are also available:

        :param metallicity: float; log[M/H], default value is 0.0
        :param gravity_darkening: float; gravity-darkening factor. If not
            supplied, it is interpolated from Claret 2003 based on ``t_eff``.
        :param limb_darkening_coefficients: Union[float, dict]; optional limb
            darkening coefficients used for the whole star, useful when the
            modelled star is outside the supported range of atmospheric
            parameters. Limb-darkening coefficients can be supplied as
            ``{passband: ld_coefs}``. If unused, elisa interpolates the values
            from supplied limb-darkening tables.

    Each component instance will after initialization contain the following
    attributes:

        :critical_surface_potential: float; potential of the star required to
            fill its Roche lobe
        :equivalent_radius: float; radius of a sphere with the same volume as a
            component
        :polar_radius: float; radius of a star towards the pole of the star
        :equatorial_radius: float; radius of a star towards the pole of the star

    The :class:`SingleSystem` can be initialized either by using valid class
    arguments, e.g.:

    ::

        from astropy import units as u

        from elisa.single_system.system import SingleSystem
        from elisa.base.star import Star

        star = Star(
            mass=1.0*u.solMass,
            t_eff=5772*u.K,
            gravity_darkening=0.32,
            polar_log_g=4.43775*u.dex(u.cm/u.s**2),
            metallicity=0.0,
            discretization_factor=2
        )

        system = SingleSystem(
            star=star,
            gamma=0*u.km/u.s,
            inclination=90*u.deg,
            rotation_period=25.380*u.d,
            reference_time=0.0*u.d,
            distance=153*u.pc
        )

    or by using :meth:`SingleSystem.from_json`, which accepts various parameter
    combinations in dictionary form such as:

    ::

        data = {
            "system": {
                "inclination": 90.0,
                "rotation_period": 10.1,
                "gamma": "10000 K",  # quantity can be defined using string representation
                "reference_time": 0.5,
                "phase_shift": 0.0,
                "distance": 64  # pc
            },
            "star": {
                "mass": 1.0,
                "t_eff": 5772.0,
                "gravity_darkening": 0.32,
                "discretization_factor": 5,
                "metallicity": 0.0,
                "polar_log_g": "4.43775 dex(cm.s-2)"  # logarithmic units are supported
            }
        }

        single = SingleSystem.from_json(data)

    See the documentation for :meth:`from_json` for details.

    The rotation of the system can be modeled using
    :meth:`calculate_lines_of_sight`. E.g.:

    ::

        single_instance.calculate_lines_of_sight(np.linspace(0, 1))

    The class contains plotting capability in :mod:`elisa.single_system.graphic`
    with functions such as:

        - ``equipotential(args)``: zx cross-sections of equipotential surface
        - ``mesh(args)``: 3D mesh (scatter) plot of the surface points
        - ``wireframe(args)``: wireframe model of the star
        - ``surface(args)``: model of the star with various surface colormaps
          such as gravity acceleration, temperature, or radiance

    Plot functions can be called through the plot module. E.g.:

    ::

        single_instance.plot.surface(phase=0.1, colormap="temperature")

    Similarly, an animation of the rotational motion can be produced using
    ``SingleSystem.animation.rotational_motion(*args)``.

    List of valid system arguments:

    :param star: Instance of the single star.
    :type star: elisa.base.star.Star
    :param inclination: Inclination of the system. If a unit is not supplied,
        the value in degrees is assumed.
    :type inclination: float | astropy.unit.quantity.Quantity
    :param rotational_period: Orbital period of the binary star system. If a
        unit is not specified, the default period unit is assumed (days).
    :type rotational_period: float | int | astropy.units.quantity.Quantity
    :param reference_time: Reference time of the ephemeris.
    :type reference_time: float | int | astropy.units.quantity.Quantity
    :param phase_shift: Phase shift with respect to the ephemeris. During
        calculations, true phase is used, where ``true_phase = phase + phase_shift``.
    :type phase_shift: float
    :param additional_light: Fraction of light that does not originate from
        the :class:`SingleSystem`.
    :type additional_light: float
    :param gamma: Center-of-mass velocity. If a unit is not specified, the
        default velocity unit is assumed (m/s).
    :type gamma: float | astropy.unit.quantity.Quantity
    :param distance: Distance between system and observer.
    :type distance: float
    """

    MANDATORY_KWARGS = ("inclination", "rotation_period")
    OPTIONAL_KWARGS = ("reference_time", "phase_shift", "additional_light", "gamma", "distance")
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    STAR_MANDATORY_KWARGS = ("mass", "t_eff", "polar_log_g")
    STAR_OPTIONAL_KWARGS = ("metallicity", "gravity_darkening")
    STAR_ALL_KWARGS = STAR_MANDATORY_KWARGS + STAR_OPTIONAL_KWARGS

    def __init__(self, star: Star, name: str | None = None, **kwargs: Any) -> None:
        """Initialize a :class:`SingleSystem` instance.

        :param star: Stellar component of the system.
        :type star: elisa.base.star.Star
        :param name: Optional user-facing object name.
        :type name: str | None
        :param kwargs: System property values accepted by
            :attr:`SingleSystem.ALL_KWARGS`.
        :type kwargs: object
        """
        utils.invalid_kwarg_checker(kwargs, SingleSystem.ALL_KWARGS, self.__class__)
        utils.check_missing_kwargs(SingleSystem.MANDATORY_KWARGS, kwargs, instance_of=SingleSystem)
        self.object_params_validity_check({"star": star}, self.STAR_MANDATORY_KWARGS)
        kwargs = self.transform_input(**kwargs)

        super().__init__(name, **kwargs)

        logger.info("initialising object %s", self.__class__.__name__)
        logger.debug("setting properties of a star in class instance %s", self.__class__.__name__)

        self.plot = graphic.plot.Plot(self)
        self.animation = graphic.animation.Animation(self)

        self.star = star
        self._components = {"star": self.star}

        # default values of properties
        self.orbit = None
        self.rotation_period = up.NaN
        self.reference_time = 0
        self.angular_velocity = None
        self.period = self.rotation_period
        self.phase_shift = 0.0

        # set attributes and test whether all parameters were initialized
        # we already ensured that all kwargs are valid and all mandatory kwargs are present so lets set class attributes
        self.init_properties(**kwargs)

        # calculation of dependent parameters
        self.angular_velocity = orbit.angular_velocity(self.rotation_period)
        self.star.surface_potential = model.surface_potential_from_polar_log_g(
            self.star.polar_log_g,
            self.star.mass,
        )

        # orbit initialisation
        self.init_orbit()

        self.setup_critical_potential()
        self.check_stability()

        # this is also a check that the star surface is closed
        self.setup_radii(calculate_equivalent_radius=True)
        self.setup_betas()
        self.assign_pulsations_amplitudes()
        self.setup_discretisation_factor()

        # setting common reference to ephemeris
        self.period = self.rotation_period
        self.t0 = self.reference_time

    @property
    def default_input_units(self) -> type:
        """Return the default input units for :class:`SingleSystem`.

        :returns: DefaultSingleSystemInputUnits type container.
        :rtype: type
        """
        return units.DefaultSingleSystemInputUnits

    @property
    def default_internal_units(self) -> type:
        """Return the internal default units used by :class:`SingleSystem`.

        :returns: DefaultSingleSystemUnits type container.
        :rtype: type
        """
        return units.DefaultSingleSystemUnits

    @classmethod
    def from_json(
        cls,
        data: SingleSystemParams,
        *,
        _verify: bool = True,
        _kind_of: str | None = None,
    ) -> SingleSystem:
        r"""Create an instance from JSON in ``standard`` or ``radius`` format.

        :param data: Input dictionary following the expected schema.
        :type data: dict[str, object]
        :param _verify: If ``True``, validate incoming data before constructing
            the object.
        :type _verify: bool
        :param _kind_of: Optional override for the JSON kind, e.g. ``"radius"``.
        :type _kind_of: str | None
        :returns: Constructed :class:`SingleSystem` instance.
        :rtype: elisa.single_system.system.SingleSystem
        """
        # Convert TypedDict to plain mapping for downstream functions that
        # expect a standard dict[str, Any]. Keep a local mutable copy because
        # some transforms return plain dicts.
        raw_data: dict[str, Any] = dict(deepcopy(data))

        if _verify:
            sys_utils.validate_single_json(raw_data)

        kind_of = _kind_of or sys_utils.resolve_json_kind(raw_data)
        if kind_of == "radius":
            raw_data = sys_utils.transform_json_radius_to_std(raw_data)

        star_kwargs: dict[str, Any] = dict(raw_data["star"])
        system_kwargs: dict[str, Any] = dict(raw_data["system"])

        star = Star(**star_kwargs)
        return cls(star=star, **system_kwargs)

    def build_container(
        self,
        phase: Float | None = None,
        time: Float | None = None,
        *,
        build_pulsations: bool = True,
    ) -> SinglePositionContainer:
        r"""Return a fully built position container for a requested phase or time.

        Exactly one of ``phase`` or ``time`` should be supplied. If ``time`` is
        provided, the corresponding phase is derived from the system ephemeris.

        :param phase: Photometric phase where the container should be built.
        :type phase: elisa.types.Float | None
        :param time: Julian date for which a phase is computed and used.
        :type time: elisa.types.Float | None
        :param build_pulsations: Whether to include pulsation perturbations.
        :type build_pulsations: bool
        :returns: Built :class:`SinglePositionContainer` for the requested
            phase or time.
        :rtype: elisa.single_system.container.SinglePositionContainer
        :raises ValueError: If both ``phase`` and ``time`` are supplied.
        """
        if phase is not None and time is not None:
            msg = (
                "Please specify whether you want to build your container EITHER at given photometric "
                "`phase` or at given `time`."
            )
            raise ValueError(msg)

        phase = phase if time is None else utils.jd_to_phase(
            time,
            period=self.period,
            t0=self.reference_time,
        )

        position = self.calculate_lines_of_sight(
            input_argument=phase,
            return_nparray=False,
            calculate_from="phase",
        )[0]
        position_container = SinglePositionContainer.from_single_system(self, position)
        position_container.build(build_pulsations=build_pulsations)

        logger.info("Orbital position container was successfully built at photometric phase %.2f.", phase)
        return position_container

    @classmethod
    def is_property(cls, kwargs: dict[str, object]) -> None:
        """Check whether provided kwargs are valid :class:`SingleSystem` properties.

        :param kwargs: Mapping of property names to check.
        :type kwargs: dict[str, object]
        :raises AttributeError: If any entry in ``kwargs`` is not a valid property.
        """
        is_not = [f"`{k}`" for k in kwargs if k not in cls.ALL_KWARGS]
        if is_not:
            msg = f"Arguments {', '.join(is_not)} are not valid {cls.__name__} properties."
            raise AttributeError(msg)

    def critical_break_up_radius(self) -> Float:
        """Return the critical break-up equatorial radius for the system.

        :returns: Critical equatorial radius.
        :rtype: elisa.types.Float
        """
        return np.power(c.G * self.star.mass / np.power(self.angular_velocity, 2), 1.0 / 3.0)

    def critical_break_up_velocity(self) -> Float:
        """Return the critical break-up equatorial rotational velocity.

        :returns: Critical equatorial velocity.
        :rtype: elisa.types.Float
        """
        return np.power(c.G * self.star.mass * self.angular_velocity, 1.0 / 3.0)

    def get_info(self) -> None:
        """Return formatted summary information about the system."""

    def init(self) -> None:
        """Re-initialize the instance after parameter changes."""
        logger.info("re/initialising class instance %s", SingleSystem.__name__)
        self.__init__(star=self.star, **self.kwargs_serializer())

    def init_orbit(self) -> None:
        """Initialize the orbit helper used for computing lines of sight."""
        logger.debug("re/initializing orbit in class instance %s / %s", self.__class__.__name__, self.name)
        orbit_kwargs = {key: getattr(self, key) for key in orbit.Orbit.ALL_KWARGS}
        self.orbit = orbit.Orbit(**orbit_kwargs)

    def setup_discretisation_factor(self) -> None:
        """Propagate star discretisation factor to spots without explicit value."""
        if self.star.has_spots():
            for spot in self.star.spots.values():
                if not spot.kwargs.get("discretization_factor"):
                    spot.discretization_factor = self.star.discretization_factor

    @staticmethod
    def is_eccentric() -> bool:
        """Return whether the system orbit is eccentric.

        Single-star systems are treated as non-eccentric in this model.

        :returns: Always ``False``.
        :rtype: bool
        """
        return False

    def calculate_radii(self) -> dict[str, dict[str, Float]]:
        """Calculate important stellar radii.

        :returns: Mapping of component name to calculated radii.
        :rtype: dict[str, dict[str, elisa.types.Float]]
        :raises ValueError: If one of the radius solvers fails for a
            non-physical system.
        """
        fns = [sradius.calculate_polar_radius, sradius.calculate_equatorial_radius]
        radii: dict[str, dict[str, Float]] = {"star": {}}

        for fn in fns:
            logger.debug(
                "initialising %s for the star",
                " ".join(str(fn.__name__).split("_")[1:]),
            )
            param = "_".join(str(fn.__name__).split("_")[1:])
            kwargs = {
                "mass": self.star.mass,
                "angular_velocity": self.angular_velocity,
                "surface_potential": self.star.surface_potential,
            }
            try:
                radius = fn(**kwargs)
            except Exception as err:
                msg = (
                    f"Function {fn.__name__} was not able to calculate its radius. "
                    f"Your system is not physical. Exception: {err}"
                )
                raise ValueError(msg) from err

            radii["star"][param] = radius

        return radii

    def setup_radii(self, *, calculate_equivalent_radius: bool = True) -> None:
        """Calculate and assign important stellar radii.

        :param calculate_equivalent_radius: Whether equivalent radius should
            also be computed and assigned.
        :type calculate_equivalent_radius: bool
        """
        radii = self.calculate_radii()
        instance: Star = self.star

        for key, value in radii["star"].items():
            setattr(instance, key, value)

        if calculate_equivalent_radius:
            instance.equivalent_radius = self.calculate_equivalent_radius()

    @property
    def components(self) -> dict[str, Star]:
        """Return system components.

        :returns: Mapping of component name to :class:`Star` instance.
        :rtype: dict[str, elisa.base.star.Star]
        """
        return self._components

    def calculate_equipotential_boundary(self) -> NDArray[Float]:
        """Calculate the equipotential boundary of the star in zx or yz plane.

        :returns: Array of boundary points with columns ``x`` and ``z``.
        :rtype: NDArray[elisa.types.Float]
        """
        points: list[list[Float]] = []
        angles = np.linspace(0, c.FULL_ARC, 300, endpoint=True)
        init_val = -c.G * self.star.mass / self.star.surface_potential
        scipy_solver_init_value = np.array([init_val])

        for angle in angles:
            precalc_args = (self.star.mass, self.angular_velocity, angle)
            args = (
                model.pre_calculate_for_potential_value(*precalc_args),
                self.star.surface_potential,
            )
            solution, _, ier, _ = fsolve(
                model.potential_fn,
                scipy_solver_init_value,
                full_output=True,
                args=args,
            )
            if ier == 1 and not np.isnan(solution[0]):
                radius = solution[0]
            else:
                continue

            points.append([radius * np.sin(angle), radius * np.cos(angle)])

        return np.array(points)

    def properties_serializer(self) -> dict[str, object]:
        """Serialize transformed system properties.

        :returns: Serialized and transformed property mapping.
        :rtype: dict[str, object]
        """
        props = SingleSystemProperties.transform_input(**self.kwargs_serializer())
        props.update(
            {
                "angular_velocity": self.angular_velocity,
            },
        )
        return props

    def transform_input(self, **kwargs: Any) -> dict[str, object]:
        """Transform and validate input keyword arguments.

        :param kwargs: Raw keyword arguments supplied for system properties.
        :type kwargs: object
        :returns: Transformed property mapping.
        :rtype: dict[str, object]
        """
        return SingleSystemProperties.transform_input(**kwargs)

    def setup_critical_potential(self) -> None:
        """Calculate and assign the critical surface potential."""
        self.star.critical_surface_potential = self.calculate_critical_potential()

    def calculate_critical_potential(self) -> Float:
        """Compute the critical surface potential.

        Critical surface potential is the potential for which the component
        remains stable for the given mass and rotation period.

        :returns: Critical surface potential.
        :rtype: elisa.types.Float
        :raises ValueError: If the root-finding iteration does not converge to
            a valid solution.
        """
        precalc_args = self.star.mass, self.angular_velocity, c.HALF_PI
        args = (model.pre_calculate_for_potential_value(*precalc_args), 0.0)

        x0 = -c.G * self.star.mass / self.star.surface_potential
        solution = optimize.newton(
            model.radial_potential_derivative,
            x0,
            args=args[0],
            tol=1e-12,
        )
        if np.isnan(solution):
            msg = (
                "Iteration process to solve critical potential seems to lead nowhere "
                "(critical potential solver has failed)."
            )
            raise ValueError(msg)

        return model.potential_fn(solution, *args)

    def check_stability(self) -> None:
        """Check whether the star is rotationally stable.

        :raises ValueError: If the star rotates above critical break-up
            velocity.
        """
        if self.star.critical_surface_potential < self.star.surface_potential:
            msg = "Non-physical system. Star rotation is above critical break-up velocity."
            raise ValueError(msg)

    def get_positions_method(self) -> Callable[..., Any]:
        """Return the method used to calculate observer line-of-sight positions.

        :returns: Bound position calculation method.
        :rtype: collections.abc.Callable[..., typing.Any]
        """
        return self.calculate_lines_of_sight

    def calculate_lines_of_sight(
        self,
        input_argument: NDArray | Float | None = None,
        *,
        return_nparray: bool = False,
        calculate_from: str = "phase",
    ) -> NDArray[Float] | list[const.Position]:
        """Return vectors oriented in the direction star -> observer.

        Positions can be calculated either from photometric phases or from
        azimuths, depending on ``calculate_from``.

        :param input_argument: Input phases or azimuths.
        :type input_argument: NDArray | elisa.types.Float | None
        :param return_nparray: If ``True``, return positions as a NumPy array.
        :type return_nparray: bool
        :param calculate_from: Either ``"phase"`` or ``"azimuth"``-like input
            mode supported by the orbit helper.
        :type calculate_from: str
        :returns: Either an array of positions or a list of
            :class:`elisa.const.Position` instances.
        :rtype: NDArray[elisa.types.Float] | list[elisa.const.Position]
        """
        input_argument = np.array([input_argument]) if np.isscalar(input_argument) else input_argument
        rotational_motion = (
            self.orbit.rotational_motion(phase=input_argument)
            if calculate_from == "phase"
            else self.orbit.rotational_motion_from_azimuths(azimuth=input_argument)
        )
        idx = np.arange(np.shape(input_argument)[0], dtype=INT)[:, np.newaxis]
        positions = np.hstack((idx, np.full(idx.shape, np.nan), rotational_motion))

        return positions if return_nparray else [const.Position(*p) for p in positions]

    def calculate_equivalent_radius(self) -> Float:
        """Return the equivalent radius of the star.

        The equivalent radius is the radius of a sphere with the same volume as
        the given component.

        :returns: Equivalent radius.
        :rtype: elisa.types.Float
        """
        volume = sys_utils.calculate_volume(self)
        return utils.calculate_equiv_radius(volume)

    def compute_lightcurve(self, **kwargs: Any) -> dict[str, NDArray[Float]]:
        """Resolve and compute the system light curve.

        Depending on the basic system properties, the appropriate light-curve
        generator function is selected.

        :param kwargs: Arguments passed to light-curve generator functions.
            Common keys include ``passband``, ``left_bandwidth``,
            ``right_bandwidth``, ``phases``, and ``position_method``.
        :type kwargs: object
        :returns: Mapping ``{passband: flux_array}``.
        :rtype: dict[str, NDArray[elisa.types.Float]]
        """
        fn_arr = (
            self._compute_light_curve_without_pulsations,
            self._compute_light_curve_with_pulsations,
        )
        curve_fn = c_router.resolve_curve_method(self, fn_arr)
        return curve_fn(**kwargs)

    def _compute_light_curve_with_pulsations(self, **kwargs: Any) -> dict[str, NDArray[Float]]:
        r"""Compute light curve including pulsations.

        This is a thin wrapper that delegates the work to
        :func:`elisa.single_system.curves.lc.compute_light_curve_with_pulsations`.

        :param kwargs: Forwarded keyword arguments (see public ``compute_lightcurve``).
        :type kwargs: typing.Any
        :returns: Mapping from passband name to flux array.
        :rtype: dict[str, numpy.typing.NDArray[elisa.types.Float]]
        """
        return lc.compute_light_curve_with_pulsations(self, **kwargs)

    def _compute_light_curve_without_pulsations(self, **kwargs: Any) -> dict[str, NDArray[Float]]:
        r"""Compute light curve without pulsations.

        Delegates to
        :func:`elisa.single_system.curves.lc.compute_light_curve_without_pulsations`.

        :param kwargs: Forwarded keyword arguments (see public ``compute_lightcurve``).
        :type kwargs: typing.Any
        :returns: Mapping from passband name to flux array.
        :rtype: dict[str, numpy.typing.NDArray[elisa.types.Float]]
        """
        return lc.compute_light_curve_without_pulsations(self, **kwargs)

    def compute_rv(self, **kwargs: Any) -> dict[str, NDArray[Float]] | None:
        r"""Resolve and compute the radial-velocity curve.

        The generator depends on the user-defined method.

        :param kwargs: Radial-velocity options. Supported keys include
            ``method``, ``position_method``, and ``phases``.
        :type kwargs: object
        :returns: Radial-velocity mapping such as ``{"star": values}`` or
            ``None`` when the requested ``method`` is not recognised.
        :rtype: dict[str, numpy.typing.NDArray[elisa.types.Float]] | None
        """
        if kwargs["method"] == "kinematic":
            return rv.com_radial_velocity(self, **kwargs)

        if kwargs["method"] == "radiometric":
            fn_arr = (
                self._compute_rv_curve_without_pulsations,
                self._compute_rv_curve_with_pulsations,
            )
            curve_fn = c_router.resolve_curve_method(self, fn_arr)

            kwargs = rv_utils.include_passband_data_to_kwargs(**kwargs)
            return curve_fn(**kwargs)

        return None

    def _compute_rv_curve_with_pulsations(self, **kwargs: Any) -> dict[str, NDArray[Float]]:
        r"""Compute radial-velocity (RV) curve including pulsations.

        Delegates to :func:`elisa.single_system.curves.rv.compute_rv_curve_with_pulsations`.

        :param kwargs: Forwarded keyword arguments (see public ``compute_rv``).
        :type kwargs: typing.Any
        :returns: Mapping of component name (e.g. ``"star"``) to RV array.
        :rtype: dict[str, numpy.typing.NDArray[elisa.types.Float]]
        """
        return rv.compute_rv_curve_with_pulsations(self, **kwargs)

    def _compute_rv_curve_without_pulsations(self, **kwargs: Any) -> dict[str, NDArray[Float]]:
        r"""Compute radial-velocity (RV) curve without pulsations.

        Delegates to :func:`elisa.single_system.curves.rv.compute_rv_curve_without_pulsations`.

        :param kwargs: Forwarded keyword arguments (see public ``compute_rv``).
        :type kwargs: typing.Any
        :returns: Mapping of component name to RV array.
        :rtype: dict[str, numpy.typing.NDArray[elisa.types.Float]]
        """
        return rv.compute_rv_curve_without_pulsations(self, **kwargs)


