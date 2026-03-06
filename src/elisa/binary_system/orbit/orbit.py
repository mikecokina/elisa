from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import scipy.optimize as _sci_opt

from elisa import const, utils
from elisa import umpy as up
from elisa.base.orbit.orbit import AbstractOrbit
from elisa.base.types import FLOAT, INT
from elisa.binary_system.orbit.transform import OrbitProperties
from elisa.logger import getLogger

logger = getLogger("binary_system.orbit.orbit")

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from elisa.base.container import StarContainer
    from elisa.binary_system.container import OrbitalPositionContainer
    from elisa.types import Float


def angular_velocity(
        period: Float | ArrayLike,
        eccentricity: Float | ArrayLike,
        distance: Float | ArrayLike,
) -> NDArray[np.floating] | Float:
    """Compute angular velocity for a given component distance.

    The formula is derived from the relation between orbital period, ellipse
    geometry and angular motion::

        w = dp/dt

        P * 1/2 * dp/dt = pi * a * b

        e = sqrt(1 - (b/a)^2)

    :param period: Orbital period in days.
    :type period: elisa.types.Float | numpy.typing.ArrayLike
    :param eccentricity: Orbital eccentricity.
    :type eccentricity: elisa.types.Float | numpy.typing.ArrayLike
    :param distance: Radial distance (in semi-major axis units).
    :type distance: elisa.types.Float | numpy.typing.ArrayLike
    :returns: Angular velocity in radians per second.
    :rtype: numpy.ndarray | elisa.types.Float
    """
    return ((2.0 * up.pi) / (period * 86400.0 * (distance ** 2))) * up.sqrt(
        (1.0 - eccentricity) * (1.0 + eccentricity),
    )


def primary_orbital_speed(
        m1: Float,
        m2: Float,
        a_red: Float,
        components_distance: Float,
) -> Float:
    """Return orbital speed of the primary component about the system centre.

    :param m1: Primary mass.
    :type m1: elisa.types.Float
    :param m2: Secondary mass.
    :type m2: elisa.types.Float
    :param a_red: Reduced semi-major axis of the primary.
    :type a_red: elisa.types.Float
    :param components_distance: Distance between components (same units as a_red).
    :type components_distance: elisa.types.Float
    :returns: Orbital speed of the primary component.
    :rtype: elisa.types.Float
    """
    m = m1 + m2
    return m2 * np.sqrt((const.G / m) * ((2 / components_distance) - (m2 / (a_red * m))))


def velocity_vector_angle(
        eccentricity: Float | ArrayLike,
        true_anomaly: Float | ArrayLike,
) -> tuple[NDArray[np.floating] | Float, NDArray[np.floating] | Float]:
    """Return sine and cosine of angle between velocity vector and join vector.

    :param eccentricity: Orbital eccentricity.
    :type eccentricity: elisa.types.Float | numpy.typing.ArrayLike
    :param true_anomaly: True anomaly (radians).
    :type true_anomaly: elisa.types.Float | numpy.typing.ArrayLike
    :returns: Tuple of (sin, cos) of the angle.
    :rtype: tuple[numpy.ndarray | elisa.types.Float, numpy.ndarray | elisa.types.Float]
    """
    den = up.sqrt(1 + eccentricity ** 2 + 2 * eccentricity * up.cos(true_anomaly))
    sin = (1 + eccentricity * up.cos(true_anomaly)) / den
    cos = - (eccentricity * up.sin(true_anomaly)) / den
    return sin, cos


def create_orb_vel_vectors(
        system: OrbitalPositionContainer,
        components_distance: Float,
) -> dict[str, NDArray[np.floating]]:
    """Return orbital velocity vectors for both components in centre-of-mass frame.

    :param system: OrbitalPositionContainer instance.
    :type system: elisa.binary_system.container.OrbitalPositionContainer
    :param components_distance: Distance between components in SMA units.
    :type components_distance: elisa.types.Float
    :returns: Mapping with "primary" and "secondary" velocity vectors.
    :rtype: dict[str, numpy.ndarray]
    """
    a_red = system.semi_major_axis * system.mass_ratio / (1 + system.mass_ratio)
    primary: StarContainer = system.primary
    secondary: StarContainer = system.secondary
    speed = primary_orbital_speed(
        primary.mass,
        secondary.mass,
        a_red,
        system.semi_major_axis * components_distance,
    )

    sin, cos = velocity_vector_angle(system.eccentricity, system.position.true_anomaly)

    velocity: dict[str, NDArray[np.floating]] = {
        "primary": np.array([cos * speed, -sin * speed, 0]),
    }
    velocity["secondary"] = -velocity["primary"] / system.mass_ratio

    return velocity


def distance_to_center_of_mass(
        primary_mass: Float,
        secondary_mass: Float,
        distance: Float,
) -> tuple[Float, Float]:
    """Return distances from primary and secondary components to center of mass.

    :param primary_mass: Primary mass.
    :type primary_mass: elisa.types.Float
    :param secondary_mass: Secondary mass.
    :type secondary_mass: elisa.types.Float
    :param distance: Separation between components.
    :type distance: elisa.types.Float
    :returns: Tuple (distance_from_primary, distance_from_secondary).
    :rtype: tuple[elisa.types.Float, elisa.types.Float]
    """
    mass = primary_mass + secondary_mass
    com_from_primary = (distance * secondary_mass) / mass
    return com_from_primary, distance - com_from_primary


def orbital_semi_major_axes(
        r: Float | ArrayLike,
        eccentricity: Float | ArrayLike,
        true_anomaly: Float | ArrayLike,
) -> NDArray[np.floating] | Float:
    """Return orbital semi-major axis from component distance and anomaly.

    :param r: Distance from center of mass to object.
    :type r: elisa.types.Float | numpy.typing.ArrayLike
    :param eccentricity: Orbital eccentricity.
    :type eccentricity: elisa.types.Float | numpy.typing.ArrayLike
    :param true_anomaly: True anomaly.
    :type true_anomaly: elisa.types.Float | numpy.typing.ArrayLike
    :returns: Semi-major axis (same shape as inputs).
    :rtype: numpy.ndarray | elisa.types.Float
    """
    return r * (1.0 + eccentricity * up.cos(true_anomaly)) / (1.0 - up.power(eccentricity, 2))


def component_distance_from_mean_anomaly(
        eccentricity: Float | ArrayLike,
        true_anomaly: Float | ArrayLike,
) -> NDArray[np.floating] | Float:
    """Return component distance from mean anomaly-related quantities.

    :param eccentricity: Orbital eccentricity.
    :type eccentricity: elisa.types.Float | numpy.typing.ArrayLike
    :param true_anomaly: True anomaly.
    :type true_anomaly: elisa.types.Float | numpy.typing.ArrayLike
    :returns: Component distance in SMA units.
    :rtype: numpy.ndarray | elisa.types.Float
    """
    return (1.0 - up.power(eccentricity, 2)) / (1.0 + eccentricity * up.cos(true_anomaly))


def get_approx_ecl_angular_width(
        forward_radius1: Float,
        forward_radius2: Float,
        components_distance: Float,
        inclination: Float,
) -> tuple[Float, Float]:
    """Return approximate angular half-widths of an eclipse for spherical components.

    :param forward_radius1: Apparent radius of component 1.
    :type forward_radius1: elisa.types.Float
    :param forward_radius2: Apparent radius of component 2.
    :type forward_radius2: elisa.types.Float
    :param components_distance: Distance between components (SMA units).
    :type components_distance: elisa.types.Float
    :param inclination: Orbital inclination in radians.
    :type inclination: elisa.types.Float
    :returns: Tuple of (outer_half_width, inner_half_width).
    :rtype: tuple[elisa.types.Float, elisa.types.Float]
    """
    # tilt of the orbital plane and z-axis in the observer reference frame
    tilt = np.abs(const.HALF_PI - inclination)
    # maximum apparent distance between components where eclipse is possible
    r_outer = forward_radius1 + forward_radius2
    r_inner = np.abs(forward_radius1 - forward_radius2)
    # closest apparent distances of component centres
    r_close = components_distance * np.sin(tilt)

    # checking if eclipses occur
    nu_outer = 0.0 if r_close >= r_outer else (
        np.arcsin(np.sqrt(np.power(r_outer / components_distance, 2) - np.power(np.sin(tilt), 2)))
    )
    nu_inner = 0.0 if r_close >= r_inner else (
        np.arcsin(np.sqrt(np.power(r_inner / components_distance, 2) - np.power(np.sin(tilt), 2)))
    )

    return nu_outer, nu_inner


class Orbit(AbstractOrbit):
    """Represent the orbit of a binary system.

    Parameters accepted via :class:`OrbitProperties.transform_input` are used to
    initialise instance attributes. See transform utilities for supported
    keyword names and conversions.
    """

    MANDATORY_KWARGS: ClassVar[tuple[str, ...]] = ("period", "inclination", "eccentricity", "argument_of_periastron")
    OPTIONAL_KWARGS: ClassVar[tuple[str, ...]] = ("phase_shift",)
    ALL_KWARGS: ClassVar[tuple[str, ...]] = MANDATORY_KWARGS + OPTIONAL_KWARGS

    def __init__(self, **kwargs: Any) -> None:
        """Initialize an Orbit instance.

        :param kwargs: Orbit parameters; mandatory and optional names are
            validated against :pyattr:`~Orbit.ALL_KWARGS`.
        :type kwargs: dict
        """
        utils.invalid_kwarg_checker(kwargs, list(Orbit.ALL_KWARGS), Orbit)
        utils.check_missing_kwargs(list(self.__class__.MANDATORY_KWARGS), kwargs, instance_of=self.__class__)
        kwargs = OrbitProperties.transform_input(**kwargs)

        super().__init__(**kwargs)

        # default values of properties
        self.period = np.nan
        self.eccentricity = np.nan
        self.argument_of_periastron = np.nan
        self.inclination = np.nan

        self.periastron_distance = np.nan
        self.periastron_phase = np.nan
        self.semi_major_axis = np.nan
        self.phase_shift = 0.0

        # values of properties
        logger.debug("setting properties of orbit")
        for kwarg in kwargs:
            setattr(self, kwarg, kwargs[kwarg])

        self.periastron_distance = self.compute_periastron_distance()
        self.conjunctions = self.get_conjuction()
        self.periastron_phase = -self.conjunctions["primary_eclipse"]["true_phase"] % 1

    @classmethod
    def phase_to_mean_anomaly(cls, phase: Float | ArrayLike) -> NDArray[np.floating] | Float:
        """Return mean anomaly corresponding to photometric phase.

        :param phase: Photometric phase(s).
        :type phase: elisa.types.Float | numpy.typing.ArrayLike
        :returns: Mean anomaly in radians (same shape as input).
        :rtype: numpy.ndarray | elisa.types.Float
        """
        return const.FULL_ARC * phase

    @classmethod
    def mean_anomaly_to_phase(cls, mean_anomaly: Float | ArrayLike) -> NDArray[np.floating] | Float:
        """Return photometric phase from mean anomaly.

        :param mean_anomaly: Mean anomaly in radians.
        :type mean_anomaly: elisa.types.Float | numpy.typing.ArrayLike
        :returns: Photometric phase(s).
        :rtype: numpy.ndarray | elisa.types.Float
        """
        return mean_anomaly / const.FULL_ARC

    def mean_anomaly_fn(self, eccentric_anomaly: Float, *args: Any) -> Float:
        """Kepler's equation residual used by solvers.

        :param eccentric_anomaly: Eccentric anomaly value for the residual.
        :param args: Additional arguments (mean_anomaly,).
        :returns: Residual of Kepler's equation.
        :rtype: elisa.types.Float
        """
        mean_anomaly, = args
        return eccentric_anomaly - self.eccentricity * up.sin(eccentric_anomaly) - mean_anomaly

    def mean_anomaly_to_eccentric_anomaly(self, mean_anomaly: Float) -> Float | bool:
        """Solve Kepler's equation for the eccentric anomaly.

        Returns the eccentric anomaly or ``False`` on failure.

        :param mean_anomaly: Mean anomaly in radians.
        :type mean_anomaly: elisa.types.Float
        :returns: Eccentric anomaly or ``False`` if solver fails.
        :rtype: elisa.types.Float | bool
        """
        try:
            solution = _sci_opt.newton(
                self.mean_anomaly_fn,
                1.0,
                args=(mean_anomaly,),
                tol=1e-10,
            )
            if not up.isnan(solution):
                if solution < 0:
                    solution += const.FULL_ARC
                return solution
        except Exception as err:  # noqa: BLE001
            logger.debug(
                "scipy.optimize.newton failed to provide solution for Orbit.mean_anomaly_to_eccentric_anomaly: %s",
                err,
            )
            return False
        else:
            return False

    def eccentric_anomaly_to_mean_anomaly(self, eccentric_anomaly: Float | ArrayLike) -> NDArray[np.floating] | Float:
        """Return mean anomaly from eccentric anomaly using Kepler's equation.

        :param eccentric_anomaly: Eccentric anomaly value(s).
        :type eccentric_anomaly: elisa.types.Float | numpy.typing.ArrayLike
        :returns: Mean anomaly in radians.
        :rtype: numpy.ndarray | elisa.types.Float
        """
        return (eccentric_anomaly - self.eccentricity * up.sin(eccentric_anomaly)) % const.FULL_ARC

    def eccentric_anomaly_to_true_anomaly(self, eccentric_anomaly: Float | ArrayLike) -> NDArray[np.floating] | Float:
        """Return true anomaly from eccentric anomaly and eccentricity.

        :param eccentric_anomaly: Eccentric anomaly value(s).
        :type eccentric_anomaly: elisa.types.Float | numpy.typing.ArrayLike
        :returns: True anomaly in radians.
        :rtype: numpy.ndarray | elisa.types.Float
        """
        true_anomaly = 2.0 * up.arctan(
            up.sqrt((1.0 + self.eccentricity) / (1.0 - self.eccentricity)) * up.tan(eccentric_anomaly / 2.0),
        )
        true_anomaly[true_anomaly < 0] += const.FULL_ARC
        return true_anomaly

    def true_anomaly_to_eccentric_anomaly(self, true_anomaly: Float | ArrayLike) -> NDArray[np.floating] | Float:
        """Return eccentric anomaly from true anomaly and eccentricity.

        :param true_anomaly: True anomaly value(s).
        :type true_anomaly: elisa.types.Float | numpy.typing.ArrayLike
        :returns: Eccentric anomaly in radians.
        :rtype: numpy.ndarray | elisa.types.Float
        """
        eccentric_anomaly = 2.0 * up.arctan(
            up.sqrt((1.0 - self.eccentricity) / (1.0 + self.eccentricity)) * up.tan(true_anomaly / 2.0),
        )
        eccentric_anomaly[eccentric_anomaly < 0] += const.FULL_ARC
        return eccentric_anomaly

    def relative_radius(self, true_anomaly: Float | ArrayLike) -> NDArray[np.floating] | Float:
        """Return radius vector length of ellipse with a=1 for given true anomaly.

        :param true_anomaly: True anomaly value(s).
        :type true_anomaly: elisa.types.Float | numpy.typing.ArrayLike
        :returns: Radius vector (same shape as input).
        :rtype: numpy.ndarray | elisa.types.Float
        """
        return (1.0 - self.eccentricity ** 2) / (1.0 + self.eccentricity * up.cos(true_anomaly))

    def true_anomaly_to_azimuth(self, true_anomaly: Float | ArrayLike) -> NDArray[np.floating] | Float:
        """Convert true anomaly to azimuth measured from the y-axis.

        :param true_anomaly: True anomaly value(s).
        :type true_anomaly: elisa.types.Float | numpy.typing.ArrayLike
        :returns: Azimuth angle(s) in radians.
        :rtype: numpy.ndarray | elisa.types.Float
        """
        return (true_anomaly + self.argument_of_periastron) % const.FULL_ARC

    def azimuth_to_true_anomaly(self, azimuth: Float | ArrayLike) -> NDArray[np.floating] | Float:
        """Return true anomaly corresponding to an azimuth.

        :param azimuth: Azimuth angle(s) in radians.
        :type azimuth: elisa.types.Float | numpy.typing.ArrayLike
        :returns: True anomaly value(s).
        :rtype: numpy.ndarray | elisa.types.Float
        """
        return (azimuth - self.argument_of_periastron) % const.FULL_ARC

    def orbital_motion(self, phase: Float | ArrayLike) -> NDArray[np.floating] | Float:
        """Compute orbital motion for given photometric phase(s).

        Returns columns: (distance, azimuth, true_anomaly, phase).

        :param phase: Photometric phase(s) as scalar or array-like.
        :type phase: elisa.types.Float | numpy.typing.ArrayLike
        :returns: 2D array with per-row (r, az, nu, phs).
        :rtype: numpy.ndarray
        """
        # ability to accept scalar as input
        if isinstance(phase, (int, INT, float, FLOAT)):
            phase = np.array([FLOAT(phase)])
        # photometric phase to phase measured from periastron
        true_phase = self.true_phase(phase=phase, phase_shift=self.conjunctions["primary_eclipse"]["true_phase"])

        mean_anomaly = self.phase_to_mean_anomaly(phase=true_phase)
        eccentric_anomaly = np.array([self.mean_anomaly_to_eccentric_anomaly(mean_anomaly=xx)
                                      for xx in mean_anomaly])
        true_anomaly = self.eccentric_anomaly_to_true_anomaly(eccentric_anomaly=eccentric_anomaly)
        distance = self.relative_radius(true_anomaly=true_anomaly)
        azimut_angle = self.true_anomaly_to_azimuth(true_anomaly=true_anomaly)

        return np.column_stack((distance, azimut_angle, true_anomaly, phase))

    def orbital_motion_from_azimuths(self, azimuth: Float | ArrayLike) -> NDArray[np.floating] | Float:
        """Compute orbital motion when azimuth(s) are provided.

        Returns columns: (distance, azimuth, true_anomaly, phase).

        :param azimuth: Azimuth angle(s) as scalar or array-like.
        :type azimuth: elisa.types.Float | numpy.typing.ArrayLike
        :returns: 2D array with per-row (r, az, nu, phs).
        :rtype: numpy.ndarray
        """
        true_anomaly = self.azimuth_to_true_anomaly(azimuth)
        distance = self.relative_radius(true_anomaly=true_anomaly)
        eccentric_anomaly = self.true_anomaly_to_eccentric_anomaly(true_anomaly)
        mean_anomaly = self.eccentric_anomaly_to_mean_anomaly(eccentric_anomaly)
        true_phase = self.mean_anomaly_to_phase(mean_anomaly)
        phase = self.phase(true_phase, phase_shift=self.conjunctions["primary_eclipse"]["true_phase"])
        return np.column_stack((distance, azimuth, true_anomaly, phase))

    def get_conjuction(self) -> dict[str, dict[str, Float]]:
        """Compute photometric phases and anomalies of conjunctions (eclipses).

        The return dictionary has entries for "primary_eclipse" and
        "secondary_eclipse" each containing true_phase, true_anomaly,
        mean_anomaly and eccentric_anomaly.

        :returns: Mapping of eclipse types to their orbital quantities.
        :rtype: dict[str, dict]
        """
        # determining order of eclipses
        conjunction_arc_list: list[float] = []
        try:
            if 0 <= self.inclination <= const.PI / 2.0:
                conjunction_arc_list = [const.PI / 2.0, 3.0 * const.PI / 2.0]
            elif const.PI / 2.0 < self.inclination <= const.PI:
                conjunction_arc_list = [3.0 * const.PI / 2.0, const.PI / 2.0]
        except TypeError as err:
            msg = f"Invalid type of {self.__class__.__name__}.inclination - {err}."
            raise TypeError(msg) from err

        conjunction_quantities: dict[str, dict[str, Float]] = {}
        for alpha, idx in zip(conjunction_arc_list, ("primary_eclipse", "secondary_eclipse"), strict=True):
            # true anomaly of conjunction (measured from periastron counter-clokwise)
            true_anomaly_of_conjunction = self.azimuth_to_true_anomaly(alpha)  # \nu_{con}

            # eccentric anomaly of conjunction (measured from apse line)
            eccentric_anomaly_of_conjunction = (2.0 * up.arctan(
                up.sqrt((1.0 - self.eccentricity) / (1.0 + self.eccentricity))
                * up.tan(true_anomaly_of_conjunction / 2.0),
            )) % const.FULL_ARC

            # mean anomaly of conjunction (measured from apse line)
            mean_anomaly_of_conjunction = (eccentric_anomaly_of_conjunction - self.eccentricity
                                           * up.sin(eccentric_anomaly_of_conjunction)
                                           ) % const.FULL_ARC

            # true phase of conjunction (measured from apse line)
            true_phase_of_conjunction = (mean_anomaly_of_conjunction / const.FULL_ARC) % 1.0

            conjunction_quantities[idx] = {
                "true_anomaly": true_anomaly_of_conjunction,
                "eccentric_anomaly": eccentric_anomaly_of_conjunction,
                "mean_anomaly": mean_anomaly_of_conjunction,
                "true_phase": true_phase_of_conjunction,
            }

        return conjunction_quantities

    def compute_periastron_distance(self) -> Float:
        """Calculate relative periastron distance in SMA units.

        :returns: Periastron distance.
        :rtype: elisa.types.Float
        """
        periastron_distance = self.relative_radius(true_anomaly=np.array([0])[0])
        logger.debug(
            "setting property periastron_distance of class instance %s to %s",
            self.__class__.__name__,
            periastron_distance,
        )
        return periastron_distance
