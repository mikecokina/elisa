from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from elisa import const
from elisa import umpy as up
from elisa import units as u
from elisa.base.types import FLOAT
from elisa.binary_system.orbit import orbit
from elisa.binary_system.transform import RadialVelocityObserverProperties
from elisa.logger import getLogger

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from elisa.types import Float

logger = getLogger("binary_system.curves.community")


class RadialVelocitySystem:
    """Compute binary component centre-of-mass radial velocities.

    This standalone class computes radial velocities of binary components using
    astro-community parameters.

    Community parameters::

        ``asini`` -- semi-major axis multiplied by ``sin(inclination)``
        ``eccentricity``
        ``argument_of_periastron``
        ``period``
        ``true_anomaly``
        ``gamma``
        ``mass_ratio``

    Input initialization parameters are transformed and validated using
    :class:`RadialVelocityObserverProperties`.

    :param eccentricity: Orbital eccentricity.
    :type eccentricity: Float
    :param argument_of_periastron: Argument of periastron.
    :type argument_of_periastron: Float
    :param period: Orbital period of the binary star system. If a unit is not
        specified, the default period unit is assumed.
    :type period: Float
    :param mass_ratio: Binary mass ratio.
    :type mass_ratio: Float
    :param asini: Projected semi-major axis.
    :type asini: Float
    :param gamma: Systemic radial velocity.
    :type gamma: Float
    """

    inclination: ClassVar[Float] = const.HALF_PI
    phase_shift: ClassVar[Float] = FLOAT(0.0)

    def __init__(
        self,
        eccentricity: Float,
        argument_of_periastron: Float,
        period: Float,
        mass_ratio: Float,
        asini: Float,
        gamma: Float,
    ) -> None:
        """Initialize the radial-velocity system.

        :param eccentricity: Orbital eccentricity.
        :type eccentricity: Float
        :param argument_of_periastron: Argument of periastron.
        :type argument_of_periastron: Float
        :param period: Orbital period.
        :type period: Float
        :param mass_ratio: Binary mass ratio.
        :type mass_ratio: Float
        :param asini: Projected semi-major axis.
        :type asini: Float
        :param gamma: Systemic radial velocity.
        :type gamma: Float
        :return: ``None``.
        :rtype: None
        """
        kwargs = self.transform_input(
            eccentricity=eccentricity,
            argument_of_periastron=argument_of_periastron,
            period=period,
            mass_ratio=mass_ratio,
            asini=asini,
            gamma=gamma,
        )

        self.eccentricity: Float = np.nan
        self.argument_of_periastron: Float = np.nan
        self.period: Float = np.nan
        self.mass_ratio: Float = np.nan
        self.asini: Float = np.nan
        self.gamma: Float = np.nan
        self.orbit: orbit.Orbit | None = None
        self.rv_unit = u.dimensionless_unscaled

        self.init_properties(**kwargs)
        self.init_orbit()

    def init_properties(self, **kwargs: Any) -> None:
        """Set instance properties from validated input.

        :param kwargs: Validated input properties.
        :type kwargs: Float
        :return: ``None``.
        :rtype: None
        """
        for key, val in kwargs.items():
            setattr(self, key, val)

    @staticmethod
    def transform_input(**kwargs: Any) -> dict[str, Float]:
        """Transform and validate input keyword arguments.

        :param kwargs: Model parameters.
        :type kwargs: Float
        :return: Transformed and validated parameters.
        :rtype: dict[str, Float]
        """
        return RadialVelocityObserverProperties.transform_input(**kwargs)

    def init_orbit(self) -> None:
        """Initialize the orbit helper for the binary-like system.

        :return: ``None``.
        :rtype: None
        """
        logger.debug(
            "re/initializing orbit in class instance %s",
            self.__class__.__name__,
        )
        orbit_kwargs = {key: getattr(self, key) for key in orbit.Orbit.ALL_KWARGS}
        self.orbit = orbit.Orbit(**orbit_kwargs)

    def radial_velocity(self, **kwargs: Any) -> dict[str, NDArray[Float]]:
        """Produce synthetic radial-velocity curves in community format.

        The calculation uses the ``asini`` and ``q`` parameterization.

        :param kwargs: Additional keyword arguments.

            Supported options include:

            - ``phases`` - photometric phases used to calculate synthetic
              radial velocities
            - ``position_method`` - callable producing an array of orbital
              positions and orientations at given photometric phases
        :type kwargs: Any
        :return: Radial-velocity curves for primary and secondary components.
        :rtype: dict[str, NDArray[Float]]
        """
        phases = kwargs.pop("phases")
        position_method = kwargs.pop("position_method")
        orbital_motion = position_method(phase=phases)

        sma_primary, sma_secondary = self.distance_to_center_of_mass(
            self.mass_ratio,
            FLOAT(1.0),
        )

        period = np.float64((self.period * u.DEFAULT_PERIOD_UNIT).to(u.s))
        asini = np.float64((self.asini * u.solRad).to(u.m))

        sma_primary *= asini
        sma_secondary *= asini

        primary_rv = (
            self._radial_velocity(
                sma_primary,
                self.eccentricity,
                self.argument_of_periastron,
                period,
                orbital_motion[:, 2],
            )
            * -1.0
        )

        secondary_rv = self._radial_velocity(
            sma_secondary,
            self.eccentricity,
            self.argument_of_periastron,
            period,
            orbital_motion[:, 2],
        )

        return {
            "primary": primary_rv + self.gamma,
            "secondary": secondary_rv + self.gamma,
        }

    @staticmethod
    def distance_to_center_of_mass(
        q: Float,
        distance: Float,
    ) -> tuple[Float, Float]:
        """Return distances of components from the barycentre.

        :param q: Mass ratio.
        :type q: Float
        :param distance: Component separation.
        :type distance: Float
        :return: Distance of the primary and secondary component from the
            barycentre.
        :rtype: tuple[Float, Float]
        """
        com_from_primary = (q * distance) / (1.0 + q)
        return com_from_primary, distance - com_from_primary

    @staticmethod
    def _radial_velocity(
        asini: Float,
        eccentricity: Float,
        argument_of_periastron: Float,
        period: Float,
        true_anomaly: NDArray,
    ) -> NDArray[Float]:
        """Compute radial velocity for the given parameters.

        :param asini: Projected semi-major axis.
        :type asini: Float
        :param eccentricity: Orbital eccentricity.
        :type eccentricity: Float
        :param argument_of_periastron: Argument of periastron.
        :type argument_of_periastron: Float
        :param period: Orbital period.
        :type period: Float
        :param true_anomaly: True anomaly value or values.
        :type true_anomaly: NDArray
        :return: Radial velocity values.
        :rtype: NDArray[Float]
        """
        true_anomaly_array = np.asarray(true_anomaly, dtype=np.float64)
        a_term = 2.0 * up.pi * asini
        b_term = period * up.sqrt(1.0 - up.power(eccentricity, 2))
        c_term = up.cos(true_anomaly_array + argument_of_periastron) + (eccentricity * up.cos(argument_of_periastron))
        return -a_term * c_term / b_term

    def get_positions_method(self) -> Callable:
        """Return the orbital-motion method.

        :return: Bound orbital-motion method.
        :rtype: Callable
        """
        return self.orbit.orbital_motion

    def compute_rv(self, **kwargs: Any) -> dict[str, NDArray[Float]]:
        """Produce synthetic radial-velocity curves in community format.

        :param kwargs: Additional keyword arguments.

            Supported options include:

            - ``phases`` - photometric phases used to calculate synthetic
              radial velocities
            - ``position_method`` - callable producing an array of orbital
              positions and orientations at given photometric phases
        :type kwargs: Any
        :return: Radial-velocity curves for primary and secondary components.
        :rtype: dict[str, NDArray[Float]]
        """
        return self.radial_velocity(**kwargs)

    @staticmethod
    def prepare_json(data: dict[str, Float]) -> dict[str, Float]:
        """Filter parameters required to initialize ``RadialVelocitySystem``.

        :param data: Mapping that can contain items not valid for initialization
            of :class:`RadialVelocitySystem`.
        :type data: dict[str, Float]
        :return: Arguments necessary to initialize
            :class:`RadialVelocitySystem`.
        :rtype: dict[str, Float]
        """
        return {
            "eccentricity": data["eccentricity"],
            "argument_of_periastron": data["argument_of_periastron"],
            "period": data["period"],
            "mass_ratio": data["mass_ratio"],
            "asini": data["asini"],
            "gamma": data["gamma"],
        }
