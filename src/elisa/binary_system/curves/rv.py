from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from elisa import settings
from elisa import umpy as up
from elisa import units as u
from elisa.base.curves import rv_point
from elisa.binary_system.curves import c_router
from elisa.binary_system.orbit.orbit import distance_to_center_of_mass

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.binary_system.system import BinarySystem
    from elisa.types import Float


def _radial_velocity(
    semi_major_axis: Float,
    inclination: Float,
    eccentricity: Float,
    argument_of_periastron: Float,
    period: Float,
    true_anomaly: NDArray,
) -> NDArray[Float]:
    """Compute radial velocity for the given parameters.

    :param semi_major_axis: Semi-major axis.
    :type semi_major_axis: Float
    :param inclination: Orbital inclination.
    :type inclination: Float
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
    a_term = 2.0 * up.pi * semi_major_axis * up.sin(inclination)
    b_term = period * up.sqrt(1.0 - up.power(eccentricity, 2))
    c_term = up.cos(true_anomaly_array + argument_of_periastron) + (
        eccentricity * up.cos(argument_of_periastron)
    )
    return -a_term * c_term / b_term


def kinematic_radial_velocity(
    binary: BinarySystem,
    **kwargs: Any,
) -> dict[str, NDArray[Float]]:
    """Calculate radial-velocity curves from component centres of mass.

    :param binary: Binary system instance.
    :type binary: BinarySystem
    :param kwargs: Additional keyword arguments.

        Supported options include:

        - ``position_method`` - function used to calculate orbital motion
        - ``phases`` - phases at which to calculate
    :type kwargs: Any
    :return: Radial velocity values for each component. Indices correspond to
        indices of the input phases.
    :rtype: dict[str, NDArray[Float]]
    """
    position_method = kwargs.pop("position_method")
    phases = kwargs.pop("phases")
    orbital_motion = position_method(
        input_argument=phases,
        return_nparray=True,
        calculate_from="phase",
    )

    sma_primary, sma_secondary = distance_to_center_of_mass(
        binary.primary.mass,
        binary.secondary.mass,
        1.0,
    )

    # in base SI units
    sma_primary *= binary.semi_major_axis
    sma_secondary *= binary.semi_major_axis
    period = np.float64(
        (binary.period * u.DefaultBinarySystemUnits.system.period).to(u.TIME_UNIT),
    )

    rv_primary = _radial_velocity(
        sma_primary,
        binary.inclination,
        binary.eccentricity,
        binary.argument_of_periastron,
        period,
        orbital_motion[:, 3],
    ) * -1.0

    rv_secondary = _radial_velocity(
        sma_secondary,
        binary.inclination,
        binary.eccentricity,
        binary.argument_of_periastron,
        period,
        orbital_motion[:, 3],
    )

    return {
        "primary": rv_primary + binary.gamma,
        "secondary": rv_secondary + binary.gamma,
    }


def compute_circular_synchronous_rv_curve(
    binary: BinarySystem,
    **kwargs: Any,
) -> dict[str, NDArray[Float]]:
    """Compute radial-velocity curve for a synchronous circular binary system.

    :param binary: Binary system instance.
    :type binary: BinarySystem
    :param kwargs: Additional keyword arguments.

        Supported options include:

        - ``passband`` - ``dict[str, elisa.observer.PassbandContainer]``
        - ``left_bandwidth`` - ``Float``
        - ``right_bandwidth`` - ``Float``
        - ``position_method`` - function definition to evaluate orbital positions
        - ``phases`` - ``numpy.array``
    :type kwargs: Any
    :return: Radial-velocity curves for each component.
    :rtype: dict[str, NDArray[Float]]
    """
    initial_system = c_router.prep_initial_system(binary)
    rv_labels = list(settings.BINARY_COUNTERPARTS.keys())
    args = (
        binary,
        initial_system,
        kwargs.pop("phases"),
        rv_point.compute_rv_at_pos,
        rv_labels,
    )
    return c_router.produce_circular_sync_curves(*args, **kwargs)


def compute_circular_spotty_asynchronous_rv_curve(
    binary: BinarySystem,
    **kwargs: Any,
) -> dict[str, NDArray[Float]]:
    """Return RV curve of asynchronous systems with circular orbits and spots.

    :param binary: Binary system instance.
    :type binary: BinarySystem
    :param kwargs: Additional keyword arguments.

        Supported options include:

        - ``passband`` - ``dict[str, elisa.observer.PassbandContainer]``
        - ``left_bandwidth`` - ``Float``
        - ``right_bandwidth`` - ``Float``
        - ``atlas`` - ``str``
    :type kwargs: Any
    :return: Radial-velocity curves for each component.
    :rtype: dict[str, NDArray[Float]]
    """
    rv_labels = list(settings.BINARY_COUNTERPARTS.keys())
    return c_router.produce_circular_spotty_async_curves(
        binary,
        rv_point.compute_rv_at_pos,
        rv_labels,
        **kwargs,
    )


def compute_circular_pulsating_rv_curve(
    binary: BinarySystem,
    **kwargs: Any,
) -> dict[str, NDArray[Float]]:
    """Return RV curve of pulsating systems with circular orbits.

    :param binary: Binary system instance.
    :type binary: BinarySystem
    :param kwargs: Additional keyword arguments.

        Supported options include:

        - ``passband`` - ``dict[str, elisa.observer.PassbandContainer]``
        - ``left_bandwidth`` - ``Float``
        - ``right_bandwidth`` - ``Float``
        - ``atlas`` - ``str``
        - ``phases`` - ``numpy.array``
    :type kwargs: Any
    :return: Radial-velocity curves for each component.
    :rtype: dict[str, NDArray[Float]]
    """
    initial_system = c_router.prep_initial_system(
        binary,
        build_pulsations=False,
    )
    rv_labels = list(settings.BINARY_COUNTERPARTS.keys())
    args = (
        binary,
        initial_system,
        kwargs.pop("phases"),
        rv_point.compute_rv_at_pos,
        rv_labels,
    )
    return c_router.produce_circular_pulsating_curves(*args, **kwargs)


def compute_eccentric_rv_curve_no_spots(
    binary: BinarySystem,
    **kwargs: Any,
) -> dict[str, NDArray[Float]]:
    """Generate RV curves of binaries with eccentric orbit and no spots.

    :param binary: Binary system instance.
    :type binary: BinarySystem
    :param kwargs: Additional keyword arguments.

        Supported options include:

        - ``passband`` - ``dict[str, elisa.observer.PassbandContainer]``
        - ``left_bandwidth`` - ``Float``
        - ``right_bandwidth`` - ``Float``
        - ``atlas`` - ``str``
    :type kwargs: Any
    :return: Radial-velocity curves for each component.
    :rtype: dict[str, NDArray[Float]]
    """
    rv_labels = list(settings.BINARY_COUNTERPARTS.keys())
    return c_router.produce_ecc_curves_no_spots(
        binary,
        rv_point.compute_rv_at_pos,
        rv_labels,
        **kwargs,
    )


def compute_eccentric_spotty_rv_curve(
    binary: BinarySystem,
    **kwargs: Any,
) -> dict[str, NDArray[Float]]:
    """Generate RV curves of binaries with eccentric orbit and spots.

    :param binary: Binary system instance.
    :type binary: BinarySystem
    :param kwargs: Additional keyword arguments.

        Supported options include:

        - ``passband`` - ``dict[str, elisa.observer.PassbandContainer]``
        - ``left_bandwidth`` - ``Float``
        - ``right_bandwidth`` - ``Float``
        - ``atlas`` - ``str``
    :type kwargs: Any
    :return: Radial-velocity curves for each component.
    :rtype: dict[str, NDArray[Float]]
    """
    rv_labels = list(settings.BINARY_COUNTERPARTS.keys())
    return c_router.produce_ecc_curves_with_spots(
        binary,
        rv_point.compute_rv_at_pos,
        rv_labels,
        **kwargs,
    )
