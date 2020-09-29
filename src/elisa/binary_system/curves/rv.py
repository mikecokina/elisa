import numpy as np

from . import rv_point
from . import c_router
from .. orbit.orbit import distance_to_center_of_mass
from ... observer.passband import init_rv_passband
from ... import settings
from ... import (
    umpy as up,
    units as u
)


def _radial_velocity(semi_major_axis, inclination, eccentricity, argument_of_periastron, period, true_anomaly):
    """
    Compute radial velocity for given paramters.

    :param semi_major_axis: float
    :param inclination: float
    :param eccentricity: float
    :param argument_of_periastron: float
    :param true_anomaly: float or numpy.array
    :param period: float
    :return: Union[float, numpy.array]
    """
    a = 2.0 * up.pi * semi_major_axis * up.sin(inclination)
    b = period * up.sqrt(1.0 - up.power(eccentricity, 2))
    c = up.cos(true_anomaly + argument_of_periastron) + (eccentricity * up.cos(argument_of_periastron))
    return - a * c / b


def com_radial_velocity(binary, **kwargs):
    """
    Calculates radial velocity curves of the `binary` system using radial velocities of centres of masses.

    :param binary: elisa.binary_system.system.BinarySystem; binary system instance
    :param kwargs: dict;
    :**kwargs options**:
        * **position_method** * -- function that is used to calculate orbital motion
        * **phases** * -- phases in which to calculate
    :return: Tuple
    """
    position_method = kwargs.pop("position_method")
    phases = kwargs.pop("phases")
    orbital_motion = position_method(input_argument=phases, return_nparray=True, calculate_from='phase')

    sma_primary, sma_secondary = distance_to_center_of_mass(binary.primary.mass, binary.secondary.mass, 1.0)

    # in base SI units
    sma_primary *= binary.semi_major_axis
    sma_secondary *= binary.semi_major_axis
    period = np.float64((binary.period * u.PERIOD_UNIT).to(u.s))

    rv_primary = _radial_velocity(sma_primary, binary.inclination, binary.eccentricity,
                                  binary.argument_of_periastron, period, orbital_motion[:, 3]) * -1.0

    rv_secondary = _radial_velocity(sma_secondary, binary.inclination, binary.eccentricity,
                                    binary.argument_of_periastron, period, orbital_motion[:, 3])

    rvs = {'primary': rv_primary + binary.gamma, 'secondary': rv_secondary + binary.gamma}
    return rvs


def include_passband_data_to_kwargs(**kwargs):
    """
    Including dummy passband from which radiometric radial velocities will be calculated.

    :param kwargs: Tuple;
    :return: Tuple;
    """
    psbnd, right_bandwidth, left_bandwidth = init_rv_passband()
    kwargs.update({'passband': {'rv_band': psbnd},
                   'left_bandwidth': left_bandwidth,
                   'right_bandwidth': right_bandwidth,
                   'atlas': settings.ATM_ATLAS})
    return kwargs


def compute_circular_synchronous_rv_curve(binary, **kwargs):
    """
    Compute radial velocity curve for synchronous circular binary system.

    :param binary: elisa.binary_system.system.BinarySystem;
    :param kwargs: Dict;
            * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
            * ** left_bandwidth ** * - float
            * ** right_bandwidth ** * - float
            * ** atlas ** * - str
            * ** position_method** * - function definition; to evaluate orbital positions
            * ** phases ** * - numpy.array
    :return: Dict[str, numpy.array];
    """
    initial_system = c_router.prep_initial_system(binary)
    rv_labels = list(settings.BINARY_COUNTERPARTS.keys())
    return c_router.produce_circular_sync_curves(binary, initial_system, kwargs.pop("phases"),
                                                 rv_point.compute_rv_at_pos, rv_labels, **kwargs)


def compute_circular_spotty_asynchronous_rv_curve(binary, **kwargs):
    """
    Function returns rv curve of asynchronous systems with circular orbits and spots.

    :param binary: elisa.binary_system.system.BinarySystem;
    :param kwargs: Dict;
    :**kwargs options**:
        * ** passband ** - Dict[str, elisa.observer.PassbandContainer]
        * ** left_bandwidth ** - float
        * ** right_bandwidth ** - float
        * ** atlas ** - str
    :return: Dict; rv for each component
    """
    rv_labels = list(settings.BINARY_COUNTERPARTS.keys())
    return c_router.produce_circular_spotty_async_curves(binary, rv_point.compute_rv_at_pos, rv_labels, **kwargs)


def compute_eccentric_rv_curve_no_spots(binary, **kwargs):
    """
    General function for generating rv curves of binaries with eccentric orbit and no spots.

    :param binary: elisa.binary_system.system.BinarySystem;
    :param kwargs: Dict;
    :**kwargs options**:
        * ** passband ** - Dict[str, elisa.observer.PassbandContainer]
        * ** left_bandwidth ** - float
        * ** right_bandwidth ** - float
        * ** atlas ** - str
    :return: Dict; rv for each component
    """
    rv_labels = list(settings.BINARY_COUNTERPARTS.keys())
    return c_router.produce_ecc_curves_no_spots(binary, rv_point.compute_rv_at_pos, rv_labels, **kwargs)


def compute_eccentric_spotty_rv_curve(binary, **kwargs):
    """
    General function for generating rv curves of binaries with eccentric orbit and spots.

    :param binary: elisa.binary_system.system.BinarySystem;
    :param kwargs: Dict;
    :**kwargs options**:
        * ** passband ** - Dict[str, elisa.observer.PassbandContainer]
        * ** left_bandwidth ** - float
        * ** right_bandwidth ** - float
        * ** atlas ** - str
    :return: Dict; rv for each component
    """
    rv_labels = list(settings.BINARY_COUNTERPARTS.keys())
    return c_router.produce_ecc_curves_with_spots(binary, rv_point.compute_rv_at_pos, rv_labels, **kwargs)
