import numpy as np
from elisa import (
    umpy as up,
    units,
    const
)
from elisa.binary_system.container import OrbitalPositionContainer
from elisa.binary_system import dynamic
from elisa.binary_system.curves import rvmp
from ...binary_system.curves import curves
from elisa.observer.passband import init_rv_passband
from elisa.conf import config
from elisa.observer.mp import manage_observations


def distance_to_center_of_mass(primary_mass, secondary_mass, positions):
    """
    Return distance from primary and from secondary component to center of mass.

    :param primary_mass: float
    :param secondary_mass: float
    :param positions: numpy.array
    :return: Tuple
    """
    distance = positions[:, 1]
    mass = primary_mass + secondary_mass
    com_from_primary = (distance * secondary_mass) / mass
    return com_from_primary, distance - com_from_primary


def orbital_semi_major_axes(r, eccentricity, true_anomaly):
    """
    Return orbital semi major axis for given parameter.

    :param r: float or numpy.array; distane from center of mass to object
    :param eccentricity: float or numpy.array; orbital eccentricity
    :param true_anomaly: float or numpy.array; true anomaly of orbital motion
    :return: Union[float, numpy.array]
    """
    return r * (1.0 + eccentricity * up.cos(true_anomaly)) / (1.0 - up.power(eccentricity, 2))


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
    r1, r2 = distance_to_center_of_mass(binary.primary.mass, binary.secondary.mass, orbital_motion)

    sma_primary = orbital_semi_major_axes(r1[-1], binary.orbit.eccentricity, orbital_motion[:, 3][-1])
    sma_secondary = orbital_semi_major_axes(r2[-1], binary.orbit.eccentricity, orbital_motion[:, 3][-1])

    # in base SI units
    sma_primary *= binary.semi_major_axis
    sma_secondary *= binary.semi_major_axis
    period = np.float64((binary.period * units.PERIOD_UNIT).to(units.s))

    rv_primary = _radial_velocity(sma_primary, binary.inclination, binary.eccentricity,
                                  binary.argument_of_periastron, period, orbital_motion[:, 3]) * -1.0

    rv_secondary = _radial_velocity(sma_secondary, binary.inclination, binary.eccentricity,
                                    binary.argument_of_periastron, period, orbital_motion[:, 3])

    rvs = {'primary': rv_primary + binary.gamma, 'secondary': rv_secondary + binary.gamma}
    return rvs


def include_passband_data_to_kwargs(**kwargs):
    """
    Including dummy passband from which radiometric radial velocities will be calculated.

    :param kwargs: tuple;
    :return: tuple;
    """
    psbnd, right_bandwidth, left_bandwidth = init_rv_passband()
    kwargs.update({'passband': {'rv_band': psbnd},
                   'left_bandwidth': left_bandwidth,
                   'right_bandwidth': right_bandwidth,
                   'atlas': config.ATM_ATLAS})
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
    initial_system = curves.prep_initial_system(binary)
    rv_labels = list(config.BINARY_COUNTERPARTS.keys())

    return curves.produce_circ_sync_curves(binary, initial_system, kwargs.pop("phases"),
                                           rvmp.compute_rv_at_pos, rv_labels, **kwargs)


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
    rv_labels = list(config.BINARY_COUNTERPARTS.keys())

    return curves.produce_circ_spotty_async_curves(binary, rvmp.compute_rv_at_pos, rv_labels, **kwargs)


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
    rv_labels = list(config.BINARY_COUNTERPARTS.keys())

    return curves.produce_ecc_curves_no_spots(binary, rvmp.compute_rv_at_pos, rv_labels, **kwargs)


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
    rv_labels = list(config.BINARY_COUNTERPARTS.keys())

    return curves.produce_ecc_curves_with_spots(binary, rvmp.compute_rv_at_pos, rv_labels, **kwargs)



