import numpy as np

from astropy import units as u
from elisa import units


def distance_to_center_of_mass(primary_mass, secondary_mass, positions):
    """
    Return distance from primary and from secondary component to center of mass.

    :param primary_mass: float
    :param secondary_mass: float
    :param positions: numpy.array
    :return:
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
    :return: float or numpy.array
    """
    return r * (1.0 + eccentricity * np.cos(true_anomaly)) / (1.0 - np.power(eccentricity, 2))


def _radial_velocity(semi_major_axis, inclination, eccentricity, argument_of_periastron, period, true_anomaly):
    """
    Compute radial velocity for given paramters.

    :param semi_major_axis: float
    :param inclination: float
    :param eccentricity: float
    :param argument_of_periastron: float
    :param true_anomaly: float or numpy.array
    :param period: float
    :return: float or numpy.array
    """
    a = 2.0 * np.pi * semi_major_axis * np.sin(inclination)
    b = period * np.sqrt(1.0 - np.power(eccentricity, 2))
    c = np.cos(true_anomaly + argument_of_periastron) + (eccentricity * np.cos(argument_of_periastron))
    return a * c / b


def radial_velocity(self, **kwargs):
    position_method = kwargs.pop("position_method")
    phases = kwargs.pop("phases")
    orbital_motion = position_method(input_argument=phases, return_nparray=True, calculate_from='phase')
    r1, r2 = distance_to_center_of_mass(self.primary.mass, self.secondary.mass, orbital_motion)

    sma_primary = orbital_semi_major_axes(r1[-1], self.orbit.eccentricity, orbital_motion[:, 3][-1])
    sma_secondary = orbital_semi_major_axes(r2[-1], self.orbit.eccentricity, orbital_motion[:, 3][-1])

    # in base SI units
    sma_primary *= self.semi_major_axis
    sma_secondary *= self.semi_major_axis
    period = np.float64((self._period * units.PERIOD_UNIT).to(u.s))

    rv_primary = _radial_velocity(sma_primary, self.inclination, self.eccentricity,
                                  self.argument_of_periastron, period, orbital_motion[:, 3]) * -1.0

    rv_secondary = _radial_velocity(sma_secondary, self.inclination, self.eccentricity,
                                    self.argument_of_periastron, period, orbital_motion[:, 3])

    # from matplotlib import pyplot as plt
    # plt.scatter(orbital_motion[:, 4], rv_primary, label="primary")
    # plt.scatter(orbital_motion[:, 4], rv_secondary, label="secondary")
    # plt.legend()
    # plt.show()
