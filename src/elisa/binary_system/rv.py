import numpy as np


def distance_to_center_of_mass(primary_mass, secondary_mass, positions):
    """
    Return distance from primary and from secondary component to center of mass.

    :param primary_mass: float
    :param secondary_mass: float
    :param positions: numpy.array
    :return:
    """
    distance = np.array(positions)[:, 1]
    mass = primary_mass + secondary_mass
    com_from_primary = (distance * secondary_mass) / mass
    return com_from_primary, distance - com_from_primary


def orbital_semi_major_axes():
    """

    :return:
    """
    pass


def radial_velocity(self, **kwargs):
    position_method = kwargs.pop("position_method")
    phases = kwargs.pop("phases")
    orbital_motion = position_method(input_argument=phases, return_nparray=False, calculate_from='phase')
    r1, r2 = distance_to_center_of_mass(self.primary.mass, self.secondary.mass, orbital_motion)

    print(r1, r2)
