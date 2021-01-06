import numpy as np

from elisa import utils


def correct_component_mesh(star, correction_factors):
    """
    Correcting the underestimation of the surface due to the discretization.

    :param correction_factors: numpy.array; (2*N) [discretization factor, correction factor],
    :param star: elisa.base.container.StarContainer;
    :return: elisa.base.container.StarContainer;
    """
    star.points *= utils.discretization_correction_factor(star.discretization_factor, correction_factors)

    if star.has_spots():
        for spot in star.spots.values():
            spot.points *= utils.discretization_correction_factor(spot.discretization_factor, correction_factors)

    return star
