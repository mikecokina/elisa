from elisa import utils


def correct_component_mesh(star):
    """
    Correcting the underestimation of the surface due to the discretization.

    :param star: elisa.base.container.StarContainer;
    :return: elisa.base.container.StarContainer;
    """
    star.points *= utils.discretization_correction_factor(star.discretization_factor)

    if star.has_spots():
        for spot in star.spots.values():
            spot.points *= utils.discretization_correction_factor(spot.discretization_factor)

    return star
