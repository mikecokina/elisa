import numpy as np

from elisa import utils
from elisa.base.surface.coverage import surface_area_coverage
from elisa.logger import getLogger

logger = getLogger('binary_system.curves.lcmp')


def compute_surface_coverage(system):
    """
    Compute surface coverage of faces for given rotational position
    defined by container/SystemContainer.

    :param system:
    :return:
    """
    logger.debug(f"computing surface coverage for {system.position}")
    star = getattr(system, 'star')

    coverage = utils.poly_areas(star.points[star.faces[star.indices]])
    coverage = surface_area_coverage(len(star.faces), star.indices, coverage)

    return {'star': coverage, }
