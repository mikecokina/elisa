from ... import utils
from ... base.surface.coverage import surface_area_coverage
from ... logger import getLogger

logger = getLogger('single_system.surface.coverage')


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
