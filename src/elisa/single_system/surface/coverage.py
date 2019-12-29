import numpy as np

from elisa import utils
from elisa.logger import getLogger

logger = getLogger('binary_system.curves.lcmp')


def compute_surface_coverage(system):
    logger.debug(f"computing surface coverage for {system.position}")
    star = getattr(system, 'star')
    visible_point_indices = np.unique(star.faces[star.indices])

    visible_projection = utils.get_visible_projection(star)
    out_of_bound = np.ones(visible_projection.shape[0], dtype=np.bool)
    pass
