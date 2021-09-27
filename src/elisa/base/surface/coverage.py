import numpy as np


def surface_area_coverage(size, visible, visible_coverage, partial=None, partial_coverage=None):
    """
    Prepare array with coverage of surface areas.

    :param size: int; size of array
    :param visible: numpy.array; full visible areas (numpy fancy indexing), array like [False, True, True, False]
    :param visible_coverage: numpy.array; defines coverage of visible (coverage onTrue positions)
    :param partial: numpy.array; partial visible areas (numpy fancy indexing)
    :param partial_coverage: numpy.array; defines coverage of partial visible
    :return: numpy.array
    """
    # initialize zeros, since there is no input for invisible (it means everything what left after is invisible)
    coverage = np.zeros(size)
    coverage[visible] = visible_coverage
    if partial is not None:
        coverage[partial] = partial_coverage
    return coverage
