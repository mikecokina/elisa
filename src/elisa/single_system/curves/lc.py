import numpy as np

from elisa import (
    const,
    utils
)
from copy import copy
from elisa.logger import getLogger
from elisa.conf import config
from elisa.single_system.container import SystemContainer
from elisa.single_system.curves import shared, lcmp
from elisa.single_system import utils
# from elisa.single_system


logger = getLogger('single_system.curves.lc')


def compute_light_curve_without_pulsations(single, **kwargs):
    """
    Compute light curve for single star objects without pulsations.

    :param single: elisa.single_system.system.SinarySystem;
    :param kwargs: Dict;
            * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
            * ** left_bandwidth ** * - float
            * ** right_bandwidth ** * - float
            * ** atlas ** * - str
            * ** position_method** * - function definition; to evaluate orbital positions
    :return: Dict[str, numpy.array];
    """
    from_this = dict(single_system=single, position=const.SinglePosition(0, 0.0, 0.0))
    initial_system = SystemContainer.from_single_system(**from_this)
    initial_system.build()

    phases = kwargs.pop("phases")
    normal_radiance, ld_cfs = shared.prep_surface_params(initial_system.copy().flatt_it(), **kwargs)

    if (config.NUMBER_OF_PROCESSES > 1) and (len(phases) >= config.NUMBER_OF_PROCESSES):
        logger.info("starting multiprocessor workers")
        batch_size = int(np.ceil(len(phases) / config.NUMBER_OF_PROCESSES))
        phase_batches = utils.split_to_batches(batch_size=batch_size, array=phases)

        raise NotImplementedError('Multiprocessing in non-pulsating case is not not implemented')

    else:
        args = (single, initial_system, phases, normal_radiance, ld_cfs, kwargs)
        band_curves = lcmp.compute_non_pulsating_lightcurve(*args)

    return band_curves


def compute_light_curve_with_pulsations(single, **kwargs):
    from_this = dict(single_system=single, position=const.SinglePosition(0, 0.0, 0.0))
    initial_system = SystemContainer.from_single_system(**from_this)
    initial_system.build_mesh()

    phases = kwargs.pop("phases")

    # return NotImplementedError('Pulsations not yet fully implemented')
