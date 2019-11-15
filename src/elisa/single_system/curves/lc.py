import numpy as np

from elisa import (
    const,
    utils
)
from elisa.logger import getLogger
from elisa.conf import config
from elisa.single_system.container import SystemContainer
from elisa.single_system.curves import shared
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
    from_this = dict(binary_system=single, position=const.SinglePosition(0, 0.0, 0.0))
    initial_system = SystemContainer.from_single_system(**from_this)
    initial_system.build()

    phases = kwargs.pop("phases")
    normal_radiance, ld_cfs = shared.prep_surface_params(initial_system.copy().flatt_it(), **kwargs)

    # if config.NUMBER_OF_PROCESSES > 1:
    #     logger.info("starting multiprocessor workers")
    #     batch_size = int(np.ceil(len(unique_phase_interval) / config.NUMBER_OF_PROCESSES))
    #     phase_batches = utils.split_to_batches(batch_size=batch_size, array=unique_phase_interval)
    #     func = lcmp.compute_circular_synchronous_lightcurve
    #     pool = Pool(processes=config.NUMBER_OF_PROCESSES)
    #
    #     result = [pool.apply_async(func, (binary, initial_system, batch, normal_radiance, ld_cfs, kwargs))
    #               for batch in phase_batches]
    #     pool.close()
    #     pool.join()
    #     # this will return output in same order as was given on apply_async init
    #     result = [r.get() for r in result]
    #     band_curves = bsutils.renormalize_async_result(result)
    # else:
    #     args = (binary, initial_system, unique_phase_interval, normal_radiance, ld_cfs, kwargs)
    #     band_curves = lcmp.compute_circular_synchronous_lightcurve(*args)
    #
    # band_curves = {band: band_curves[band][reverse_phase_map] for band in band_curves}
    # return band_curves

    # TODO: finish


def compute_light_curve_with_pulsations(self, **kwargs):
    return NotImplementedError('Pulsations not yet fully implemented')
