from multiprocessing.pool import Pool

from elisa.logger import getLogger
from elisa.conf import config
from elisa import (
    umpy as up,
    utils
)

logger = getLogger('observer.mp')


def observe_lc_worker(*args):
    func, order, phase_batch, kwargs = args
    logger.info(f'starting observation worker for batch index {order}')
    kwargs.update({"phases": phase_batch})
    result = func(**kwargs)
    logger.info(f'observation worker for batch index {order} finished')
    return result


def manage_observations(fn, fn_args, position, **kwargs):
    """
    function decides whether LC will be calculated using single or multi-process aproach

    :param fn: function used for LC integration
    :param fn_args: tuple; some of the argument in `fn`
    :param position: list;
    :param kwargs: dict;
    :return: dict; calculated LCs in different passbands
    """
    args = fn_args + (kwargs, )
    if config.NUMBER_OF_PROCESSES > 1:
        logger.info("starting multiprocessor workers")
        batch_size = int(up.ceil(len(position) / config.NUMBER_OF_PROCESSES))
        phase_batches = utils.split_to_batches(batch_size=batch_size, array=position)
        pool = Pool(processes=config.NUMBER_OF_PROCESSES)

        result = [pool.apply_async(fn, args[:2] + (batch,) + args[2:]) for batch in phase_batches]
        pool.close()
        pool.join()
        # this will return output in same order as was given on apply_async init
        result = [r.get() for r in result]
        band_curves = utils.renormalize_async_result(result)
        return band_curves
    else:
        args = args[:2] + (position,) + args[2:]
        return fn(*args)
