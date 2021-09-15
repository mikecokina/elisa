from multiprocessing.pool import Pool

from .. import utils
from .. logger import getLogger
from .. import settings

logger = getLogger('observer.mp')


def manage_observations(fn, fn_args, position, **kwargs):
    """
    Function decides whether curve will be calculated using single or multi-process approach.

    :param fn: function used for curve integration
    :param fn_args: Tuple; some of the argument in `fn`
    :param position: List;
    :param kwargs: Dict;
    :return: Dict; calculated curves (in each passbands)
    """
    args = fn_args + (kwargs, )
    if len(position) >= settings.NUMBER_OF_PROCESSES > 1:
        logger.info("starting multiprocessor workers")
        phase_batches = utils.split_to_batches(array=position, n_proc=settings.NUMBER_OF_PROCESSES)
        pool = Pool(processes=settings.NUMBER_OF_PROCESSES)

        result = [pool.apply_async(fn, args[:2] + (batch, ) + args[2:]) for batch in phase_batches]
        pool.close()
        pool.join()
        # this will return output in same order as was given on apply_async init
        result = [r.get() for r in result]
        band_curves = utils.renormalize_async_result(result)
        return band_curves
    else:
        args = args[:2] + (position, ) + args[2:]
        return fn(*args)
