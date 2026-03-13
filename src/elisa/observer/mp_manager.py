from __future__ import annotations

import multiprocessing.pool
from typing import TYPE_CHECKING

from elisa import settings
from elisa.logger import getLogger
from elisa.utils import renormalize_async_result, split_to_batches

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from numpy.typing import NDArray

logger = getLogger("observer.mp")


def manage_observations(
        fn: Callable,
        fn_args: tuple[Any, ...],
        position: NDArray,
        **kwargs: Any,
) -> dict[str, Any]:
    """Decide whether curve will be calculated using single or multi-process approach.

    Manages calculation of curves using either a single process or a multi-process approach
    depending on the number of positions and the configured number of processes. Batches
    the positions and distributes them across workers if multiprocessing is used.

    :param fn: Function used for curve integration.
    :type fn: Callable
    :param fn_args: Tuple of arguments for `fn` (excluding position and kwargs).
    :type fn_args: tuple[Any, ...]
    :param position: Array of positions (phases) to process.
    :type position: NDArray
    :param kwargs: Additional keyword arguments passed to `fn`.
    :type kwargs: dict[str, Any]
    :returns: Calculated curves (in each passband).
    :rtype: dict[str, Any]
    """
    args = (*fn_args, kwargs)
    if len(position) >= settings.NUMBER_OF_PROCESSES > 1:
        logger.info("starting multiprocessor workers")
        phase_batches = split_to_batches(array=position, n_proc=settings.NUMBER_OF_PROCESSES)
        pool = multiprocessing.pool.Pool(processes=settings.NUMBER_OF_PROCESSES)

        result = [pool.apply_async(fn, (*args[:2], batch, *args[2:])) for batch in phase_batches]
        pool.close()
        pool.join()
        result = [r.get() for r in result]
        return renormalize_async_result(result)
    # argument has to follow singature e.g. produce_circ_sync_curves_mp(...) from c_managed.py
    args = (*args[:2], position, *args[2:])
    return fn(*args)
