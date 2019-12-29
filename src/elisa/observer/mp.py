from elisa.logger import getLogger

logger = getLogger('observer.mp')


def observe_lc_worker(*args):
    func, order, phase_batch, kwargs = args
    logger.info(f'starting observation worker for batch index {order}')
    kwargs.update({"phases": phase_batch})
    result = func(**kwargs)
    logger.info(f'observation worker for batch index {order} finished')
    return result

