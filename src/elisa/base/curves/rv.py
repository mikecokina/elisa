from ... import settings
from ...observer.passband import init_rv_passband


def include_passband_data_to_kwargs(**kwargs):
    """
    Including dummy passband from which radiometric radial velocities will be calculated.

    :param kwargs: Tuple;
    :return: Tuple;
    """
    psbnd, right_bandwidth, left_bandwidth = init_rv_passband()
    kwargs.update({
        'passband': {'rv_band': psbnd},
        'left_bandwidth': left_bandwidth,
        'right_bandwidth': right_bandwidth
    })
    return kwargs
