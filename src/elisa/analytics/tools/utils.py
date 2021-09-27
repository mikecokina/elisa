import numpy as np

from ... import units as u
from ... binary_system import t_layer


def lightcurves_mean_error(lc):
    """
    If observation errors are not provided, the default 5 percent relative error
    is used to generate synthetic errors.

    :param lc: numpy.array; light curve
    :return: numpy.array; synthetic errors
    """
    return np.mean(lc) * 0.05


def radialcurves_mean_error(rv):
    """
    If observation errors are not provided, the default 5 percent relative error
    is used to generate synthetic errors.

    :param rv: numpy.array; radial velocities
    :return: numpy.array; synthetic errors
    """
    return np.mean(rv) * 0.05


def is_time_dependent(labels):
    """
    If 'system@primary_minimum_time' is located in the fit parameters, the fit parameters are considered
    time dependent and observations are therefore expected to be supplied in JD.

    :param labels: List[str]; parameter labels
    :return: bool
    """
    if 'system@period' in labels and 'system@primary_minimum_time' in labels:
        return True
    return False


def time_layer_resolver(x_data, pop=False, **kwargs):
    """
    If kwargs contain `period` and `primary_minimum_time`, then xs is expected to be JD
    time not phases. Then, x_data (observational time) has to be converted to phases.

    :param pop: bool; determine if remove the system@primary_minimum_time parameter from the fit
                      parameters or just read it
    :param x_data: Union[List, numpy.array];
    :param kwargs: Dict;
    :return: Tuple;
    """

    if is_time_dependent(list(kwargs.keys())):
        t0 = kwargs['system@primary_minimum_time']
        if pop:
            kwargs.pop('system@primary_minimum_time')
        period = kwargs['system@period']
        x_data_new = t_layer.jd_to_phase(t0, period, x_data, centre=0.5)
    else:
        x_data_new = x_data % 1.0
    return x_data_new, kwargs
