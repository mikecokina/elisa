import pandas as pd
import numpy as np

from ... import units as u
from ... import utils
from ... import settings


def convert_data(data, unit, to_unit):
    """
    Converts data to desired format or leaves it dimensionless.

    :param data: numpy.array;
    :param unit: astropy.unit;
    :param to_unit: astropy.unit;
    :return: numpy.array;
    """
    return data if unit == u.dimensionless_unscaled else (data * unit).to(to_unit).value


def convert_flux(data, unit, zero_point=None):
    """
    If data are in magnitudes, they are converted to normalized flux.

    :param data: numpy.array;
    :param unit: astropy.unit.Unit;
    :param zero_point: float;
    :return: numpy.array;
    """
    if unit == u.mag:
        if zero_point is None:
            raise ValueError('You supplied your data in magnitudes. Please also specify '
                             'a zero point using keyword argument `reference_magnitude`.')
        else:
            data = utils.magnitude_to_flux(data, zero_point)

    return data


def convert_flux_error(error, unit, zero_point=None):
    """
    If data an its errors are in magnitudes, they are converted to normalized flux.

    :param error: numpy.array;
    :param unit: astropy.unit.Unit;
    :param zero_point: float;
    :return: numpy.array;
    """
    if unit == u.mag:
        if zero_point is None:
            raise ValueError('You supplied your data in magnitudes. Please also specify '
                             'a zero point using keyword argument `reference_magnitude`.')
        else:
            error = utils.magnitude_error_to_flux_error(error)
    return error


def convert_unit(unit, to_unit):
    """
    Converts to desired unit  or leaves it dimensionless.

    :param unit: astropy.unit;
    :param to_unit: astropy.unit;
    :return: astropy.unit;
    """
    return unit if unit == u.dimensionless_unscaled else to_unit


def read_data_file(filename, data_columns, delimiter=settings.DELIM_WHITESPACE):
    """
    Function loads observation datafile. Rows with column names and comments should start with `#`.
    It deals with missing data by omitting given line

    :param delimiter: str; regex to define columns separtor
    :param filename: str;
    :param data_columns: Tuple;
    :return: numpy.array;
    """
    data = pd.read_csv(filename, header=None, comment='#', delimiter=delimiter,
                       error_bad_lines=False, engine='python')[list(data_columns)]
    data = data.apply(lambda s: pd.to_numeric(s, errors='coerce')).dropna()
    return data.to_numpy(dtype=float)


def central_moving_average(dt_set, n_bins=100, radius=2, cyclic_boundaries=True):
    """
    Function performs central moving averages in order to smooth observations in given dataset. Use this function only
    on phased data.

    :param dt_set:
    :param n_bins:
    :param radius:
    :param cyclic_boundaries:
    :return:
    """
    bin_boundaries = np.linspace(dt_set.x_data.min(), dt_set.x_data.max(), num=n_bins + 1, endpoint=True)
    bin_centres = 0.5 * (bin_boundaries[:-1] + bin_boundaries[1:])
    bin_idxs = np.digitize(dt_set.x_data, bin_boundaries[1:], right=True)

    if cyclic_boundaries:
        bins = [(np.arange(start=ii-radius, stop=ii+radius+1, step=1.0, dtype=np.int) % n_bins)
                for ii in range(n_bins)]
    else:
        bins = [np.arange(start=0, stop=ii+radius+1, step=1.0, dtype=np.int) for ii in range(radius)]
        bins += [np.arange(start=ii-radius, stop=ii+radius+1, step=1.0, dtype=np.int)
                 for ii in range(radius, n_bins-radius)]
        bins += [np.arange(start=ii-radius, stop=n_bins, step=1.0, dtype=np.int)
                 for ii in range(n_bins-radius, n_bins)]

    bin_masks = [np.isin(bin_idxs, bins[ii]) for ii in range(n_bins)]
    bin_masks_count = np.sum(bin_masks, axis=1)

    non_empty_bins = bin_masks_count > 0
    iterator = np.arange(n_bins)[non_empty_bins]

    if dt_set.y_err is not None:
        bin_averages = np.array([np.average(dt_set.y_data[bin_masks[ii]],
                                            weights=1/dt_set.y_err[bin_masks[ii]]**2)
                                 for ii in iterator])
        bin_errors = np.array([np.mean(dt_set.y_err[bin_masks[ii]]) for ii in iterator])
    else:
        bin_averages = np.array([np.average(dt_set.y_data[bin_masks[ii]]) for ii in iterator])
        bin_errors = np.array([np.std(dt_set.y_data[bin_masks[ii]]) for ii in iterator(n_bins)])

    dt_set.x_data = bin_centres[non_empty_bins]
    dt_set.y_data = bin_averages
    dt_set.y_err = bin_errors
