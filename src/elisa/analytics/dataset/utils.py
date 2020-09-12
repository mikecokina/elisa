import astropy.units as au
import pandas as pd

from elisa import utils
from elisa.conf import config


def convert_data(data, unit, to_unit):
    """
    Converts data to desired format or leaves it dimensionless.

    :param data: numpy.array;
    :param unit: astropy.unit;
    :param to_unit: astropy.unit;
    :return: numpy.array;
    """
    return data if unit == au.dimensionless_unscaled else (data * unit).to(to_unit).value


def convert_flux(data, unit, zero_point=None):
    """
    If data are in magnitudes, they are converted to normalized flux.

    :param data: numpy.array;
    :param unit: astropy.unit.Unit;
    :param zero_point: float;
    :return: numpy.array;
    """
    if unit == au.mag:
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
    if unit == au.mag:
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
    return unit if unit == au.dimensionless_unscaled else to_unit


def read_data_file(filename, data_columns, delimiter=config.DELIM_WHITESPACE):
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


def central_moving_average(dt_set, **kwargs):
    pass

