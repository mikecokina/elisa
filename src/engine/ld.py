import logging
import os

import numpy as np
import pandas as pd

from conf import config
from engine import utils, const
from scipy import interpolate


config.set_up_logging()
logger = logging.getLogger("ld")


def get_metallicity_from_ld_table_filename(filename):
    m = str(filename).split(".")[-2]
    sign = 1 if str(m).startswith("p") else -1
    value = int(m[1:]) / 10.0
    return value * sign


def get_van_hamme_ld_table_filename(passband, metallicity, law=None):
    law = law or config.LIMB_DARKENING_LAW
    return "{model}.{passband}.{metallicity}.csv".format(
        model=config.LD_LAW_TO_FILE_PREFIX[law],
        passband=passband,
        metallicity=utils.numeric_metallicity_to_string(metallicity)
    )


def get_van_hamme_ld_table(passband, metallicity, law=None):
    law = law or config.LIMB_DARKENING_LAW
    filename = get_van_hamme_ld_table_filename(passband, metallicity, law=law)
    path = os.path.join(config.VAN_HAMME_LD_TABLES, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError("there is no file like {}".format(path))
    return pd.read_csv(path)


def get_van_hamme_ld_table_by_name(fname):
    path = os.path.join(config.VAN_HAMME_LD_TABLES, fname)
    if not os.path.isfile(path):
        raise FileNotFoundError("there is no file like {}".format(path))
    return pd.read_csv(path)


def get_relevant_ld_tables(passband, metallicity):
    # todo: make better decision which values should be used
    surrounded = utils.find_surounded(const.VAN_HAMME_METALLICITY_LIST_LD, metallicity)
    files = [get_van_hamme_ld_table_filename(passband, m) for m in surrounded]
    return files


def interpolate_on_ld_grid(passband, temperature, log_g, metallicity, author=None):
    config.LIMB_DARKENING_LAW = "square_root"

    logger.debug('interpolating ld coefficients')
    relevant_tables = get_relevant_ld_tables(passband=passband, metallicity=metallicity)
    csv_columns = config.LD_LAW_COLS_ORDER[config.LIMB_DARKENING_LAW]
    all_columns = csv_columns + ["metallicity"]
    df = pd.DataFrame(columns=all_columns)

    for table in relevant_tables:
        _df = get_van_hamme_ld_table_by_name(table)[csv_columns]
        _df["metallicity"] = get_metallicity_from_ld_table_filename(table)
        df = df.append(_df)

    xyz_domain = np.array([np.array(val) for val in df[config.LD_DOMAIN_COLS].to_records(index=False)]).tolist()
    xyz_values = df[config.LD_LAW_CFS_COLUMNS[config.LIMB_DARKENING_LAW]].to_records(index=False).tolist()

    uvw_domain = pd.DataFrame({
        "temperature": temperature,
        "gravity": log_g,
        "metallicity": [metallicity] * len(temperature)
    })[config.LD_DOMAIN_COLS].to_records(index=False).tolist()

    xyz_domain = np.asarray([np.asarray(val) for val in xyz_domain])
    uvw_domain = np.asarray([np.asarray(val) for val in uvw_domain])
    xyz_values = np.asarray([np.asarray(val) for val in xyz_values])

    uvw_values = interpolate.griddata(xyz_domain, xyz_values, uvw_domain, method="linear")

    result_df = pd.DataFrame({
        "temperature": temperature,
        "log_g": log_g,
        "metallicity": [metallicity] * len(temperature),
    })

    for col, vals in zip(config.LD_LAW_CFS_COLUMNS[config.LIMB_DARKENING_LAW], uvw_values.T):
        result_df[col] = vals

    return result_df



def limb_darkening_factor(normal_vector=None, line_of_sight=None, coefficients=None, limb_darkening_law=None):
    """
    calculates limb darkening factor for given surface element given by radius vector and line of sight vector
    :param line_of_sight: numpy.array - vector (or vectors) of line of sight (normalized to 1 !!!)
    :param normal_vector: numpy.array - single or multiple normal vectors (normalized to 1 !!!)
    :param coefficients: np.float in case of linear law
                         np.array in other cases
    :param limb_darkening_law: str -  `linear` or `cosine`, `logarithmic`, `square_root`
    :return:  gravity darkening factor(s), the same type/shape as theta
    """
    if normal_vector is None:
        raise ValueError('Normal vector(s) was not supplied.')
    if line_of_sight is None:
        raise ValueError('Line of sight vector(s) was not supplied.')

    # if line_of_sight.ndim != 1 and normal_vector.ndim != line_of_sight.ndim:
    #     raise ValueError('`line_of_sight` should be either one vector or ther same amount of vectors as provided in'
    #                      ' radius vectors')

    if coefficients is None:
        raise ValueError('Limb darkening coefficients were not supplied.')
    elif limb_darkening_law is None:
        raise ValueError('Limb darkening rule was not supplied choose from: `linear` or `cosine`, `logarithmic`, '
                         '`square_root`.')
    elif limb_darkening_law in ['linear', 'cosine']:
        if not np.isscalar(coefficients):
            raise ValueError('Only one scalar limb darkening coefficient is required for linear cosine law. You '
                             'used: {}'.format(coefficients))
    elif limb_darkening_law in ['logarithmic', 'square_root']:
        if not np.shape(coefficients) == (2,):
            raise ValueError('Invalid number of limb darkening coefficients. Expected 2, given: '
                             '{}'.format(coefficients))

    cos_theta = np.sum(normal_vector * line_of_sight, axis=-1)
    if limb_darkening_law in ['linear', 'cosine']:
        return 1 - coefficients + coefficients * cos_theta
    elif limb_darkening_law == 'logarithmic':
        return 1 - coefficients[0] * (1 - cos_theta) - coefficients[1] * cos_theta * np.log(cos_theta)
    elif limb_darkening_law == 'square_root':
        return 1 - coefficients[0] * (1 - cos_theta) - coefficients[1] * (1 - np.sqrt(cos_theta))


def calculate_bolometric_limb_darkening_factor(limb_darkening_law=None, coefficients=None):
    """
    Calculates limb darkening factor D(int) used when calculating flux from given intensity on surface.
    D(int) = integral over hemisphere (D(theta)cos(theta)

    :param limb_darkening_law: str -  `linear` or `cosine`, `logarithmic`, `square_root`
    :param coefficients: np.float in case of linear law
                         np.array in other cases
    :return: float - bolometric_limb_darkening_factor (scalar for the whole star)
    """
    if coefficients is None:
        raise ValueError('Limb darkening coefficients were not supplied.')
    elif limb_darkening_law is None:
        raise ValueError('Limb darkening rule was not supplied choose from: `linear` or `cosine`, `logarithmic`, '
                         '`square_root`.')
    elif limb_darkening_law in ['linear', 'cosine']:
        if not np.isscalar(coefficients):
            raise ValueError('Only one scalar limb darkening coefficient is required for linear cosine law. You '
                             'used: {}'.format(coefficients))
    elif limb_darkening_law in ['logarithmic', 'square_root']:
        if not np.shape(coefficients) == (2,):
            raise ValueError('Invalid number of limb darkening coefficients. Expected 2, given: '
                             '{}'.format(coefficients))

    if limb_darkening_law in ['linear', 'cosine']:
        return np.pi * (1 - coefficients / 3)
    elif limb_darkening_law == 'logarithmic':
        return np.pi * (1 - coefficients[0] / 3 + 2 * coefficients[1] / 9)
    elif limb_darkening_law == 'square_root':
        return np.pi * (1 - coefficients[0] / 3 - coefficients[1] / 5)


if __name__ == '__main__':
    _temperature = [
        5551.36,
        5552.25,
        6531.81,
        7825.66,
        9874.85
    ]

    _metallicity = 0.11

    _logg = [
        4.12,
        3.92,
        2.85,
        2.99,
        3.11
    ]

    interpolate_on_ld_grid(passband='Generic.Bessell.B', temperature=_temperature,
                           log_g=_logg, metallicity=_metallicity)
