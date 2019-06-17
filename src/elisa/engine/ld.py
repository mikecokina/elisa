import os
import logging
import numpy as np
import pandas as pd

from scipy import interpolate
from elisa.conf import config
from elisa.engine import utils, const
from elisa.engine.utils import is_empty

config.set_up_logging()
logger = logging.getLogger("limb-darkening-module")


def get_metallicity_from_ld_table_filename(filename):
    """
    Get metallicity as number from filename typicaly used in van hame ld tables.

    :param filename: str
    :return: float
    """
    filename = os.path.basename(filename)
    m = str(filename).split(".")[-2]
    sign = 1 if str(m).startswith("p") else -1
    value = float(m[1:]) / 10.0
    return value * sign


def get_van_hamme_ld_table_filename(passband, metallicity, law=None):
    """
    Get filename with stored coefficients for given passband, metallicity and limb darkening law.

    :param passband: str
    :param metallicity: str
    :param law: str; limb darkening law (`linear`, `cosine`, `logarithmic`, `square_root`)
    :return: str
    """
    law = law if not is_empty(law) else config.LIMB_DARKENING_LAW
    return f"{config.LD_LAW_TO_FILE_PREFIX[law]}.{passband}.{utils.numeric_metallicity_to_string(metallicity)}.csv"


def get_van_hamme_ld_table(passband, metallicity, law=None):
    """
    Get content of van hamme table (read csv file).

    :param passband: str
    :param metallicity: str
    :param law: str; in not specified, default law specified in `elisa.conf.config` is used
    :return: pandas.DataFrame
    """
    law = law if not is_empty(law) else config.LIMB_DARKENING_LAW
    filename = get_van_hamme_ld_table_filename(passband, metallicity, law=law)
    path = os.path.join(config.VAN_HAMME_LD_TABLES, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"there is no file like {path}")
    return pd.read_csv(path)


def get_van_hamme_ld_table_by_name(fname):
    """
    Get content of van hamme table defined by filename (assume it is stored in configured directory).

    :param fname: str
    :return: pandas.DataFrame
    """
    path = os.path.join(config.VAN_HAMME_LD_TABLES, fname)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"there is no file like {path}")
    return pd.read_csv(path)


def get_relevant_ld_tables(passband, metallicity, law=None):
    """
    Get filename of van hamme tables for surrounded metallicities and given passband.

    :param law: str; limb darkening law (`linear`, `cosine`, `logarithmic`, `square_root`)
    :param passband: str
    :param metallicity: str
    :return: list
    """
    # todo: make better decision which values should be used
    surrounded = utils.find_surrounded(const.VAN_HAMME_METALLICITY_LIST_LD, metallicity)
    files = [get_van_hamme_ld_table_filename(passband, m, law) for m in surrounded]
    return files


def interpolate_on_ld_grid(temperature, log_g, metallicity, passband, author=None):
    """
    Get limb darkening coefficients based on van hamme tables for given temperatures, log_gs and metallicity.

    :param passband: Dict
    :param temperature: Iterable float
    :param log_g: Iterable float
    :param metallicity: float
    :param author: str; (not implemented)
    :return: pandas.DataFrame
    """
    if isinstance(passband, dict):
        passband = passband.keys()

    results = dict()
    logger.debug('interpolating limb darkening coefficients')
    for band in passband:
        relevant_tables = get_relevant_ld_tables(passband=band, metallicity=metallicity, law=config.LIMB_DARKENING_LAW)
        csv_columns = config.LD_LAW_COLS_ORDER[config.LIMB_DARKENING_LAW]
        all_columns = csv_columns
        df = pd.DataFrame(columns=all_columns)

        for table in relevant_tables:
            _df = get_van_hamme_ld_table_by_name(table)[csv_columns]
            df = df.append(_df)

        df = df.drop_duplicates()
        xyz_domain = np.array([np.array(val) for val in df[config.LD_DOMAIN_COLS].to_records(index=False)]).tolist()
        xyz_values = df[config.LD_LAW_CFS_COLUMNS[config.LIMB_DARKENING_LAW]].to_records(index=False).tolist()

        uvw_domain = pd.DataFrame({
            "temperature": temperature,
            "gravity": log_g,
        })[config.LD_DOMAIN_COLS].to_records(index=False).tolist()

        xyz_domain = np.asarray([np.asarray(val) for val in xyz_domain])
        uvw_domain = np.asarray([np.asarray(val) for val in uvw_domain])
        xyz_values = np.asarray([np.asarray(val) for val in xyz_values])

        uvw_values = interpolate.griddata(xyz_domain, xyz_values, uvw_domain, method="linear")

        result_df = pd.DataFrame({"temperature": temperature, "log_g": log_g})

        for col, vals in zip(config.LD_LAW_CFS_COLUMNS[config.LIMB_DARKENING_LAW], uvw_values.T):
            if np.isin(np.nan, vals):
                raise ValueError("Limb darkening interpolation lead to np.nan/None value.\n"
                                 "It might be caused by definition of unphysical object on input.")
            result_df[col] = vals
        results[band] = result_df
    logger.debug('limb darkening coefficients interpolation finished')
    return results


# todo: discusse following shits
def limb_darkening_factor(normal_vector=None, line_of_sight=None, coefficients=None, limb_darkening_law=None,
                          cos_theta=None):
    """
    calculates limb darkening factor for given surface element given by radius vector and line of sight vector

    :param line_of_sight: numpy.ndarray; vector (or vectors) of line of sight (normalized to 1 !!!)
    :param normal_vector: numpy.ndarray; single or multiple normal vectors (normalized to 1 !!!)
    :param coefficients: numpy.ndarray
    :param limb_darkening_law: str;  `linear` or `cosine`, `logarithmic`, `square_root`
    :param cos_theta: numpy.ndarray; if supplied, function will skip calculation of its own cos theta and will disregard
    `normal_vector` and `line_of_sight`
    :return: numpy.ndarray; gravity darkening factors, the same type/shape as cos_theta
    """
    if normal_vector is None and cos_theta is None:
        raise ValueError('Normal vector(s) was not supplied.')
    if line_of_sight is None and cos_theta is None:
        raise ValueError('Line of sight vector(s) was not supplied.')
    if coefficients is None:
        raise ValueError('Limb darkening coefficients were not supplied.')
    if limb_darkening_law is None:
        raise ValueError('Limb darkening rule was not supplied choose from: '
                         '`linear` or `cosine`, `logarithmic`, `square_root`.')
    # if normal_vector is not None and line_of_sight is not None:
    #     if line_of_sight.ndim != 1 and normal_vector.ndim != line_of_sight.ndim:
    #         raise ValueError('A `line_of_sight` should be either one vector or ther '
    #                          'name amount of vectors as provided in radius vectors.')

    if cos_theta is None:
        cos_theta = np.sum(normal_vector * line_of_sight, axis=-1)

    # fixme: force order of coefficients; what is order now? x then y or y then x??? what index 0 or 1 means???
    if limb_darkening_law in ['linear', 'cosine']:
        return 1 - coefficients + coefficients * cos_theta
    elif limb_darkening_law == 'logarithmic':
        return 1 - coefficients[0] * (1 - cos_theta) - coefficients[1] * cos_theta * np.log(cos_theta)
    elif limb_darkening_law == 'square_root':
        return 1 - coefficients[0] * (1 - cos_theta) - coefficients[1] * (1 - np.sqrt(cos_theta))


def calculate_bolometric_limb_darkening_factor(limb_darkening_law: str = None, coefficients=None):
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
