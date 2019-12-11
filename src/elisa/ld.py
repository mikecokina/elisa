import os
import numpy as np
import pandas as pd

from elisa.base.error import LimbDarkeningError
from elisa.logger import getLogger
from scipy import interpolate
from elisa.conf import config
from elisa import (
    utils,
    const,
    umpy as up
)

logger = getLogger(__name__)


def get_metallicity_from_ld_table_filename(filename):
    """
    Get metallicity as number from filename typicaly used in van hame ld tables.

    :param filename: str;
    :return: float;
    """
    filename = os.path.basename(filename)
    m = str(filename).split(".")[-2]
    return utils.numeric_metallicity_from_string(m)


def get_van_hamme_ld_table_filename(passband, metallicity, law=None):
    """
    Get filename with stored coefficients for given passband, metallicity and limb darkening law.

    :param passband: str
    :param metallicity: str
    :param law: str; limb darkening law (`linear`, `cosine`, `logarithmic`, `square_root`)
    :return: str
    """
    law = law if not utils.is_empty(law) else config.LIMB_DARKENING_LAW
    return f"{config.LD_LAW_TO_FILE_PREFIX[law]}.{passband}.{utils.numeric_metallicity_to_string(metallicity)}.csv"


def get_van_hamme_ld_table(passband, metallicity, law=None):
    """
    Get content of van hamme table (read csv file).

    :param passband: str;
    :param metallicity: str;
    :param law: str; in not specified, default law specified in `elisa.conf.config` is used
    :return: pandas.DataFrame;
    """
    law = law if not utils.is_empty(law) else config.LIMB_DARKENING_LAW
    filename = get_van_hamme_ld_table_filename(passband, metallicity, law=law)
    path = os.path.join(config.VAN_HAMME_LD_TABLES, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"There is no file like {path}.")
    return pd.read_csv(path)


def get_van_hamme_ld_table_by_name(fname):
    """
    Get content of van hamme table defined by filename (assume it is stored in configured directory).

    :param fname: str;
    :return: pandas.DataFrame;
    """
    logger.debug(f"accessing limb darkening file {fname}")
    path = os.path.join(config.VAN_HAMME_LD_TABLES, fname)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"There is no file like {path}.")
    return pd.read_csv(path)


def get_relevant_ld_tables(passband, metallicity, law=None):
    """
    Get filename of van hamme tables for surrounded metallicities and given passband.

    :param law: str; limb darkening law (`linear`, `cosine`, `logarithmic`, `square_root`)
    :param passband: str;
    :param metallicity: str;
    :return: List;
    """
    # todo: make better decision which values should be used
    surrounded = utils.find_surrounded(const.VAN_HAMME_METALLICITY_LIST_LD, metallicity)
    files = [get_van_hamme_ld_table_filename(passband, m, law) for m in surrounded]
    return files


def interpolate_on_ld_grid(temperature, log_g, metallicity, passband, author=None):
    """
    Get limb darkening coefficients based on van hamme tables for given temperatures, log_gs and metallicity.

    :param passband: Dict;
    :param temperature: Iterable[float];
    :param log_g: Iterable[float]; values expected in log_SI units
    :param metallicity: float;
    :param author: str; (not implemented)
    :return: pandas.DataFrame;
    """
    if isinstance(passband, dict):
        passband = passband.keys()

    # convert logg from log(SI) to log(cgs)
    log_g = utils.convert_gravity_acceleration_array(log_g, units='log_cgs')

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
        xyz_domain = df[config.LD_DOMAIN_COLS].values
        xyz_values = df[config.LD_LAW_CFS_COLUMNS[config.LIMB_DARKENING_LAW]].values

        uvw_domain = np.column_stack((temperature, log_g))
        uvw_values = interpolate.griddata(xyz_domain, xyz_values, uvw_domain, method="linear")

        result_df = pd.DataFrame({"temperature": temperature, "log_g": log_g})

        for col, vals in zip(config.LD_LAW_CFS_COLUMNS[config.LIMB_DARKENING_LAW], uvw_values.T):
            if np.any(up.isnan(vals)):
                raise LimbDarkeningError("Limb darkening interpolation lead to numpy.nan/None value. "
                                         "It might be caused by definition of unphysical object on input.")
            result_df[col] = vals
        results[band] = result_df
    logger.debug('limb darkening coefficients interpolation finished')
    return results


# todo: discuss following shits
def limb_darkening_factor(normal_vector=None, line_of_sight=None, coefficients=None, limb_darkening_law=None,
                          cos_theta=None):
    """
    calculates limb darkening factor for given surface element given by radius vector and line of sight vector

    :param line_of_sight: numpy.array; vector (or vectors) of line of sight (normalized to 1 !!!)
    :param normal_vector: numpy.array; single or multiple normal vectors (normalized to 1 !!!)
    :param coefficients: numpy.array;

    shape::

        - numpy.array[[c0, c2, c3, c4,..., cn]] for linear law
        - numpy.array[[c0, c2, c3, c4,..., cn],
                      [d0, d2, d3, c4,..., dn]] for sqrt and log law

    :param limb_darkening_law: str;  `linear` or `cosine`, `logarithmic`, `square_root`
    :param cos_theta: numpy.array; if supplied, function will skip calculation of its own cos theta and will disregard
                                   `normal_vector` and `line_of_sight`
    :return: numpy.array; gravity darkening factors, the same type/shape as cos_theta
    """
    if normal_vector is None and cos_theta is None:
        raise ValueError('Normal vector(s) was not supplied.')
    if line_of_sight is None and cos_theta is None:
        raise ValueError('Line of sight vector(s) was not supplied.')
    if coefficients is None:
        raise LimbDarkeningError('Limb darkening coefficients were not supplied.')
    if limb_darkening_law is None:
        raise LimbDarkeningError('Limb darkening rule was not supplied choose from: '
                                 '`linear` or `cosine`, `logarithmic`, `square_root`.')

    if cos_theta is None:
        cos_theta = np.sum(normal_vector * line_of_sight, axis=-1)
    else:
        if cos_theta.ndim == 1:
            cos_theta = cos_theta[:, np.newaxis]

    cos_theta = cos_theta.copy()
    negative_cos_theta_test = cos_theta <= 0
    if limb_darkening_law in ['linear', 'cosine']:
        cos_theta[negative_cos_theta_test] = 0.0
        retval = 1.0 - coefficients + coefficients * cos_theta
        retval[negative_cos_theta_test] = 0.0
    elif limb_darkening_law == 'logarithmic':
        cos_theta_for_log = cos_theta.copy()
        cos_theta[negative_cos_theta_test] = 0.0
        cos_theta_for_log[negative_cos_theta_test] = 1.0
        retval = \
            1.0 - coefficients[:, :1] * (1 - cos_theta) - coefficients[:, 1:] * cos_theta * up.log(cos_theta_for_log)
        retval[negative_cos_theta_test] = 0.0
    elif limb_darkening_law == 'square_root':
        cos_theta[negative_cos_theta_test] = 0.0
        retval = 1.0 - coefficients[:, :1] * (1 - cos_theta) - coefficients[:, 1:] * (1 - up.sqrt(cos_theta))
        retval[negative_cos_theta_test] = 0.0
    else:
        raise LimbDarkeningError("Invalid limb darkening.")
    return retval[:, 0] if retval.shape[1] == 1 else retval


def calculate_bolometric_limb_darkening_factor(limb_darkening_law=None, coefficients=None):
    """
    Calculates limb darkening factor D(int) used when calculating flux from given intensity on surface.
    D(int) = integral over hemisphere (D(theta)cos(theta)

    :param limb_darkening_law: str -  `linear` or `cosine`, `logarithmic`, `square_root`
    :param coefficients: numpy.array;
    :return: np.array; - bolometric_limb_darkening_factor (scalar for the whole star)
    """
    if coefficients is None:
        raise LimbDarkeningError('Limb darkening coefficients were not supplied.')
    elif limb_darkening_law is None:
        raise LimbDarkeningError('Limb darkening rule was not supplied choose from: '
                                 '`linear` or `cosine`, `logarithmic`, `square_root`.')

    if limb_darkening_law in ['linear', 'cosine']:
        return const.PI * (1 - coefficients[0, :] / 3)
    elif limb_darkening_law == 'logarithmic':
        return const.PI * (1 - coefficients[0, :] / 3 + 2 * coefficients[1, :] / 9)
    elif limb_darkening_law == 'square_root':
        return const.PI * (1 - coefficients[0, :] / 3 - coefficients[1, :] / 5)


def get_bolometric_ld_coefficients(temperature, log_g, metallicity):
    columns = config.LD_LAW_CFS_COLUMNS[config.LIMB_DARKENING_LAW]
    coeffs = interpolate_on_ld_grid(temperature, log_g, metallicity, passband=["bolometric"])["bolometric"][columns]
    return np.array(coeffs).T