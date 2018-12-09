import os
import pandas as pd
import numpy as np

from conf import config
from engine import utils

import logging

config.set_up_logging()
logger = logging.getLogger("ld")


def get_van_hamme_ld_table(passband, metallicity):
    filename = "{model}.{passband}.{metallicity}.csv".format(
        model=config.LD_LAW_TO_FILE_PREFIX[config.LIMB_DARKENING_LAW],
        passband=passband,
        metallicity=utils.numeric_metallicity_to_string(metallicity)
    )
    path = os.path.join(config.VAN_HAMME_LD_TABLES, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError("there is no file like {}".format(path))
    return pd.read_csv(path)


def interpolate_on_ld_grid(passband, metallicity, author=None):
    logger.debug('interpolating ld coefficients')
    ld_table = get_van_hamme_ld_table(passband=passband, metallicity=metallicity)
    # todo: implement kind of interp function


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
    interpolate_on_ld_grid(passband='Generic.Bessell.B', metallicity=0.0)
