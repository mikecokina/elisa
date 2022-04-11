import os
import json

from elisa import settings


def load_standard(system):
    """
    Loads zero points for magnitude calculations.

    :param system: str; available: `vega`
    :return: dict; containing zero point for each filter
    """
    file = os.path.join(settings.DATA_PATH, "zero_points", f'{system}.json')
    with open(file, 'r') as fl:
        standard_fluxes = json.load(fl)

    return standard_fluxes
