import numpy as np

from elisa.conf import config
from elisa.base.transform import TransformProperties
from elisa.analytics.dataset.base import RVData, LCData

WHEN_FLOAT64 = (int, np.int, np.int32, np.int64, float, np.float, np.float32, np.float64)


class AnalyticsProperties(TransformProperties):
    @staticmethod
    def radial_velocities(value):
        if isinstance(value, dict):
            for key, val in value.items():
                if key not in config.BINARY_COUNTERPARTS.keys():
                    ValueError(f'{key} is invalid designation for radial velocity dataset. Please choose from '
                               f'{config.BINARY_COUNTERPARTS.keys()}')
                elif isinstance(val, RVData):
                    TypeError(f'{val} is not of instance of RVData class.')
        else:
            TypeError('`radial_velocities` are not of type `dict`')

        return value

    @staticmethod
    def light_curves(value):
        if isinstance(value, dict):
            for key, val in value.items():
                if key not in config.PASSBANDS:
                    ValueError(f'{key} is invalid passband. Please choose from available passbands: \n'
                               f'{config.PASSBANDS}')
                elif isinstance(val, LCData):
                    TypeError(f'{val} is not of instance of LCData class.')
        else:
            TypeError('`light_curves` are not of type `dict`')

        return value
