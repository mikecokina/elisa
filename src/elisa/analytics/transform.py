from . dataset.base import RVData, LCData
from .. base.transform import TransformProperties
from .. import settings


class RVBinaryAnalyticsTask(TransformProperties):
    """
    Evaluating whether the observational data in elisa.tasks.RVBinaryAnalyticsTask are supplied in valid form.
    """
    @staticmethod
    def data(value):
        if isinstance(value, dict):
            for key, val in value.items():
                if key not in settings.BINARY_COUNTERPARTS:
                    raise ValueError(f'{key} is invalid designation for radial velocity dataset. '
                                     f'Please choose from {settings.BINARY_COUNTERPARTS.keys()}')
                elif not isinstance(val, RVData):
                    raise TypeError(f'{val} is not of instance of RVData class.')
            return value
        raise TypeError('`radial_velocities` are not of type `dict`')


class LCBinaryAnalyticsProperties(TransformProperties):
    """
    Evaluating whether the observational data in elisa.tasks.LCBinaryAnalyticsTask are supplied in valid form.
    """
    @staticmethod
    def data(value):
        if isinstance(value, dict):
            for key, val in value.items():
                if key not in settings.PASSBANDS:
                    raise ValueError(f'{key} is invalid passband. Please choose '
                                     f'from available passbands: \n{settings.PASSBANDS}')
                elif not isinstance(val, LCData):
                    raise TypeError(f'{val} is not of instance of LCData class.')
            return value
        raise TypeError('`light_curves` are not of type `dict`')
