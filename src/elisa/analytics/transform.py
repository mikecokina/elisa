import numpy as np

WHEN_FLOAT64 = (int, np.int, np.int32, np.int64, float, np.float, np.float32, np.float64)


def dataset_transform(value, unit, when_float64):
    pass


class TransformProperties(object):
    @classmethod
    def transform_input(cls, **kwargs):
        """
        Function transforms input dictionary of keyword arguments of the Analytics instance to internally usable state
        (conversion of units)

        :param kwargs: Dict
        :return: Dict
        """
        return {key: getattr(cls, key)(val) if hasattr(cls, key) else val for key, val in kwargs.items()}


class AnalyticsProperties(TransformProperties):
    pass


class DatasetProperties(TransformProperties):
    pass