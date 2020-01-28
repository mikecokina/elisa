import numpy as np

from elisa.base.transform import TransformProperties

WHEN_FLOAT64 = (int, np.int, np.int32, np.int64, float, np.float, np.float32, np.float64)


class AnalyticsProperties(TransformProperties):
    pass

