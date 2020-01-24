from elisa.logger import getLogger
from elisa import (
    utils
)
from elisa.conf import config
from elisa.analytics.transform import DatasetProperties

logger = getLogger('analytics.dataset.base')


class Dataset():
    MANDATORY_KWARGS = config.DATASET_MANDATORY_KWARGS
    OPTIONAL_KWARGS = config.DATASET_OPTIONAL_KWARGS
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    def __init__(self, name=None, **kwargs):
        utils.invalid_kwarg_checker(kwargs, Dataset.ALL_KWARGS, Dataset)
        kwargs = self.transform_input(**kwargs)

    def transform_input(self, **kwargs):
        """
        Transform and validate input kwargs.

        :param kwargs: Dict;
        :return: Dict;
        """
        return DatasetProperties.transform_input(**kwargs)
