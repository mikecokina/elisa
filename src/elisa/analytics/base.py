from elisa.logger import getLogger
from elisa import (
    utils
)
from elisa.conf import config
from elisa.analytics.transform import AnalyticsProperties

logger = getLogger('analytics.base')


class Analytics():
    MANDATORY_KWARGS = []
    OPTIONAL_KWARGS = ['radial_velocities', 'light_curve']
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    DATASET_MANDATORY_KWARGS = config.DATASET_MANDATORY_KWARGS
    DATASET_OPTIONAL_KWARGS = config.DATASET_OPTIONAL_KWARGS
    DATASET_ALL_KWARGS = DATASET_MANDATORY_KWARGS + DATASET_OPTIONAL_KWARGS

    def __init__(self, datasets, **kwargs):
        # initial validity checks
        utils.invalid_kwarg_checker(kwargs, Analytics.ALL_KWARGS, self.__class__)
        utils.check_missing_kwargs(Analytics.MANDATORY_KWARGS, kwargs, instance_of=Analytics)
        self.dataset_params_validity_check(datasets, self.DATASET_MANDATORY_KWARGS)
        kwargs = self.transform_input(**kwargs)

    @staticmethod
    def dataset_params_validity_check(datasets, mandatory_kwargs):
        pass

    def transform_input(self, **kwargs):
        """
        Transform and validate input kwargs.

        :param kwargs: Dict;
        :return: Dict;
        """
        return AnalyticsProperties.transform_input(**kwargs)
