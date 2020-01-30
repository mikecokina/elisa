from ...logger import getLogger
from ... import (
    utils,
)

logger = getLogger('analytics.binary_fit.rv_fit')


class RVFit(object):
    MANDATORY_KWARGS = ['radial_velocities', ]
    OPTIONAL_KWARGS = []
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    def __init__(self, **kwargs):
        utils.invalid_kwarg_checker(kwargs, RVFit.ALL_KWARGS, RVFit)
        utils.check_missing_kwargs(self.__class__.MANDATORY_KWARGS, kwargs, instance_of=self.__class__)
        # kwargs = RVFit.transform_input(**kwargs)

        # values of properties
        logger.debug(f"setting properties of orbit")
        for kwarg in kwargs:
            setattr(self, kwarg, kwargs[kwarg])

    def fit(self, **kwargs):
        pass
