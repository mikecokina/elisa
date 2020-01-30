from abc import ABCMeta, abstractmethod

from elisa.logger import getLogger
from elisa import (
    utils
)
from elisa.conf import config
from elisa.analytics.transform import AnalyticsProperties

logger = getLogger('analytics.base')


class FittingTask(metaclass=ABCMeta):
    """
    Abstract class defining fitting task. This structure aims to provide a framework for solving inverse problem inside
    one object that embeds observed data and fitting methods and provides unified output from fitting methods along with
    capability to visualize the fit.

    :param name: str; arbitrary name of instance
    :param radial_velocities: dict; {component: elisa.analytics.dataset.RVData, ...}
    :param light_curves: dict; {component: elisa.analytics.dataset.LCData, ...}
    """
    ID = 1
    MANDATORY_KWARGS = []
    OPTIONAL_KWARGS = []
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    def __init__(self, name=None, **kwargs):
        # default params
        self.radial_velocities = dict()
        self.light_curves = dict()

        self.rv_fit = None
        self.lc_fit = None

        if utils.is_empty(name):
            self.name = str(FittingTask.ID)
            logger.debug(f"name of class instance {self.__class__.__name__} set to {self.name}")
            self.__class__.ID += 1
        else:
            self.name = str(name)

    @staticmethod
    def transform_input(**kwargs):
        """
        Transform and validate input kwargs.

        :param kwargs: Dict;
        :return: Dict;
        """
        return AnalyticsProperties.transform_input(**kwargs)

    def init_properties(self, **kwargs):
        """
        Setup system properties from input.

        :param kwargs: Dict; all supplied input properties
        """
        logger.debug(f"initialising properties of system {self.name}, values: {kwargs}")
        for kwarg in kwargs:
            setattr(self, kwarg, kwargs[kwarg])


class BinarySystemFittingTask(FittingTask):
    MANDATORY_KWARGS = []
    OPTIONAL_KWARGS = ['radial_velocities', 'light_curves']
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    def __init__(self, name=None, **kwargs):
        # initial validity checks
        utils.invalid_kwarg_checker(kwargs, BinarySystemFittingTask.ALL_KWARGS, self.__class__)
        utils.check_missing_kwargs(BinarySystemFittingTask.MANDATORY_KWARGS, kwargs, instance_of=FittingTask)
        kwargs = self.transform_input(**kwargs)

        super(BinarySystemFittingTask, self).__init__(name, **kwargs)

        # self.plot = graphic.plot.Plot(self)

        self.init_properties(**kwargs)

    # def
