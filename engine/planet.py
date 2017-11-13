from engine.body import Body
from astropy import units as u
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s : [%(levelname)s] : %(name)s : %(message)s')


class Planet(Body):

    KWARGS = ['mass', 't_eff', 'vertices', 'faces', 'normals', 'temperatures', 'synchronicity', 'polar_radius', 'albedo']

    def __init__(self, name=None, **kwargs):
        # get logger
        self._logger = logging.getLogger(Planet.__name__)

        self.is_property(kwargs)
        super(Planet, self).__init__(name=name, **kwargs)

        # default values of properties

        # values of properties
        for kwarg in Planet.KWARGS:
            if kwarg in kwargs:
                self._logger.debug("Setting property {} "
                                   "of class instance {} to {}".format(kwarg, Planet.__name__, kwargs[kwarg]))
                setattr(self, kwarg, kwargs[kwarg])

    @classmethod
    def is_property(cls, kwargs):
        """
        method for checking if keyword arguments are valid properties of this class

        :param kwargs: dict
        :return:
        """
        is_not = ['`{}`'.format(k) for k in kwargs if k not in cls.KWARGS]
        if is_not:
            raise AttributeError('Arguments {} are not valid {} properties.'.format(', '.join(is_not), cls.__name__))
