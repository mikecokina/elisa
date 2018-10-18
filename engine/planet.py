from engine.body import Body
from astropy import units as u
import numpy as np
import logging
from engine import utils


class Planet(Body):

    KWARGS = ['mass', 't_eff', 'vertices', 'faces', 'normals', 'temperatures', 'synchronicity', 'polar_radius',
              'albedo']
    OPTIONAL_KWARGS = []
    ALL_KWARGS = KWARGS + OPTIONAL_KWARGS

    def __init__(self, name=None, **kwargs):
        # get logger
        self._logger = logging.getLogger(Planet.__name__)

        utils.invalid_kwarg_checker(kwargs, Planet.ALL_KWARGS, Planet)
        super(Planet, self).__init__(name=name, **kwargs)

        # default values of properties

        # values of properties
        for kwarg in Planet.KWARGS:
            if kwarg in kwargs:
                self._logger.debug("Setting property {} "
                                   "of class instance {} to {}".format(kwarg, Planet.__name__, kwargs[kwarg]))
                setattr(self, kwarg, kwargs[kwarg])
