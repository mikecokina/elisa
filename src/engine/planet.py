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

        utils.check_missing_kwargs(Planet.KWARGS, kwargs, instance_of=Planet)

        # we already ensured that all kwargs are valid and all mandatory kwargs are present so lets set class attributes
        for kwarg in kwargs:
            self._logger.debug("Setting property {} "
                               "of class instance {} to {}".format(kwarg, Planet.__name__, kwargs[kwarg]))
            setattr(self, kwarg, kwargs[kwarg])