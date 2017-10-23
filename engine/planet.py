from engine.body import Body
from astropy import units as u
import numpy as np


class Planet(Body):

    KWARGS = ['mass', 't_eff', 'vertices', 'faces', 'normals', 'temperatures', 'synchronicity', 'polar_radius', 'albedo']

    def __init__(self, name=None, **kwargs):
        self.is_property(kwargs)
        super(Planet, self).__init__(name=name, **kwargs)

        # default values of properties

        # values of properties
        for kwarg in Planet.KWARGS:
            if kwarg in kwargs:
                setattr(self, kwarg, kwargs[kwarg])

    @classmethod
    def is_property(cls, kwargs):
        is_not = ['`{}`'.format(k) for k in kwargs if k not in cls.KWARGS]
        if is_not:
            raise AttributeError('Arguments {} are not valid {} properties.'.format(', '.join(is_not), cls.__name__))
