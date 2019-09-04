import numpy as np
from astropy import units as u

from elisa import utils, const, graphics, units


class Plot(object):
    """
    universal plot interface for Observer class, more detailed documentation for each value of descriptor is
    available in graphics library

                        `orbit` - plots orbit in orbital plane
    :return:
    """

    def __init__(self, instance):
        self._self = instance

    def light_curve(self, **kwargs):
        phases = kwargs.get('phases', None)
        curves = kwargs.get('curves', None)

        if phases is None:
            ValueError('Light curve phases were not specified.')
        if curves is None:
            ValueError('Light curves were not supplied.')

        graphics.light_curve(**kwargs)

