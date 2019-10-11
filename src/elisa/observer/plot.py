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

    def phase_curve(self, **kwargs):
        """
        function plots phase curves calculated in Observer class

        :param kwargs: dict; `phases` (array), `fluxes` (dict), `flux_unit` - unit of flux measurements,
        `legend` (bool) onn/off, `legend_location` (int)
        :return:
        """
        kwargs['phases'] = kwargs.get('phases', self._self.phases)
        kwargs['fluxes'] = kwargs.get('fluxes', self._self.fluxes)
        kwargs['flux_unit'] = kwargs.get('flux_unit', self._self.fluxes_unit)
        kwargs['legend'] = kwargs.get('legend', True)
        kwargs['legend_location'] = kwargs.get('legend', 4)

        graphics.phase_curve(**kwargs)

