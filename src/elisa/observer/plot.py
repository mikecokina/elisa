from elisa import graphics


class Plot(object):
    """
    Universal plot interface for Observer class, more detailed documentation for each value of descriptor is
    available in graphics library.

    Available methods::

        `orbit` - plots orbit in orbital plane

    """

    def __init__(self, observer):
        self.observer = observer

    def phase_curve(self, **kwargs):
        """
        Function plots phase curves calculated in Observer class.

        :param kwargs: Dict;
        :**kwargs options**:
            * **phases** * -- numpy.array;
            * **fluxes** * -- Dict;
            * **flux_unit** * -- astropy.units.quantity.Quantity; unit of flux measurements,
            * **legend** * -- bool; on/off,
            * **legend_location** * -- int;
        """
        kwargs['phases'] = kwargs.get('phases', self.observer.phases)
        kwargs['fluxes'] = kwargs.get('fluxes', self.observer.fluxes)
        kwargs['flux_unit'] = kwargs.get('flux_unit', self.observer.fluxes_unit)
        kwargs['legend'] = kwargs.get('legend', True)
        kwargs['legend_location'] = kwargs.get('legend_location', 4)

        graphics.phase_curve(**kwargs)

    def rv_curve(self, phases, primary_rv, secondary_rv, unit=None, legend=True, legend_location=4):
        """
        Function plots radial velocity curves calculated in Observer class.

        :param phases: numpy.array;
        :param primary_rv: numpy.array;
        :param secondary_rv: numpy.array;
        :param unit: Union[None, astropy.units.quantity.Quantity];
        :param legend: bool;
        :param legend_location: int;
        """

        kwargs = {
            "phases": phases,
            "primary_rv": primary_rv,
            "secondary_rv": secondary_rv,
            "unit": unit or self.observer.rv_unit,
            "legend": legend,
            "legend_location": legend_location
        }
        graphics.rv_curve(**kwargs)
