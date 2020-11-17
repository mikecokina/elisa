from elisa.graphic import graphics
from .. import units as u


class Plot(object):
    """
    Universal plot interface for Observer class, more detailed documentation for each value of descriptor is
    available in graphics library.

    Available methods::

        `orbit` - plots orbit in orbital plane

    """

    def __init__(self, observer):
        self.observer = observer

    def phase_curve(self, phases=None, fluxes=None, unit=None, legend=True, legend_location=4):
        """
        Function plots phase curves calculated in Observer class.

        :param phases: numpy.ndarray;
        :param fluxes: dict; fluxes in each passband
        :param unit: Union[NoneType, astropy.units]; units of flux
        :param legend: bool;
        :param legend_location: int; wrapper for matplotlib `loc` argument for legend location
        """
        if (phases is None) != (fluxes is None) != (unit is None):
            raise TypeError('You have to supply `phases`, `fluxes` and `unit` variables or none of them.')

        kwargs = {
            "phases": self.observer.phases if phases is None else phases,
            "fluxes": self.observer.fluxes if fluxes is None else fluxes,
            "unit": self.observer.fluxes_unit if unit is None else unit,
            "legend": legend,
            "legend_location": legend_location
        }
        if isinstance(unit, type(u.W/u.m**2)) and fluxes is None:
            for _filter, fluxes in kwargs['fluxes'].items():
                kwargs['fluxes'][_filter] = (fluxes*self.observer.fluxes_unit).to(unit).value

        graphics.phase_curve(**kwargs)

    def rv_curve(self, phases=None, radial_velocities=None, unit=None, legend=True, legend_location=4):
        """
        Function plots radial velocity curves calculated in Observer class.

        :param phases: numpy.array;
        :param radial_velocities: dict;
        :param unit: Union[None, astropy.units.quantity.Quantity]; unit of input 'radial_velocities', if they are not
                     supplied, values calculated in Observer instance are converted to `unit`
        :param legend: bool;
        :param legend_location: int;
        """
        if (phases is None) != (radial_velocities is None) != (unit is None):
            raise TypeError('You have to supply both `phases` `fluxes`, `radial_velocities` and `None` or none of '
                            'them.')

        kwargs = {
            "phases": self.observer.phases if phases is None else phases,
            'rvs': self.observer.radial_velocities if radial_velocities is None else radial_velocities,
            "unit": self.observer.rv_unit if unit is None else unit,
            "legend": legend,
            "legend_location": legend_location
        }

        if isinstance(unit, type(u.km/u.s)) and radial_velocities is None:
            for component, rvs in kwargs['rvs'].items():
                kwargs['rvs'][component] = (rvs*self.observer.rv_unit).to(unit).value

        graphics.rv_curve(**kwargs)

    rv = rv_curve
    lc = phase_curve
