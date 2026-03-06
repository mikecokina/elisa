from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from elisa import units as u
from elisa.base.types import FLOAT
from elisa.graphic import graphics

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from elisa import Observer
    from elisa.observer.passband import PassbandContainer


class PassbandPlot:
    """Plot utilities for passband data.

    Provides methods to generate plots for passband throughput curves.
    """

    def __init__(self, container: PassbandContainer) -> None:
        """Initialize the passband plot instance.

        :param container: Container object with table attribute containing wavelength and throughput data.
        :type container: object
        """
        self.container: PassbandContainer = container

    def passband(self) -> object:
        """Generate a passband throughput plot.

        Creates a plot of the passband throughput as a function of wavelength.
        The plot is generated using the graphics library.

        :returns: Plot figure instance or display depending on graphics configuration.
        :rtype: object
        """
        xs = np.array(self.container.table.wavelength, dtype=FLOAT)
        ys = np.array(self.container.table.throughput, dtype=FLOAT)
        x_unit = r"Wavelength [$\dot{\AA}$]"
        y_unit = "Transmissivity [%/100]"

        kwargs = {
            "xs": xs,
            "ys": ys,
            "x_unit": x_unit,
            "y_unit": y_unit,
            "passband": self.container.passband,
        }
        return graphics.passband_plot(**kwargs)


class Plot:
    """Universal plot interface for Observer class.

    Provides plotting methods for phase curves and radial velocity curves from Observer instances.
    More detailed documentation for each plotting method is available in the graphics library.

    Available methods::

        phase_curve / lc - plots light curves
        rv_curve / rv - plots radial velocity curves

    """

    def __init__(self, observer: Observer) -> None:
        """Initialize the plot interface for an Observer instance.

        :param observer: Observer instance containing phase, flux, and radial velocity data.
        :type observer: object
        """
        self.observer = observer

    def phase_curve(
            self,
            phases: NDArray | None = None,
            fluxes: dict[str, NDArray] | None = None,
            unit: object | None = None,
            legend_location: int = 4,
            *,
            legend: bool = True,
            return_figure_instance: bool = False,
    ) -> Figure:
        """Plot phase curves calculated in the Observer class.

        Generates a plot of flux vs phase for one or more passbands. If phases, fluxes,
        and unit are not provided, values from the Observer instance are used.
        If a unit is provided and differs from the Observer's flux_unit, the fluxes
        are converted to the specified unit.

        :param phases: Phase values corresponding to flux measurements. If None, uses
            phases from the Observer instance. Defaults to None.
        :type phases: NDArray | None
        :param fluxes: Dictionary mapping filter/passband names to flux arrays. If None,
            uses fluxes from the Observer instance. Defaults to None.
        :type fluxes: dict[str, NDArray] | None
        :param unit: Unit for flux values (astropy units). If None, uses flux_unit from
            the Observer instance. Defaults to None.
        :type unit: object | None
        :param legend: Whether to include a legend in the plot. Defaults to True.
        :type legend: bool
        :param legend_location: Matplotlib legend location code (see matplotlib documentation
            for loc argument values). Defaults to 4 (lower right).
        :type legend_location: int
        :param return_figure_instance: If True, returns the Figure instance instead of
            displaying the plot. Defaults to False.
        :type return_figure_instance: bool
        :returns: Plot figure instance if return_figure_instance is True, otherwise displays
            the plot and returns result from graphics.phase_curve.
        :rtype: object
        :raises TypeError: If phases, fluxes, and unit are provided inconsistently
            (either all three or none must be provided).
        """
        if (phases is None) != (fluxes is None) != (unit is None):
            msg = "You have to supply `phases`, `fluxes` and `unit` variables or none of them."
            raise TypeError(msg)

        kwargs = {
            "return_figure_instance": return_figure_instance,
            "phases": self.observer.phases if phases is None else phases,
            "fluxes": self.observer.fluxes if fluxes is None else fluxes,
            "unit": self.observer.flux_unit if unit is None else unit,
            "legend": legend,
            "legend_location": legend_location,
        }
        if isinstance(unit, type(u.W / u.m ** 2)) and fluxes is None:
            for _filter, flux_values in kwargs["fluxes"].items():
                kwargs["fluxes"][_filter] = (flux_values * self.observer.flux_unit).to(unit).value

        return graphics.phase_curve(**kwargs)

    def rv_curve(
            self,
            phases: NDArray | None = None,
            radial_velocities: dict[str, NDArray] | None = None,
            unit: object | None = None,
            legend_location: int = 4,
            *,
            legend: bool = True,
            return_figure_instance: bool = False,
    ) -> Figure:
        """Plot radial velocity curves calculated in the Observer class.

        Generates a plot of radial velocity vs phase for one or more components. If phases,
        radial_velocities, and unit are not provided, values from the Observer instance
        are used. If a unit is provided and differs from the Observer's rv_unit, the
        radial velocities are converted to the specified unit.

        :param phases: Phase values corresponding to radial velocity measurements. If None,
            uses phases from the Observer instance. Defaults to None.
        :type phases: NDArray | None
        :param radial_velocities: Dictionary mapping component names to radial velocity arrays.
            If None, uses radial_velocities from the Observer instance. Defaults to None.
        :type radial_velocities: dict[str, NDArray] | None
        :param unit: Unit for radial velocity values (astropy units). If None, uses rv_unit
            from the Observer instance. Defaults to None.
        :type unit: object | None
        :param legend: Whether to include a legend in the plot. Defaults to True.
        :type legend: bool
        :param legend_location: Matplotlib legend location code (see matplotlib documentation
            for loc argument values). Defaults to 4 (lower right).
        :type legend_location: int
        :param return_figure_instance: If True, returns the Figure instance instead of
            displaying the plot. Defaults to False.
        :type return_figure_instance: bool
        :returns: Plot figure instance if return_figure_instance is True, otherwise displays
            the plot and returns result from graphics.rv_curve.
        :rtype: object
        :raises TypeError: If phases, radial_velocities, and unit are provided inconsistently
            (either all three or none must be provided).
        """
        if (phases is None) != (radial_velocities is None) != (unit is None):
            msg = "You have to supply both `phases` `fluxes`, `radial_velocities` and `None` or none of them."
            raise TypeError(msg)

        kwargs = {
            "return_figure_instance": return_figure_instance,
            "phases": self.observer.phases if phases is None else phases,
            "rvs": self.observer.radial_velocities if radial_velocities is None else radial_velocities,
            "unit": self.observer.rv_unit if unit is None else unit,
            "legend": legend,
            "legend_location": legend_location,
        }

        if isinstance(unit, type(u.km / u.s)) and radial_velocities is None:
            for component, rvs in kwargs["rvs"].items():
                kwargs["rvs"][component] = (rvs * self.observer.rv_unit).to(unit).value

        return graphics.rv_curve(**kwargs)

    rv = rv_curve
    lc = phase_curve
