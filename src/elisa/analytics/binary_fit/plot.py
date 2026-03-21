from __future__ import annotations

import re
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
from emcee.autocorr import function_1d, integrated_time
from scipy.interpolate import interp1d

from elisa import units as u
from elisa.analytics.binary_fit import shared
from elisa.analytics.binary_fit.mixins import MCMCMixin
from elisa.analytics.models.lc import synthetic_binary
from elisa.analytics.models.rv import central_rv_synthetic
from elisa.analytics.params import conf, parameters
from elisa.binary_system import t_layer
from elisa.binary_system.system import BinarySystem
from elisa.graphic import graphics
from elisa.graphic.mcmc_graphics import Plot as MCMCPlot
from elisa.logger import getLogger
from elisa.observer.observer import Observer
from elisa.observer.utils import normalize_light_curve
from elisa.utils import is_empty

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from astropy.units import UnitBase
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from elisa.analytics.binary_fit.lc_fit import LCFitLeastSquares, LCFitMCMC
    from elisa.analytics.binary_fit.rv_fit import RVFitLeastSquares, RVFitMCMC
    from elisa.analytics.dataset.base import LCData, RVData
    from elisa.types import Float, Int


logger = getLogger("analytics.binary_fit.plot")

PLOT_UNITS: dict[str, UnitBase] = {
    "system@asini": u.solRad,
    "system@argument_of_periastron": u.degree,
    "system@gamma": u.km / u.s,
    "system@primary_minimum_time": u.d,
}


class MCMCPlotMixin:
    """Graphics mixin for visualization of MCMC sampling results."""

    fit = None

    def corner(
        self,
        flat_chain: NDArray[Float] | None = None,
        variable_labels: Sequence[str] | None = None,
        normalization: dict[str, tuple[Float, Float]] | None = None,
        quantiles: Iterable[Float] | None = None,
        plot_units: dict[str, UnitBase] | None = None,
        sigma: Float = 5,
        n_bins: Int = 20,
        *,
        truths: bool | Sequence[Float] = False,
        show_titles: bool = True,
        sigma_clip: bool = False,
        **kwargs,
    ) -> None:
        """Plot a complete corner plot for posterior MCMC samples.

        This method is useful for visualizing the posterior distribution of the
        sampled parameters.

        :param flat_chain: Flattened chain of all parameters. If omitted, the
            internal ``flat_chain`` attribute is used.
        :type flat_chain: NDArray[elisa.types.Float] | None
        :param variable_labels: Variable names corresponding to columns in
            ``flat_chain``. Use only with a custom chain.
        :type variable_labels: Sequence[str] | None
        :param normalization: Per-variable normalization boundaries used to
            reconstruct real values from a normalized custom chain.
        :type normalization: dict[str, tuple[elisa.types.Float, elisa.types.Float]] | None
        :param quantiles: Fractional quantiles drawn as vertical dashed lines on
            one-dimensional histograms.
        :type quantiles: Iterable[elisa.types.Float] | None
        :param truths: If ``True``, fitted values are shown. If ``False``, no
            truths are shown. A sequence behaves like the equivalent argument in
            the corner plotting backend.
        :type truths: bool | Sequence[elisa.types.Float]
        :param show_titles: Whether to show titles with parameter names, values,
            errors, and units.
        :type show_titles: bool
        :param plot_units: Units used for displayed output values.
        :type plot_units: dict[str, astropy.units.UnitBase] | None
        :param sigma_clip: Whether to crop posterior samples around the fitted
            value within a symmetric sigma interval.
        :type sigma_clip: bool
        :param sigma: Width of the sigma-clipping interval in multiples of the
            confidence width.
        :type sigma: elisa.types.Float
        :param n_bins: Number of bins in each histogram.
        :type n_bins: elisa.types.Int
        """
        corner(
            self.fit,
            flat_chain=flat_chain,
            variable_labels=variable_labels,
            normalization=normalization,
            quantiles=quantiles,
            truths=truths,
            show_titles=show_titles,
            plot_units=plot_units,
            sigma_clip=sigma_clip,
            sigma=sigma,
            n_bins=n_bins,
            **kwargs,
        )

    def autocorrelation(
        self,
        correlations_to_plot: Sequence[str] | None = None,
        flat_chain: NDArray[Float] | None = None,
        variable_labels: Sequence[str] | None = None,
    ) -> None:
        """Plot the autocorrelation function for the MCMC chain.

        :param correlations_to_plot: Names of variables whose autocorrelation
            functions should be displayed.
        :type correlations_to_plot: Sequence[str] | None
        :param flat_chain: Optional flattened chain of all parameters. If
            omitted, the internal ``flat_chain`` attribute is used.
        :type flat_chain: NDArray[elisa.types.Float] | None
        :param variable_labels: Variable names corresponding to columns in
            ``flat_chain``. Use only with a custom chain.
        :type variable_labels: Sequence[str] | None
        """
        autocorrelation(
            self.fit,
            correlations_to_plot=correlations_to_plot,
            flat_chain=flat_chain,
            variable_labels=variable_labels,
        )

    def traces(
        self,
        traces_to_plot: Sequence[str] | None = None,
        flat_chain: NDArray[Float] | None = None,
        variable_labels: Sequence[str] | None = None,
        normalization: dict[str, tuple[Float, Float]] | None = None,
        plot_units: dict[str, UnitBase] | None = None,
        *,
        truths: bool = False,
    ) -> None:
        """Plot parameter traces of the MCMC samples.

        :param traces_to_plot: Names of variables whose traces should be
            displayed.
        :type traces_to_plot: Sequence[str] | None
        :param flat_chain: Optional flattened chain of all parameters. If
            omitted, the internal ``flat_chain`` attribute is used.
        :type flat_chain: NDArray[elisa.types.Float] | None
        :param variable_labels: Variable names corresponding to columns in
            ``flat_chain``. Use only with a custom chain.
        :type variable_labels: Sequence[str] | None
        :param normalization: Per-variable normalization boundaries used to
            reconstruct real values from a normalized custom chain.
        :type normalization: dict[str, tuple[elisa.types.Float, elisa.types.Float]] | None
        :param plot_units: Units used for displayed output values.
        :type plot_units: dict[str, astropy.units.UnitBase] | None
        :param truths: Whether to indicate fitted values. This does not work
            with a custom chain.
        :type truths: bool
        """
        traces(
            self.fit,
            traces_to_plot=traces_to_plot,
            flat_chain=flat_chain,
            variable_labels=variable_labels,
            normalization=normalization,
            plot_units=plot_units,
            truths=truths,
        )


class RVPlot:
    """Graphics helper for RV fit visualization."""

    def __init__(
        self,
        instance: RVFitLeastSquares | RVFitMCMC,
        data: dict[str, RVData],
    ) -> None:
        """Initialize RV plot helper.

        :param instance: Fit instance owning the plotting helper.
        :type instance: (elisa.analytics.binary_fit.rv_fit.RVFitLeastSquares |
                         elisa.analytics.binary_fit.rv_fit.RVFitMCMC)
        :param data: Observational RV data.
        :type data: dict[str, elisa.analytics.dataset.base.RVData]
        """
        self.fit = instance
        self.data = data

    def model(
        self,
        start_phase: Float = -0.6,
        stop_phase: Float = 0.6,
        number_of_points: Int = 300,
        y_axis_unit: UnitBase = u.km / u.s,
        *,
        return_figure_instance: bool = False,
        **kwargs,
    ) -> Figure | None:
        """Plot the RV model described by fit parameters.

        The model is taken either from supplied ``fit_result`` or from the last
        run of the fitting procedure stored on the instance.

        :param start_phase: Initial orbital phase of synthetic observations.
        :type start_phase: elisa.types.Float
        :param stop_phase: Final orbital phase of synthetic observations.
        :type stop_phase: elisa.types.Float
        :param number_of_points: Number of synthetic model points.
        :type number_of_points: elisa.types.Int
        :param y_axis_unit: Unit used for the vertical axis.
        :type y_axis_unit: astropy.units.UnitBase
        :param return_figure_instance: If ``True``, return the figure instance
            instead of displaying it.
        :type return_figure_instance: bool
        :param kwargs: Additional plotting options. Supported key:
            ``fit_result``.
        :type kwargs: dict[str, object]
        :return: Figure instance if return_figure_instance is True.
        :rtype: matplotlib.figure.Figure | None
        """
        logger.debug("Producing/retrieving data for RV plot.")
        plot_result_kwargs: dict[str, object] = {}
        fit_result: dict = kwargs.get("fit_result", self.fit.result)

        if fit_result is None:
            msg = (
                "You did not performed radial velocity fit on this instance "
                "or you did not provided result parameter dictionary."
            )
            raise ValueError(msg)

        x_data: dict[str, NDArray[Float]] = {}
        y_data: dict[str, NDArray[Float]] = {}
        y_err: dict[str, NDArray[Float] | None] = {}

        for component, data_obj in self.data.items():
            if data_obj.x_unit is u.dimensionless_unscaled:
                x_data[component] = t_layer.adjust_phases(
                    phases=data_obj.x_data,
                    centre=0.0,
                )
            else:
                x_data[component] = t_layer.jd_to_phase(
                    fit_result["system"]["primary_minimum_time"]["value"],
                    fit_result["system"]["period"]["value"],
                    data_obj.x_data,
                    centre=0.0,
                )

            y_data[component] = (
                (data_obj.y_data * data_obj.y_unit).to(y_axis_unit).value
            )
            if data_obj.y_err is not None:
                err_value: NDArray[Float] = (  # type: ignore[assignment]
                    (data_obj.y_err * data_obj.y_unit).to(y_axis_unit).value  # type: ignore[operator]
                )
                y_err[component] = err_value
            else:
                y_err[component] = None

        x_data, y_data, y_err = shared.extend_observations_to_desired_interval(
            start_phase,
            stop_phase,
            x_data,
            y_data,
            y_err,
        )

        plot_result_kwargs.update(
            {
                "x_data": x_data,
                "y_data": y_data,
                "y_err": y_err,
                "y_unit": y_axis_unit,
            },
        )

        kwargs_to_replot = parameters.deserialize_result(fit_result)
        kwargs_to_replot = {key: val["value"] for key, val in kwargs_to_replot.items()}
        kwargs_to_replot.pop("system@primary_minimum_time", None)

        synth_phases = np.linspace(start_phase, stop_phase, number_of_points)
        rv_fit = central_rv_synthetic(synth_phases, Observer(), **kwargs_to_replot)
        rv_fit = {component: (data * u.VELOCITY_UNIT).to(y_axis_unit).value for component, data in rv_fit.items()}

        interp_fn = {component: interp1d(synth_phases, rv_fit[component]) for component in self.data}
        residuals = {component: y_data[component] - interp_fn[component](x_data[component]) for component in self.data}

        plot_result_kwargs.update(
            {
                "return_figure_instance": return_figure_instance,
                "synth_phases": synth_phases,
                "rv_fit": rv_fit,
                "residuals": residuals,
                "y_unit": y_axis_unit,
            },
        )

        logger.debug("Sending data to matplotlib interface.")
        return graphics.binary_rv_fit_plot(**plot_result_kwargs)


class RVPlotLsqr(RVPlot):
    """Least-squares RV plot helper."""

    def __init__(self, instance: RVFitLeastSquares, data: dict[str, RVData]) -> None:
        """Initialize least-squares RV plot helper."""
        super().__init__(instance, data)


class RVPlotMCMC(RVPlot, MCMCPlotMixin):
    """MCMC RV plot helper."""

    def __init__(self, instance: RVFitMCMC, data: dict[str, RVData]) -> None:
        """Initialize MCMC RV plot helper."""
        super().__init__(instance, data)


class LCPlot:
    """Graphics functions for visualization of LC fit results."""

    def __init__(
        self,
        instance: LCFitMCMC | LCFitLeastSquares,
        data: dict[str, LCData],
    ) -> None:
        """Initialize LC plot helper.

        :param instance: Fit instance owning the plotting helper.
        :type instance: (elisa.analytics.binary_fit.lc_fit.LCFitMCMC |
                         elisa.analytics.binary_fit.lc_fit.LCFitLeastSquares)
        :param data: Observational LC data.
        :type data: dict[str, elisa.analytics.dataset.base.LCData]
        """
        self.fit = instance
        self.data = data

    def model(
        self,
        start_phase: Float = -0.6,
        stop_phase: Float = 0.6,
        number_of_points: Int = 300,
        discretization: Float = 5,
        separation: Float = 0.1,
        data_frac_to_normalize: Float = 0.1,
        normalization_kind: str = "maximum",
        loc: Int = 1,
        *,
        plot_legend: bool = True,
        return_figure_instance: bool = False,
        rasterize: bool | None = None,
        **kwargs,
    ) -> Figure | None:
        """Prepare data for plotting the LC model described by fit parameters.

        The model is taken either from supplied ``fit_result`` or from the last
        run of the fitting procedure stored on the instance.

        :param start_phase: Initial orbital phase of synthetic observations.
        :type start_phase: elisa.types.Float
        :param stop_phase: Final orbital phase of synthetic observations.
        :type stop_phase: elisa.types.Float
        :param number_of_points: Number of synthetic model points.
        :type number_of_points: elisa.types.Int
        :param discretization: Discretization factor used during synthetic
            observation calculation.
        :type discretization: elisa.types.Float
        :param separation: Vertical separation between passbands in the plotted
            normalized curves.
        :type separation: elisa.types.Float
        :param data_frac_to_normalize: Fraction of the highest-flux data points
            used during normalization.
        :type data_frac_to_normalize: elisa.types.Float
        :param normalization_kind: Normalization mode, typically ``"average"``
            or ``"maximum"``.
        :type normalization_kind: str
        :param plot_legend: Whether to display the legend.
        :type plot_legend: bool
        :param loc: Legend location.
        :type loc: elisa.types.Int
        :param return_figure_instance: If ``True``, return the figure instance
            instead of displaying it.
        :type return_figure_instance: bool
        :param rasterize: Whether the figure should be rasterized.
        :type rasterize: bool | None
        :param kwargs: Additional plotting options. Supported key:
            ``fit_result``.
        :type kwargs: dict[str, object]
        :return: Figure instance if return_figure_instance is True.
        :rtype: matplotlib.figure.Figure | None
        """
        logger.debug("Producing/retrieving data for LC plot.")
        average_kind = normalization_kind
        plot_result_kwargs: dict[str, object] = {}
        fit_result: dict = kwargs.get("fit_result", self.fit.result)

        if fit_result is None:
            msg = "You did not performed light curve fit on this instance or you did not provided parameter dictionary."
            raise ValueError(msg)

        x_data: dict[str, NDArray[Float]] = {}
        y_data: dict[str, NDArray[Float]] = {}
        y_err: dict[str, NDArray[Float] | None] = {}

        for band, data in self.data.items():
            if data.x_unit is u.dimensionless_unscaled:
                x_data[band] = t_layer.adjust_phases(
                    phases=data.x_data,
                    centre=0.0,
                )
            else:
                x_data[band] = t_layer.jd_to_phase(
                    fit_result["system"]["primary_minimum_time"]["value"],
                    fit_result["system"]["period"]["value"],
                    data.x_data,
                    centre=0.0,
                )
            y_data[band] = data.y_data
            y_err[band] = data.y_err

        y_data, y_err = normalize_light_curve(
            y_data,
            y_err,
            kind=average_kind,
            top_fraction_to_average=data_frac_to_normalize,
        )

        y_len = len(y_data)
        for idx, curve in enumerate(y_data.values()):
            curve -= separation * (idx - int(y_len / 2))  # noqa: PLW2901

        for band in self.data:
            phases_extended = np.concatenate((x_data[band] - 1.0, x_data[band], x_data[band] + 1.0))
            phases_extended_filter = np.logical_and(
                start_phase < phases_extended,
                phases_extended < stop_phase,
            )
            x_data[band] = phases_extended[phases_extended_filter]
            y_data[band] = np.tile(y_data[band], 3)[phases_extended_filter]

            if not is_empty(y_err[band]):
                y_err[band] = np.tile(y_err[band], 3)[phases_extended_filter]

        x_data, y_data, y_err = shared.extend_observations_to_desired_interval(
            start_phase,
            stop_phase,
            x_data,
            y_data,
            y_err,
        )

        plot_result_kwargs.update(
            {
                "x_data": x_data,
                "y_data": y_data,
                "y_err": y_err,
            },
        )

        kwargs_to_replot = parameters.deserialize_result(fit_result)
        kwargs_to_replot = {key: val["value"] for key, val in kwargs_to_replot.items()}
        kwargs_to_replot.pop("system@primary_minimum_time", None)

        kwargs_to_replot = parameters.extend_json_with_atm_params(
            kwargs_to_replot,
            atmosphere_model=self.fit.fit_method_instance.atmosphere_model,
            limb_darkening_coefficients=self.fit.fit_method_instance.limb_darkening_coefficients,
        )

        synth_phases = np.linspace(start_phase, stop_phase, number_of_points)
        observer = Observer(passband=[*self.data.keys()], system=None)
        observer._system_cls = BinarySystem  # noqa: SLF001

        lc_fit = synthetic_binary(
            synth_phases,
            discretization,
            observer,
            **kwargs_to_replot,
        )
        lc_fit, _ = normalize_light_curve(
            lc_fit,
            kind=average_kind,
            top_fraction_to_average=0.001,
        )

        for idx, curve in enumerate(lc_fit.values()):
            curve -= separation * (idx - int(y_len / 2))  # noqa: PLW2901

        interp_fn = {
            band: interp1d(synth_phases, lc_fit[band], kind="cubic")
            for band in self.data  # type: ignore[assignment]
        }
        residuals: dict[str, NDArray[Float]] = {
            band: (
                y_data[band]
                - np.mean(y_data[band])
                - interp_fn[band](x_data[band])
                + np.mean(interp_fn[band](x_data[band]))
            )
            for band in self.data  # type: ignore[assignment]
        }

        plot_result_kwargs.update(
            {
                "return_figure_instance": return_figure_instance,
                "synth_phases": synth_phases,
                "lcs": lc_fit,
                "residuals": residuals,
                "legend": plot_legend,
                "loc": loc,
                "rasterize": rasterize,
            },
        )

        logger.debug("Sending data to matplotlib interface.")
        return graphics.binary_lc_fit_plot(**plot_result_kwargs)


class LCPlotLsqr(LCPlot):
    """Least-squares LC plot helper."""

    def __init__(self, instance: LCFitLeastSquares, data: dict[str, LCData]) -> None:
        """Initialize least-squares LC plot helper."""
        super().__init__(instance, data)


class LCPlotMCMC(LCPlot, MCMCPlotMixin):
    """MCMC LC plot helper."""

    def __init__(self, instance: LCFitMCMC, data: dict[str, LCData]) -> None:
        """Initialize MCMC LC plot helper."""
        super().__init__(instance, data)


def serialize_plot_labels(variable_labels: Sequence[str]) -> list[str]:
    """Return TeX-compatible labels of model parameters.

    :param variable_labels: Flat-format labels of model parameters, for example
        ``system@inclination``.
    :type variable_labels: Sequence[str]
    :returns: Plot labels.
    :rtype: list[str]
    """
    labels: list[str] = []
    composite_pattern = "|".join(conf.COMPOSITE_FLAT_PARAMS)

    for lbl in variable_labels:
        lbl_s = lbl.split(conf.PARAM_PARSER)

        if re.search(composite_pattern, lbl):
            labels.append(f"{lbl_s[-2]} {conf.PARAMS_KEY_TEX_MAP[lbl_s[-1]]}")
        else:
            labels.append(conf.PARAMS_KEY_TEX_MAP[lbl])

    return labels


def corner(
    mcmc_fit_instance: RVFitMCMC | LCFitMCMC,
    flat_chain: NDArray[Float] | None = None,
    variable_labels: Sequence[str] | None = None,
    normalization: dict[str, tuple[Float, Float]] | None = None,
    quantiles: Iterable[Float] | None = None,
    plot_units: dict[str, UnitBase] | None = None,
    sigma: Float = 5,
    n_bins: Int = 20,
    *,
    sigma_clip: bool = False,
    truths: bool | Sequence[Float] = False,
    show_titles: bool = True,
    **kwargs,
) -> None:
    """Plot a complete corner plot from supplied parameters.

    :param mcmc_fit_instance: MCMC fit instance.
    :type mcmc_fit_instance: object
    :param flat_chain: Flattened chain of all parameters. If omitted, the
        internal ``flat_chain`` attribute is used.
    :type flat_chain: NDArray[elisa.types.Float] | None
    :param variable_labels: Variable names corresponding to columns in
        ``flat_chain``. Use only with a custom chain.
    :type variable_labels: Sequence[str] | None
    :param normalization: Per-variable normalization boundaries used to
        reconstruct real values from a normalized custom chain.
    :type normalization: dict[str, tuple[elisa.types.Float, elisa.types.Float]] | None
    :param quantiles: Fractional quantiles drawn as vertical dashed lines on
        one-dimensional histograms.
    :type quantiles: Iterable[elisa.types.Float] | None
    :param truths: If ``True``, fitted values are shown. If ``False``, no
        truths are shown. A sequence behaves like the equivalent argument in
        the corner plotting backend.
    :type truths: bool | Sequence[elisa.types.Float]
    :param show_titles: Whether to show titles with parameter names, values,
        errors, and units.
    :type show_titles: bool
    :param plot_units: Units used for displayed output values.
    :type plot_units: dict[str, astropy.units.UnitBase] | None
    :param sigma_clip: Whether to crop posterior samples around the fitted
        value within a symmetric sigma interval.
    :type sigma_clip: bool
    :param sigma: Width of the sigma-clipping interval in multiples of the
        confidence width.
    :type sigma: elisa.types.Float
    :param n_bins: Number of bins in each histogram.
    :type n_bins: elisa.types.Int
    """
    logger.debug("Producing/retrieving data for corner plot.")
    flat_chain = deepcopy(mcmc_fit_instance.flat_chain) if flat_chain is None else deepcopy(flat_chain)
    variable_labels = mcmc_fit_instance.variable_labels if variable_labels is None else variable_labels
    normalization = mcmc_fit_instance.normalization if normalization is None else normalization
    quantiles = [0.16, 0.5, 0.84] if quantiles is None else quantiles
    flat_result = deepcopy(mcmc_fit_instance.flat_result)

    if flat_chain is None:
        msg = "You can use corner plot after running mcmc method or after loading the flat chain."
        raise ValueError(msg)

    plot_labels = serialize_plot_labels(variable_labels)
    flat_chain = MCMCMixin.renormalize_flat_chain(
        flat_chain,
        mcmc_fit_instance.variable_labels,
        variable_labels,
        normalization,
    )

    flat_chain_reduced = np.empty((flat_chain.shape[0], len(variable_labels)))
    plot_units = PLOT_UNITS if plot_units is None else plot_units

    for ii, lbl in enumerate(variable_labels):
        if lbl in plot_units:
            unit = u.Unit(flat_result[lbl]["unit"])
            flat_chain_reduced[:, ii] = (flat_chain[:, ii] * unit).to(plot_units[lbl]).value
            flat_result[lbl]["value"] = (flat_result[lbl]["value"] * unit).to(plot_units[lbl]).value
            flat_result[lbl]["confidence_interval"]["min"] = (
                (flat_result[lbl]["confidence_interval"]["min"] * unit).to(plot_units[lbl]).value
            )
            flat_result[lbl]["confidence_interval"]["max"] = (
                (flat_result[lbl]["confidence_interval"]["max"] * unit).to(plot_units[lbl]).value
            )
            flat_result[lbl]["unit"] = plot_units[lbl].to_string()
        else:
            flat_chain_reduced[:, ii] = flat_chain[:, ii]

    if truths is True:
        truths_values = [flat_result[lbl]["value"] for lbl in variable_labels]
    elif truths is False:
        truths_values = None
    else:
        truths_values = truths

    if sigma_clip:
        for ii, lbl in enumerate(variable_labels):
            tol = (
                0.5
                * sigma
                * np.abs(
                    flat_result[lbl]["confidence_interval"]["max"] - flat_result[lbl]["confidence_interval"]["min"],
                )
            )
            mask = np.logical_and(
                flat_chain_reduced[:, ii] > flat_result[lbl]["value"] - tol,
                flat_chain_reduced[:, ii] < flat_result[lbl]["value"] + tol,
            )
            flat_chain_reduced = flat_chain_reduced[mask]

    corner_plot_kwargs = {
        "flat_chain": flat_chain_reduced,
        "truths": truths_values,
        "variable_labels": variable_labels,
        "labels": plot_labels,
        "quantiles": quantiles,
        "show_titles": show_titles,
        "fit_params": flat_result,
        "bins": n_bins,
    }
    corner_plot_kwargs.update(**kwargs)

    logger.debug("Sending data to matplotlib interface.")
    MCMCPlot.corner(**corner_plot_kwargs)


def autocorrelation(
    mcmc_fit_instance: RVFitMCMC | LCFitMCMC,
    correlations_to_plot: Sequence[str] | None = None,
    flat_chain: NDArray[Float] | None = None,
    variable_labels: Sequence[str] | None = None,
) -> None:
    """Plot autocorrelation functions of selected parameters.

    :param mcmc_fit_instance: MCMC fit instance.
    :type mcmc_fit_instance: RVFitMCMC | LCFitMCMC
    :param correlations_to_plot: Names of variables whose autocorrelation
        functions should be displayed.
    :type correlations_to_plot: Sequence[str] | None
    :param flat_chain: Optional flattened chain of all parameters. If omitted,
        the internal ``flat_chain`` attribute is used.
    :type flat_chain: NDArray[elisa.types.Float] | None
    :param variable_labels: Variable names corresponding to columns in
        ``flat_chain``. Use only with a custom chain.
    :type variable_labels: Sequence[str] | None
    """
    flat_chain = deepcopy(mcmc_fit_instance.flat_chain) if flat_chain is None else deepcopy(flat_chain)
    variable_labels = mcmc_fit_instance.variable_labels if variable_labels is None else variable_labels
    correlations_to_plot = variable_labels if correlations_to_plot is None else correlations_to_plot

    if flat_chain is None:
        msg = "You can use trace plot only in case of mcmc method or for some reason the flat chain was not found."
        raise ValueError(msg)

    labels = serialize_plot_labels(variable_labels)

    n_params = len(variable_labels)
    autocorr_fns = np.empty((flat_chain.shape[0], n_params))
    autocorr_time = np.empty(n_params)

    for idx, _lbl in enumerate(variable_labels):
        autocorr_fns[:, idx] = function_1d(flat_chain[:, idx])
        autocorr_time[idx] = integrated_time(flat_chain[:, idx], quiet=True)

    autocorr_plot_kwargs = {
        "correlations_to_plot": correlations_to_plot,
        "autocorr_fns": autocorr_fns,
        "autocorr_time": autocorr_time,
        "variable_labels": variable_labels,
        "labels": labels,
    }

    MCMCPlot.autocorr(**autocorr_plot_kwargs)


def traces(
    mcmc_fit_instance: RVFitMCMC | LCFitMCMC,
    traces_to_plot: Sequence[str] | None = None,
    flat_chain: NDArray[Float] | None = None,
    variable_labels: Sequence[str] | None = None,
    normalization: dict[str, tuple[Float, Float]] | None = None,
    plot_units: dict[str, UnitBase] | None = None,
    *,
    truths: bool = False,
) -> None:
    """Plot traces of selected parameters.

    :param mcmc_fit_instance: MCMC fit instance.
    :type mcmc_fit_instance: RVFitMCMC | LCFitMCMC
    :param traces_to_plot: Names of variables whose traces should be displayed.
    :type traces_to_plot: Sequence[str] | None
    :param flat_chain: Optional flattened chain of all parameters. If omitted,
        the internal ``flat_chain`` attribute is used.
    :type flat_chain: NDArray[elisa.types.Float] | None
    :param variable_labels: Variable names corresponding to columns in
        ``flat_chain``. Use only with a custom chain.
    :type variable_labels: Sequence[str] | None
    :param normalization: Per-variable normalization boundaries used to
        reconstruct real values from a normalized custom chain.
    :type normalization: dict[str, tuple[elisa.types.Float, elisa.types.Float]] | None
    :param plot_units: Units used for displayed output values.
    :type plot_units: dict[str, astropy.units.UnitBase] | None
    :param truths: Whether to indicate fitted values. This does not work with a
        custom chain.
    :type truths: bool
    """
    logger.debug("Producing/retrieving data for traces plot.")

    variable_labels = mcmc_fit_instance.variable_labels if variable_labels is None else variable_labels
    normalization = mcmc_fit_instance.normalization if normalization is None else normalization
    flat_chain = deepcopy(mcmc_fit_instance.flat_chain) if flat_chain is None else deepcopy(flat_chain)
    flat_result = deepcopy(mcmc_fit_instance.flat_result)

    if flat_chain is None:
        msg = "You can use trace plot only in case of mcmc method or for some reason the flat chain was not found."
        raise ValueError(msg)

    flat_chain = MCMCMixin.renormalize_flat_chain(
        flat_chain,
        mcmc_fit_instance.variable_labels,
        variable_labels,
        normalization,
    )
    labels = serialize_plot_labels(variable_labels)

    plot_units = PLOT_UNITS if plot_units is None else plot_units
    for ii, lbl in enumerate(variable_labels):
        if lbl in plot_units:
            unit = u.Unit(flat_result[lbl]["unit"])
            flat_chain[:, ii] = (flat_chain[:, ii] * unit).to(plot_units[lbl]).value
            flat_result[lbl]["value"] = (flat_result[lbl]["value"] * unit).to(plot_units[lbl]).value
            flat_result[lbl]["confidence_interval"]["min"] = (
                (flat_result[lbl]["confidence_interval"]["min"] * unit).to(plot_units[lbl]).value
            )
            flat_result[lbl]["confidence_interval"]["max"] = (
                (flat_result[lbl]["confidence_interval"]["max"] * unit)
                .to(plot_units[lbl])
                .value
            )
            flat_result[lbl]["unit"] = plot_units[lbl].to_string()

    traces_to_plot = variable_labels if traces_to_plot is None else traces_to_plot

    traces_plot_kwargs = {
        "traces_to_plot": traces_to_plot,
        "flat_chain": flat_chain,
        "variable_labels": variable_labels,
        "fit_params": flat_result,
        "truths": truths,
        "labels": labels,
    }

    logger.debug("Sending data to matplotlib interface.")
    MCMCPlot.paramtrace(**traces_plot_kwargs)
