import re

import numpy as np
from scipy.interpolate import interp1d
from copy import deepcopy
from emcee.autocorr import integrated_time, function_1d

from . import shared
from . mixins import MCMCMixin
from .. models.lc import synthetic_binary
from .. models.rv import central_rv_synthetic
from .. params import parameters, conf
from ... import units as u
from ... observer.utils import normalize_light_curve
from ... binary_system import t_layer
from ... binary_system.system import BinarySystem
from ... graphic.mcmc_graphics import Plot as MCMCPlot
from ... observer.observer import Observer
from ... graphic import graphics
from ... logger import getLogger
from elisa.utils import is_empty


logger = getLogger('analytics.binary_fit.plot')

PLOT_UNITS = {
    'system@asini': u.solRad,
    'system@argument_of_periastron': u.degree,
    'system@gamma': u.km/u.s,
    'system@primary_minimum_time': u.d
}


class MCMCPlotMixin(object):
    """
    Graphics module for visualization of MCMC sampling results.
    """
    fit = None

    def corner(self, flat_chain=None, variable_labels=None, normalization=None,
               quantiles=None, truths=False, show_titles=True, plot_units=None, sigma_clip=False, sigma=5, n_bins=20,
               **kwargs):
        """
        Plots complete correlation plot from supplied parameters. Useful only for visualizing the posterior distribution
        of the MCMC samples.

        :param sigma_clip: bool; if True, posterior distribution is cropped within
                                 (mean - `sigma` * std, mean + `sigma` * std) to filter out the outliers,
        :param sigma: float; positive, with of the sigma cropping interval in the multiples of standard deviation of
                             the chain distribution
        :param flat_chain: numpy.array; flattened chain of all parameters, use only if you want display your own chain.
                                        By default, internal `flat_chain` attribute will be used as a source of the
                                        posterior distribution.
        :param variable_labels: List; list of variables during a MCMC run, which is used to identify columns in
                                      `flat_chain`, use only if you want display your own chain
        :param normalization: Dict[str, Tuple(float, float)]; {var_name: (min_boundary, max_boundary), ...} dictionary
                                                              of boundaries defined by user for each variable needed
                                                              to reconstruct real values from normalized `flat_chain`,
                                                              use only if you want display your own chain
        :param quantiles: Iterable; A list of fractional quantiles to show on the 1-D histograms as vertical dashed
                                   lines.
        :param truths: Union[bool, List]; if True, fit results are used to indicate position of found values. If False,
                                          none are shown. If list is supplied, it functions the same
                                          as in corner.corner function
        :param show_titles: bool; If True, labels above histogram with name of the variable, value, error and units
                                  are displayed
        :param plot_units: Dict; Units in which to display the output {variable_name: unit, ...}
        :param n_bins: int; positive, number of bins in each histogram
        """
        corner(self.fit, flat_chain=flat_chain, variable_labels=variable_labels, normalization=normalization,
               quantiles=quantiles, truths=truths, show_titles=show_titles, plot_units=plot_units,
               sigma_clip=sigma_clip, sigma=sigma, n_bins=n_bins, **kwargs)

    def autocorrelation(self, correlations_to_plot=None, flat_chain=None, variable_labels=None):
        """
        Plots correlation function for the output MCMC chain.

        :param correlations_to_plot: List; names of variables (in flat format e.g. system@inclination) which
                                           autocorrelation function will be displayed
        :param flat_chain: numpy.array; optional, flattened chain of all parameters. If None, internal `flat_chain`
                                        attribute will be used as a source of the posterior distribution.
        :param variable_labels: List; list of variables during a MCMC run, which is used to identify columns in
                                      `flat_chain`. Use only if `flat_chain` is not None.
        """
        autocorrelation(self.fit, correlations_to_plot, flat_chain, variable_labels)

    def traces(self, traces_to_plot=None, flat_chain=None, variable_labels=None,
               normalization=None, plot_units=None, truths=False):
        """
        Plots traces of the MCMC samples.

        :param traces_to_plot: List; names of variables which traces will be displayed
        :param flat_chain: numpy.array; optional, flattened chain of all parameters. If None, internal `flat_chain`
                                        attribute will be used as a source of the posterior distribution.
        :param variable_labels: List; list of variables during a MCMC run, which is used to identify columns in
                                      `flat_chain`. Use only if `flat_chain` is not None.
        :param normalization: Dict[str, Tuple(float, float)]; {var_name: (min_boundary, max_boundary), ...} dictionary
                                                              of boundaries defined by user for each variable
                                                              needed to reconstruct real values from normalized
                                                              `flat_chain`, use only if you want display your own chain
        :param plot_units: Dict; Units in which to display the output {variable@name: unit, ...}
        :param truths: bool; if True, fit results are used to indicate position of found values. If False,
                             none are shown. It will not work with a custom chain. (if `flat_chain` is not None).
        """
        traces(self.fit, traces_to_plot=traces_to_plot, flat_chain=flat_chain, variable_labels=variable_labels,
               normalization=normalization, plot_units=plot_units, truths=truths)


class RVPlot(object):
    def __init__(self, instance, data):
        self.fit = instance
        self.data = data

    def model(self, start_phase=-0.6, stop_phase=0.6, number_of_points=300, y_axis_unit=u.km / u.s,
              return_figure_instance=False, **kwargs):
        """
        Plots the RV model described by fit params or calculated by last run of fitting procedure.

        :param start_phase: float; initial orbital phase of the synthetic observations
        :param stop_phase: float; final orbital phase of the synthetic observations
        :param number_of_points: int; number of model points in the synthetic data
        :param y_axis_unit: astropy.unit.Unit;
        :param return_figure_instance: bool; if True, the Figure instance is returned instead of displaying the
                                             produced figure
        :param kwargs: Dict;
        :**kwargs options for mcmc**:
            * **fit_result** * - Dict - {result_parameter: {value: float, unit: astropy.unit.Unit,
                                         ...(fitting method dependent)}
        """
        logger.debug('Producing/retrieving data for RV plot.')
        plot_result_kwargs = dict()
        fit_result = kwargs.get('fit_result', self.fit.result)

        if fit_result is None:
            raise ValueError('You did not performed radial velocity fit on this instance '
                             'or you did not provided result parameter dictionary.')

        # converting to phase space if necessary
        x_data, y_data, y_err = dict(), dict(), dict()

        for component, data in self.data.items():

            if data.x_unit is u.dimensionless_unscaled:
                x_data[component] = t_layer.adjust_phases(phases=data.x_data, centre=0.0)
            else:
                x_data[component] = t_layer.jd_to_phase(fit_result['system']['primary_minimum_time']['value'],
                                                        fit_result['system']['period']['value'],
                                                        data.x_data, centre=0.0)

            y_data[component] = (data.y_data * data.y_unit).to(y_axis_unit).value
            y_err[component] = (data.y_err * data.y_unit).to(y_axis_unit).value if data.y_err is not None else None

        x_data, y_data, y_err = \
            shared.extend_observations_to_desired_interval(start_phase, stop_phase, x_data, y_data, y_err)

        plot_result_kwargs.update({
            'x_data': x_data,
            'y_data': y_data,
            'y_err': y_err,
            'y_unit': y_axis_unit,
        })

        kwargs_to_replot = parameters.deserialize_result(fit_result)
        kwargs_to_replot = {key: val["value"] for key, val in kwargs_to_replot.items()}
        if 'system@primary_minimum_time' in kwargs_to_replot.keys():
            del kwargs_to_replot['system@primary_minimum_time']
        synth_phases = np.linspace(start_phase, stop_phase, number_of_points)
        rv_fit = central_rv_synthetic(synth_phases, Observer(), **kwargs_to_replot)
        rv_fit = {component: (data * u.VELOCITY_UNIT).to(y_axis_unit).value for component, data in rv_fit.items()}

        interp_fn = {component: interp1d(synth_phases, rv_fit[component])
                     for component in self.data.keys()}
        residuals = {component: y_data[component] - interp_fn[component](x_data[component])
                     for component in self.data.keys()}

        plot_result_kwargs.update({
            'return_figure_instance': return_figure_instance,
            'synth_phases': synth_phases,
            'rv_fit': rv_fit,
            'residuals': residuals,
            'y_unit': y_axis_unit
        })

        logger.debug('Sending data to matplotlib interface.')
        return graphics.binary_rv_fit_plot(**plot_result_kwargs)


class RVPlotLsqr(RVPlot):
    def __init__(self, instance, data):
        super().__init__(instance, data)


class RVPlotMCMC(RVPlot, MCMCPlotMixin):
    def __init__(self, instance, data):
        super().__init__(instance, data)


class LCPlot(object):
    """
    Graphics functions for visualization of LC fit result.
    """
    def __init__(self, instance, data):
        self.fit = instance
        self.data = data

    def model(self, start_phase=-0.6, stop_phase=0.6, number_of_points=300, discretization=5,
              separation=0.1, data_frac_to_normalize=0.1, normalization_kind='maximum', plot_legend=True, loc=1,
              return_figure_instance=False, rasterize=None, **kwargs):
        """
        Prepares data for plotting the model described by fit params or calculated by last run of fitting procedure.

        :param separation: float; separation between different filters, useful while plotting normalized LCs in
                                  different passbands
        :param start_phase: float; initial orbital phase of the synthetic observations
        :param stop_phase: float; final orbital phase of the synthetic observations
        :param number_of_points: int; number of model points in the synthetic data
        :param discretization: float; discretization factor for the primary component during calculation of the
                                      synthetic observations
        :param data_frac_to_normalize: float; optional, between (0, 1), fraction of data points with the highest flux
                                              used for normalization, depends on level of noise in your data
        :param normalization_kind: str; `average` or `maximum`
        :param loc: int; location of the legend
        :param plot_legend: bool; display legend
        :param return_figure_instance: bool; if True, the Figure instance is returned instead of displaying the
                                             produced figure
        :param rasterize: if True, figure is returned in rasterized form, thus reducing the size of the image
        :param kwargs: Dict;
        :**kwargs options for mcmc**:
            * **fit_result** * - Dict - {result_parameter: {value: float, unit: astropy.unit.Unit,
                                        ...(fitting method dependent)}
        """
        logger.debug('Producing/retrieving data for LC plot.')
        average_kind = normalization_kind
        plot_result_kwargs = dict()
        fit_result = kwargs.get('fit_result', self.fit.result)

        if fit_result is None:
            raise ValueError('You did not performed light curve fit on this instance '
                             'or you did not provided parameter dictionary.')

        # converting to phase space if necessary
        x_data, y_data, y_err = dict(), dict(), dict()

        for band, data in self.data.items():
            if data.x_unit is u.dimensionless_unscaled:
                x_data[band] = t_layer.adjust_phases(phases=data.x_data, centre=0.0)
            else:
                x_data[band] = t_layer.jd_to_phase(fit_result['system']['primary_minimum_time']['value'],
                                                   fit_result['system']['period']['value'],
                                                   data.x_data, centre=0.0)
            y_data[band] = data.y_data
            y_err[band] = data.y_err

        # normalize
        y_data, y_err = normalize_light_curve(y_data, y_err, kind=average_kind,
                                              top_fraction_to_average=data_frac_to_normalize)

        # add y separation to normalized data
        i, y_len = 0, len(y_data)
        for curve in y_data.values():
            curve -= separation * (i - int(y_len/2))
            i += 1

        # extending observations to desired phase interval
        for band, curve in self.data.items():
            phases_extended = np.concatenate((x_data[band] - 1.0, x_data[band], x_data[band] + 1.0))
            phases_extended_filter = np.logical_and(start_phase < phases_extended,  phases_extended < stop_phase)
            x_data[band] = phases_extended[phases_extended_filter]

            y_data[band] = np.tile(y_data[band], 3)[phases_extended_filter]
            if not is_empty(y_err[band]):
                y_err[band] = np.tile(y_err[band], 3)[phases_extended_filter]

        x_data, y_data, y_err = \
            shared.extend_observations_to_desired_interval(start_phase, stop_phase, x_data, y_data, y_err)

        plot_result_kwargs.update({
            'x_data': x_data,
            'y_data': y_data,
            'y_err': y_err,
        })

        kwargs_to_replot = parameters.deserialize_result(fit_result)
        kwargs_to_replot = {key: val["value"] for key, val in kwargs_to_replot.items()}
        if 'system@primary_minimum_time' in kwargs_to_replot.keys():
            del kwargs_to_replot['system@primary_minimum_time']
        synth_phases = np.linspace(start_phase, stop_phase, number_of_points)
        observer = Observer(passband=self.data.keys(), system=None)
        observer._system_cls = BinarySystem

        lc_fit = synthetic_binary(synth_phases, discretization, observer, **kwargs_to_replot)
        lc_fit, _ = normalize_light_curve(lc_fit, kind=average_kind, top_fraction_to_average=0.001)

        idx = 0
        for curve in lc_fit.values():
            curve -= separation * (idx - int(y_len/2))
            idx += 1

        # interpolating synthetic curves to observations and its residuals
        interp_fn = {band: interp1d(synth_phases, lc_fit[band], kind='cubic') for band in self.data.keys()}
        residuals = {band: y_data[band] - np.mean(y_data[band]) - interp_fn[band](x_data[band]) +
                           np.mean(interp_fn[band](x_data[band])) for band in self.data.keys()}

        plot_result_kwargs.update({
            'return_figure_instance': return_figure_instance,
            'synth_phases': synth_phases,
            'lcs': lc_fit,
            'residuals': residuals,
            'legend': plot_legend,
            'loc': loc,
            'rasterize': rasterize
        })

        logger.debug('Sending data to matplotlib interface.')
        return graphics.binary_lc_fit_plot(**plot_result_kwargs)


class LCPlotLsqr(LCPlot):
    def __init__(self, instance, data):
        super().__init__(instance, data)


class LCPlotMCMC(LCPlot, MCMCPlotMixin):
    def __init__(self, instance, data):
        super().__init__(instance, data)


def serialize_plot_labels(variable_labels):
    """
    Return Tex compatible labels of model parameters.

    :param variable_labels: List; flat format labels of model parameters, e.g. system@inclination
    :return: List; plot labels
    """
    labels = []
    for lbl in variable_labels:
        lbl_s = lbl.split(conf.PARAM_PARSER)

        if re.search(r"|".join(conf.COMPOSITE_FLAT_PARAMS), lbl):
            labels.append(f'{lbl_s[-2]} {conf.PARAMS_KEY_TEX_MAP[lbl_s[-1]]}')
        else:
            labels.append(conf.PARAMS_KEY_TEX_MAP[lbl])
    return labels


def corner(mcmc_fit_instance, flat_chain=None, variable_labels=None, normalization=None,
           quantiles=None, truths=False, show_titles=True, plot_units=None, sigma_clip=False, sigma=5, n_bins=20,
           **kwargs):
    """
    Plots complete correlation plot from supplied parameters. Usefull only for MCMC method

    :param sigma: float; positive, with of the sigma cropping interval in the multiples of standard deviation of
                         the chain distribution
    :param sigma_clip: bool; if True, posterior distribution is cropped within
                             (mean - `sigma` * std, mean + `sigma` * std) to filter out the outliers,
    :param mcmc_fit_instance: Union[elisa.analytics.binary_fit.lc_fit.LCFitMCMC,
                                    elisa.analytics.binary_fit.rv_fit.RVFitMCMC];
    :param flat_chain: numpy.array; flattened chain of all parameters, use only if you want display your own chain.
                                    By default, internal `flat_chain` attribute will be used as a source of the
                                    posterior distribution.
    :param variable_labels: List; list of variables during a MCMC run, which is used to identify columns in
                                  `flat_chain`, use only if you want display your own chain
    :param normalization: Dict[str, Tuple(float, float)]; {var_name: (min_boundary, max_boundary), ...} dictionary
                                                          of boundaries defined by user for each variable needed to
                                                          reconstruct real values from normalized `flat_chain`,
                                                          use only if you want display your own chain
    :param quantiles: Iterable; A list of fractional quantiles to show on the 1-D histograms
                                as vertical dashed lines.
    :param truths: Union[bool, list]; if True, fit results are used to indicate position of found values. If False,
                                      none are shown. If list is supplied, it functions
                                      the same as in corner.corner function.
    :param show_titles: bool; If True, labels above histogram with name of the variable, value,
                              errors and units are displayed
    :param plot_units: Dict; Units in which to display the output {variable_name: unit, ...}
    :param n_bins: int; positive, number of bins in each histogram
    """
    logger.debug('Producing/retrieving data for corner plot.')
    flat_chain = deepcopy(mcmc_fit_instance.flat_chain) if flat_chain is None else deepcopy(flat_chain)
    variable_labels = mcmc_fit_instance.variable_labels if variable_labels is None else variable_labels
    normalization = mcmc_fit_instance.normalization if normalization is None else normalization
    quantiles = [0.16, 0.5, 0.84] if quantiles is None else quantiles
    flat_result = deepcopy(mcmc_fit_instance.flat_result)

    corner_plot_kwargs = dict()
    if flat_chain is None:
        raise ValueError('You can use corner plot after running mcmc method or after loading the flat chain.')

    plot_labels = serialize_plot_labels(variable_labels)
    flat_chain = MCMCMixin.renormalize_flat_chain(flat_chain, mcmc_fit_instance.variable_labels, variable_labels,
                                                  normalization)

    # # transforming units and rearanging them into the correct order
    flat_chain_reduced = np.empty((flat_chain.shape[0], len(variable_labels)))
    plot_units = PLOT_UNITS if plot_units is None else plot_units
    for ii, lbl in enumerate(variable_labels):
        if lbl in plot_units.keys():
            unt = u.Unit(flat_result[lbl]['unit'])
            flat_chain_reduced[:, ii] = (flat_chain[:, ii] * unt).to(plot_units[lbl]).value
            flat_result[lbl]['value'] = (flat_result[lbl]['value'] * unt).to(plot_units[lbl]).value
            flat_result[lbl]["confidence_interval"]['min'] = \
                (flat_result[lbl]["confidence_interval"]['min'] * unt).to(plot_units[lbl]).value
            flat_result[lbl]["confidence_interval"]['max'] = \
                (flat_result[lbl]["confidence_interval"]['max'] * unt).to(plot_units[lbl]).value
            flat_result[lbl]['unit'] = plot_units[lbl].to_string()
        else:
            flat_chain_reduced[:, ii] = flat_chain[:, ii]

    truths = [flat_result[lbl]['value'] for lbl in variable_labels] if truths is True else None

    if sigma_clip:
        for ii, lbl in enumerate(variable_labels):
            tol = 0.5 * sigma * np.abs(flat_result[lbl]["confidence_interval"]['max'] -
                                       flat_result[lbl]["confidence_interval"]['min'])
            mask = np.logical_and(flat_chain_reduced[:, ii] > flat_result[lbl]['value'] - tol,
                                  flat_chain_reduced[:, ii] < flat_result[lbl]['value'] + tol)
            flat_chain_reduced = flat_chain_reduced[mask]

    corner_plot_kwargs.update({
        'flat_chain': flat_chain_reduced,
        'truths': truths,
        'variable_labels': variable_labels,
        'labels': plot_labels,
        'quantiles': quantiles,
        'show_titles': show_titles,
        'fit_params': flat_result,
        'bins': n_bins
    })
    corner_plot_kwargs.update(**kwargs)
    logger.debug('Sending data to matplotlib interface.')
    MCMCPlot.corner(**corner_plot_kwargs)


def autocorrelation(mcmc_fit_instance, correlations_to_plot=None, flat_chain=None, variable_labels=None):
    """
    Plots correlation function of defined parameters.

    :param mcmc_fit_instance: Union[elisa.analytics.binary_fit.lc_fit.LCFitMCMC,
                                    elisa.analytics.binary_fit.rv_fit.RVFitMCMC];
    :param correlations_to_plot: List; names of variables (in flat format e.g. system@inclination) which
                                       autocorrelation function will be displayed
    :param flat_chain: numpy.array; optional, flattened chain of all parameters. If None, internal `flat_chain`
                                    attribute will be used as a source of the posterior distribution.
    :param variable_labels: List; list of variables during a MCMC run, which is used to identify columns in
                                  `flat_chain`. Use only if `flat_chain` is not None.
    """
    autocorr_plot_kwargs = dict()

    flat_chain = deepcopy(mcmc_fit_instance.flat_chain) if flat_chain is None else deepcopy(flat_chain)
    variable_labels = mcmc_fit_instance.variable_labels if variable_labels is None else variable_labels
    correlations_to_plot = variable_labels if correlations_to_plot is None else correlations_to_plot

    if flat_chain is None:
        raise ValueError('You can use trace plot only in case of mcmc method '
                         'or for some reason the flat chain was not found.')

    labels = serialize_plot_labels(variable_labels)

    autocorr_fns = np.empty((flat_chain.shape[0], len(variable_labels)))
    autocorr_time = np.empty((flat_chain.shape[0]))

    for i, lbl in enumerate(variable_labels):
        autocorr_fns[:, i] = function_1d(flat_chain[:, i])
        autocorr_time[i] = integrated_time(flat_chain[:, i], quiet=True)

    autocorr_plot_kwargs.update({
        'correlations_to_plot': correlations_to_plot,
        'autocorr_fns': autocorr_fns,
        'autocorr_time': autocorr_time,
        'variable_labels': variable_labels,
        'labels': labels
    })

    MCMCPlot.autocorr(**autocorr_plot_kwargs)


def traces(mcmc_fit_instance, traces_to_plot=None, flat_chain=None, variable_labels=None,
           normalization=None, plot_units=None, truths=False):
    """
    Plots traces of defined parameters.

    :param mcmc_fit_instance: Union[elisa.analytics.binary_fit.lc_fit.LCFitMCMC,
                                    elisa.analytics.binary_fit.rv_fit.RVFitMCMC];
    :param traces_to_plot: List; names of variables which traces will be displayed
    :param flat_chain: numpy.array; optional, flattened chain of all parameters. If None, internal `flat_chain`
                                    attribute will be used as a source of the posterior distribution.
    :param variable_labels: List; list of variables during a MCMC run, which is used to identify columns in
                                  `flat_chain`. Use only if `flat_chain` is not None.
    :param normalization: Dict[str, Tuple(float, float)]; {var_name: (min_boundary, max_boundary), ...} dictionary
                                                          of boundaries defined by user for each variable needed to
                                                          reconstruct real values from normalized `flat_chain`,
                                                          use only if you want display your own chain
    :param plot_units: Dict; Units in which to display the output {variable_name: unit, ...}
    :param truths: bool; if True, fit results are used to indicate position of found values. If False,
                         none are shown. It will not work with a custom chain. (if `flat_chain` is not None).
    """
    logger.debug('Producing/retrieving data for traces plot.')
    traces_plot_kwargs = dict()

    variable_labels = mcmc_fit_instance.variable_labels if variable_labels is None else variable_labels
    normalization = mcmc_fit_instance.normalization if normalization is None else normalization
    flat_chain = deepcopy(mcmc_fit_instance.flat_chain) if flat_chain is None else deepcopy(flat_chain)
    flat_result = deepcopy(mcmc_fit_instance.flat_result)

    if flat_chain is None:
        raise ValueError('You can use trace plot only in case of mcmc method '
                         'or for some reason the flat chain was not found.')

    flat_chain = MCMCMixin.renormalize_flat_chain(flat_chain, mcmc_fit_instance.variable_labels, variable_labels,
                                                  normalization)
    labels = serialize_plot_labels(variable_labels)

    # transforming units
    plot_units = PLOT_UNITS if plot_units is None else plot_units
    for ii, lbl in enumerate(variable_labels):
        if lbl in plot_units.keys():
            unt = u.Unit(flat_result[lbl]['unit'])
            flat_chain[:, ii] = (flat_chain[:, ii] * unt).to(plot_units[lbl]).value
            flat_result[lbl]['value'] = (flat_result[lbl]['value'] * unt).to(plot_units[lbl]).value
            flat_result[lbl]["confidence_interval"]['min'] = \
                (flat_result[lbl]["confidence_interval"]['min'] * unt).to(plot_units[lbl]).value
            flat_result[lbl]["confidence_interval"]['max'] = \
                (flat_result[lbl]["confidence_interval"]['max'] * unt).to(plot_units[lbl]).value
            flat_result[lbl]['unit'] = plot_units[lbl].to_string()

    traces_to_plot = variable_labels if traces_to_plot is None else traces_to_plot

    traces_plot_kwargs.update({
        'traces_to_plot': traces_to_plot,
        'flat_chain': flat_chain,
        'variable_labels': variable_labels,
        'fit_params': flat_result,
        'truths': truths,
        'labels': labels,
    })

    logger.debug('Sending data to matplotlib interface.')
    MCMCPlot.paramtrace(**traces_plot_kwargs)
