import numpy as np
from astropy import units as u
from scipy.interpolate import interp1d
from copy import copy, deepcopy
from emcee.autocorr import integrated_time, function_1d

from elisa.binary_system import t_layer
from elisa import units as eu
from elisa.analytics.binary.models import central_rv_synthetic
from elisa.graphic.mcmc_graphics import Plot as MCMCPlot
from elisa.observer.observer import Observer
from elisa.graphic import graphics
from elisa.analytics.binary import (
    params,
    models,
    utils as bsutils
)
from elisa.analytics import utils as autils
from elisa import utils


PLOT_UNITS = {
    'asini': eu.solRad, 
    'argument_of_periastron': eu.degree, 
    'gamma': eu.km/eu.s, 
    'primary_minimum_time': eu.d
}


class RVPlot(object):
    """
    Universal plot interface for RVFit class.
    """
    def __init__(self, instance):
        self.rv_fit = instance

    def model(self, fit_params=None, start_phase=-0.6, stop_phase=0.6, number_of_points=300,
              y_axis_unit=eu.km / eu.s):
        """
        Prepares data for plotting the model described by fit params or calculated by last run of fitting procedure.

        :param fit_params: Dict; {fit_parameter: {value: float, unit: astropy.unit.Unit, ...(fitting method dependent)}
        :param start_phase: float;
        :param stop_phase: float;
        :param number_of_points: int;
        :param y_axis_unit: astropy.unit.Unit;
        """
        plot_result_kwargs = dict()
        fit_params = self.rv_fit.fit_params if fit_params is None else fit_params

        if fit_params is None:
            raise ValueError('You did not performed radial velocity fit on this instance or you did not provided result'
                             ' parameter dictionary.')

        fit_params = autils.transform_initial_values(fit_params)

        # converting to phase space
        x_data, y_data, yerr = dict(), dict(), dict()
        for component, data in self.rv_fit.radial_velocities.items():
            x_data[component] = t_layer.adjust_phases(phases=data.x_data, centre=0.0) \
                if data.x_unit is u.dimensionless_unscaled else \
                t_layer.jd_to_phase(fit_params['primary_minimum_time']['value'], fit_params['period']['value'],
                                    data.x_data, centre=0.0)
            y_data[component] = (data.y_data * data.y_unit).to(y_axis_unit).value
            yerr[component] = (data.yerr * data.y_unit).to(y_axis_unit).value if data.yerr is not None else None
        plot_result_kwargs.update({
            'x_data': x_data,
            'y_data': y_data,
            'yerr': yerr,
            'y_unit': y_axis_unit,
        })

        kwargs_to_replot = {}
        for key, val in fit_params.items():
            kwargs_to_replot[key] = val['value']

        if 'primary_minimum_time' in kwargs_to_replot.keys():
            del kwargs_to_replot['primary_minimum_time']
        synth_phases = np.linspace(start_phase, stop_phase, number_of_points)
        rv_fit = central_rv_synthetic(synth_phases, Observer(), **kwargs_to_replot)
        rv_fit = {component: (data * eu.VELOCITY_UNIT).to(y_axis_unit).value for component, data in rv_fit.items()}

        interp_fn = {component: interp1d(synth_phases, rv_fit[component])
                     for component in self.rv_fit.radial_velocities.keys()}
        residuals = {component: y_data[component] - interp_fn[component](x_data[component])
                     for component in self.rv_fit.radial_velocities.keys()}
        plot_result_kwargs.update({
            'synth_phases': synth_phases,
            'rv_fit': rv_fit,
            'residuals': residuals,
            'y_unit': y_axis_unit
        })

        graphics.binary_rv_fit_plot(**plot_result_kwargs)

    def corner(self, flat_chain=None, variable_labels=None, normalization=None, quantiles=None, truths=False,
               show_titles=True, plot_units=None, **kwargs):
        """
        Plots complete correlation plot from supplied parameters. Usefull only for MCMC method.

        :param flat_chain: numpy.array; flattened chain of all parameters, use only if you want display your own chain
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
        :param show_titles: bool; If True, labels above histogram with name of the variable, value, errorss and units
                                  are displayed
        :param plot_units: Dict; Units in which to display the output {variable_name: unit, ...}
        """
        corner(self.rv_fit, flat_chain=flat_chain, variable_labels=variable_labels, normalization=normalization,
               quantiles=quantiles, truths=truths, show_titles=show_titles, plot_units=plot_units, **kwargs)

    def traces(self, traces_to_plot=None, flat_chain=None, variable_labels=None, normalization=None, plot_units=None,
               truths=False):
        """
        Plots traces of defined parameters.

        :param traces_to_plot: List; names of variables which traces will be displayed
        :param flat_chain: numpy.array; flattened chain of all parameters
        :param variable_labels: List; list of variables during a MCMC run, which is used to identify columns in
                                     `flat_chain`
        :param normalization: Dict[str, Tuple(float, float)]; {var_name: (min_boundary, max_boundary), ...} dictionary
                                                              of boundaries defined by user for each variable
                                                              needed to reconstruct real values from normalized
                                                              `flat_chain`, use only if you want display your own chain
        :param plot_units: Dict; Units in which to display the output {variable_name: unit, ...}
        :param truths: bool; if True, fit results are used to indicate position of found values. If False,
                             none are shown. It will not work with a custom chain. (if `flat_chain` is not None).
        """
        traces(self.rv_fit, traces_to_plot=traces_to_plot, flat_chain=flat_chain, variable_labels=variable_labels,
               normalization=normalization, plot_units=plot_units, truths=truths)

    def autocorrelation(self, correlations_to_plot=None, flat_chain=None, variable_labels=None):
        """
        Plots correlation function of defined parameters.

        :param correlations_to_plot: List; names of variables which autocorrelation function will be displayed
        :param flat_chain: numpy.array; flattened chain of all parameters
        :param variable_labels: List; list of variables during a MCMC run, which is used
                                      to identify columns in `flat_chain`
        """
        autocorrelation(self.rv_fit, correlations_to_plot, flat_chain, variable_labels)


class LCPlot(object):
    """
    Universal plot interface for LCFit class.
    """

    def __init__(self, instance):
        self.lc_fit = instance

    def model(self, fit_params=None, start_phase=-0.6, stop_phase=0.6, number_of_points=300,
              y_axis_unit=u.dimensionless_unscaled, discretization=3):
        """
        Prepares data for plotting the model described by fit params or calculated by last run of fitting procedure.

        :param fit_params: dict; {fit_parameter: {value: float, unit: astropy.unit.Unit, ...(fitting method dependent)}
        :param start_phase: float;
        :param stop_phase: float;
        :param number_of_points: int;
        :param y_axis_unit: astropy.unit.Unit;
        :param discretization: unit
        """
        plot_result_kwargs = dict()
        fit_params = self.lc_fit.fit_params if fit_params is None else fit_params

        if fit_params is None:
            raise ValueError('You did not performed light curve fit on this instance or you did not provided '
                             'result parameter dictionary.')

        fit_params = autils.transform_initial_values(fit_params)

        # converting to phase space
        x_data, y_data, yerr = dict(), dict(), dict()
        for _filter, curve in self.lc_fit.light_curves.items():
            x_data[_filter] = t_layer.adjust_phases(phases=curve.x_data, centre=0.0) \
                if curve.x_unit is u.dimensionless_unscaled else \
                t_layer.jd_to_phase(fit_params['primary_minimum_time']['value'], fit_params['period']['value'],
                                    curve.x_data, centre=0.0)

            y_data[_filter] = utils.flux_to_magnitude(curve.y_data, curve.yerr) if y_axis_unit == u.mag \
                else curve.y_data

            yerr[_filter] = utils.flux_error_to_magnitude_error(curve.yerr, curve.reference_magnitude) \
                if y_axis_unit == u.mag else curve.yerr
        y_data = bsutils.normalize_light_curve(y_data, kind='global_maximum')

        # extending observations to desired phase interval
        for filter, curve in self.lc_fit.light_curves.items():
            phases_extended = np.concatenate((x_data[filter] - 1.0, x_data[filter], x_data[filter] + 1.0))
            phases_extended_filter = np.logical_and(start_phase < phases_extended,  phases_extended < stop_phase)
            x_data[filter] = phases_extended[phases_extended_filter]

            y_data[filter] = np.tile(y_data[filter], 3)[phases_extended_filter]
            yerr[filter] = np.tile(yerr[filter], 3)[phases_extended_filter]

        plot_result_kwargs.update({
            'x_data': x_data,
            'y_data': y_data,
            'yerr': yerr,
            'y_unit': y_axis_unit,
        })

        kwargs_to_replot = {}
        for key, val in fit_params.items():
            kwargs_to_replot[key] = val['value']

        if 'primary_minimum_time' in kwargs_to_replot.keys():
            del kwargs_to_replot['primary_minimum_time']
        synth_phases = np.linspace(start_phase, stop_phase, number_of_points)

        # system_kwargs = params.prepare_kwargs()
        system_kwargs = {key: val['value'] for key, val in fit_params.items()}
        period = system_kwargs.pop('period')
        system = models.prepare_binary(period=period, discretization=discretization, **system_kwargs)
        observer = Observer(passband=self.lc_fit.light_curves.keys(), system=system)
        synthetic_curves = models.synthetic_binary(synth_phases,
                                                   period,
                                                   discretization=discretization,
                                                   morphology=None,
                                                   observer=observer,
                                                   _raise_invalid_morphology=False,
                                                   **system_kwargs)
        synthetic_curves = bsutils.normalize_light_curve(synthetic_curves, kind='global_maximum')

        # interpolating synthetic curves to observations and its residuals
        interp_fn = {component: interp1d(synth_phases, synthetic_curves[component], kind='cubic')
                     for component in self.lc_fit.light_curves.keys()}
        residuals = {component: y_data[component] - np.mean(y_data[component]) -
                                interp_fn[component](x_data[component]) +
                                np.mean(interp_fn[component](x_data[component]))
                     for component in self.lc_fit.light_curves.keys()}

        plot_result_kwargs.update({
            'synth_phases': synth_phases,
            'lcs': synthetic_curves,
            'residuals': residuals,
            'y_unit': y_axis_unit
        })

        graphics.binary_lc_fit_plot(**plot_result_kwargs)

    def corner(self, flat_chain=None, variable_labels=None, normalization=None, quantiles=None, truths=False,
               show_titles=True, plot_units=None, **kwargs):
        """
        Plots complete correlation plot from supplied parameters. Usefull only for MCMC method.

        :param flat_chain: numpy.array; flattened chain of all parameters, use only if you want display your own chain
        :param variable_labels: List; list of variables during a MCMC run, which is used to identify columns in
                                      `flat_chain`, , use only if you want display your own chain
        :param normalization: Dict[str, Tuple(float, float)]; {var_name: (min_boundary, max_boundary), ...} dictionary
                                                              of boundaries defined by user for each variable needed
                                                              to reconstruct real values from normalized `flat_chain`,
                                                              use only if you want display your own chain
        :param quantiles: Iterable; A list of fractional quantiles to show on the 1-D histograms
                                    as vertical dashed lines.
        :param truths: Union[bool, list]; if true, fit results are used to indicate position of found values. If False,
                                          none are shown. If list is supplied, it functions
                                          the same as in corner.corner function.
        :param show_titles: bool; If True, labels above histogram with name of the variable, value, errorss and units
                                  are displayed
        :param plot_units: dict; Units in which to display the output {variable_name: unit, ...}
        """
        corner(self.lc_fit, flat_chain=flat_chain, variable_labels=variable_labels, normalization=normalization,
               quantiles=quantiles, truths=truths, show_titles=show_titles, plot_units=plot_units, **kwargs)

    def traces(self, traces_to_plot=None, flat_chain=None, variable_labels=None, normalization=None, plot_units=None,
               truths=False):
        """
        Plots traces of defined parameters.

        :param traces_to_plot: List; names of variables which traces will be displayed
        :param flat_chain: numpy.array; flattened chain of all parameters
        :param variable_labels: List; list of variables during a MCMC run, which is used to
                                      identify columns in `flat_chain`
        :param normalization: Dict[str, Tuple(float, float)]; {var_name: (min_boundary, max_boundary), ...} dictionary
                                                              of boundaries defined by user for each variable needed
                                                              to reconstruct real values from normalized `flat_chain`,
                                                              use only if you want display your own chain
        :param plot_units: dict; Units in which to display the output {variable_name: unit, ...}
        :param truths: bool; if True, fit results are used to indicate position of found values. If False,
                             none are shown. It will not work with a custom chain. (if `flat_chain` is not None).
        """
        traces(self.lc_fit, traces_to_plot=traces_to_plot, flat_chain=flat_chain, variable_labels=variable_labels,
               normalization=normalization, plot_units=plot_units, truths=truths)

    def autocorrelation(self, correlations_to_plot=None, flat_chain=None, variable_labels=None):
        """
        Plots correlation function of defined parameters.

        :param correlations_to_plot: List; names of variables which autocorrelation function will be displayed
        :param flat_chain: numpy.array; flattened chain of all parameters
        :param variable_labels: List; list of variables during a MCMC run, which is used
                                      to identify columns in `flat_chain`
        """
        autocorrelation(self.lc_fit, correlations_to_plot, flat_chain, variable_labels)


def corner(fit_instance, flat_chain=None, variable_labels=None, normalization=None, quantiles=None, truths=False,
           show_titles=True, plot_units=None, **kwargs):
    """
    Plots complete correlation plot from supplied parameters. Usefull only for MCMC method

    :param fit_instance: Union[elisa.analytics.binary_fit.lc_fit.LCFit, elisa.analytics.binary_fit.rv_fit.RVFit];
    :param flat_chain: numpy.array; flattened chain of all parameters, use only if you want display your own chain
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
    :param plot_units: dict; Units in which to display the output {variable_name: unit, ...}
    """
    flat_chain = copy(fit_instance.flat_chain) if flat_chain is None else copy(flat_chain)

    variable_labels = fit_instance.variable_labels if variable_labels is None else variable_labels
    normalization = fit_instance.normalization if normalization is None else normalization

    fit_params = deepcopy(fit_instance.fit_params)

    corner_plot_kwargs = dict()
    if flat_chain is None:
        raise ValueError('You can use corner plot after running mcmc method or after loading the flat chain.')

    labels = [params.PARAMS_KEY_TEX_MAP[label] for label in variable_labels]
    quantiles = [0.16, 0.5, 0.84] if quantiles is None else quantiles

    # renormalizing flat chain to meaningful values
    flat_chain = params.renormalize_flat_chain(flat_chain, variable_labels, normalization)

    # transforming units
    plot_units = PLOT_UNITS if plot_units is None else plot_units
    for ii, lbl in enumerate(variable_labels):
        if lbl in plot_units.keys():
            unt = u.Unit(fit_params[lbl]['unit'])
            flat_chain[:, ii] = (flat_chain[:, ii] * unt).to(plot_units[lbl]).value
            fit_params[lbl]['value'] = (fit_params[lbl]['value'] * unt).to(plot_units[lbl]).value
            fit_params[lbl]['min'] = (fit_params[lbl]['min'] * unt).to(plot_units[lbl]).value
            fit_params[lbl]['max'] = (fit_params[lbl]['max'] * unt).to(plot_units[lbl]).value
            fit_params[lbl]['unit'] = plot_units[lbl].to_string()

    truths = [fit_params[lbl]['value'] for lbl in variable_labels] if truths is True else None

    corner_plot_kwargs.update({
        'flat_chain': flat_chain,
        'truths': truths,
        'variable_labels': variable_labels,
        'labels': labels,
        'quantiles': quantiles,
        'show_titles': show_titles,
        'fit_params': fit_params
    })
    corner_plot_kwargs.update(**kwargs)

    MCMCPlot.corner(**corner_plot_kwargs)


def traces(fit_instance, traces_to_plot=None, flat_chain=None, variable_labels=None,
           normalization=None, plot_units=None, truths=False):
    """
    Plots traces of defined parameters.

    :param fit_instance: Union[elisa.analytics.binary_fit.lc_fit.LCFit, elisa.analytics.binary_fit.rv_fit.RVFit];
    :param traces_to_plot: List; names of variables which traces will be displayed
    :param flat_chain: numpy.array; flattened chain of all parameters
    :param variable_labels: List; list of variables during a MCMC run, which is used
                                  to identify columns in `flat_chain`
    :param normalization: Dict[str, Tuple(float, float)]; {var_name: (min_boundary, max_boundary), ...} dictionary
                                                          of boundaries defined by user for each variable needed to
                                                          reconstruct real values from normalized `flat_chain`,
                                                          use only if you want display your own chain
    :param plot_units: Dict; Units in which to display the output {variable_name: unit, ...}
    :param truths: bool; if True, fit results are used to indicate position of found values. If False,
                         none are shown. It will not work with a custom chain. (if `flat_chain` is not None).
    """
    traces_plot_kwargs = dict()

    flat_chain = copy(fit_instance.flat_chain) if flat_chain is None else copy(flat_chain)

    if flat_chain is None:
        raise ValueError('You can use trace plot only in case of mcmc method or for some reason the flat chain was '
                         'not found.')

    variable_labels = fit_instance.variable_labels if variable_labels is None else variable_labels
    normalization = fit_instance.normalization if normalization is None else normalization

    flat_chain = params.renormalize_flat_chain(flat_chain, variable_labels, normalization)

    # transforming units
    fit_params = deepcopy(fit_instance.fit_params)
    plot_units = PLOT_UNITS if plot_units is None else plot_units
    for ii, lbl in enumerate(variable_labels):
        if lbl in plot_units.keys():
            unt = u.Unit(fit_params[lbl]['unit'])
            flat_chain[:, ii] = (flat_chain[:, ii] * unt).to(plot_units[lbl]).value
            fit_params[lbl]['value'] = (fit_params[lbl]['value'] * unt).to(plot_units[lbl]).value
            fit_params[lbl]['min'] = (fit_params[lbl]['min'] * unt).to(plot_units[lbl]).value
            fit_params[lbl]['max'] = (fit_params[lbl]['max'] * unt).to(plot_units[lbl]).value
            fit_params[lbl]['unit'] = plot_units[lbl].to_string()

    traces_to_plot = variable_labels if traces_to_plot is None else traces_to_plot

    traces_plot_kwargs.update({
        'traces_to_plot': traces_to_plot,
        'flat_chain': flat_chain,
        'variable_labels': variable_labels,
        'fit_params': fit_params,
        'truths': truths,
    })

    MCMCPlot.paramtrace(**traces_plot_kwargs)


def autocorrelation(fit_instance, correlations_to_plot=None, flat_chain=None, variable_labels=None):
    """
    Plots correlation function of defined parameters.

    :param fit_instance: Union[elisa.analytics.binary_fit.lc_fit.LCFit, elisa.analytics.binary_fit.rv_fit.RVFit];
    :param correlations_to_plot: List; names of variables which autocorrelation function will be displayed
    :param flat_chain: numpy.array; flattened chain of all parameters
    :param variable_labels: List; list of variables during a MCMC run, which is used
                                  to identify columns in `flat_chain`
    """
    autocorr_plot_kwargs = dict()

    flat_chain = copy(fit_instance.flat_chain) if flat_chain is None else copy(flat_chain)

    if flat_chain is None:
        raise ValueError('You can use trace plot only in case of mcmc method or for some reason the flat chain was '
                         'not found.')

    variable_labels = fit_instance.variable_labels if variable_labels is None else variable_labels

    autocorr_fns = np.empty((flat_chain.shape[0], len(variable_labels)))
    autocorr_time = np.empty((flat_chain.shape[0]))
    for ii, lbl in enumerate(variable_labels):
        autocorr_fns[:, ii] = function_1d(flat_chain[:, ii])
        # autocorr_time[ii] = integrated_time(flat_chain[:, ii])
        autocorr_time[ii] = integrated_time(flat_chain[:, ii], quiet=True)

    correlations_to_plot = variable_labels if correlations_to_plot is None else correlations_to_plot

    autocorr_plot_kwargs.update({
        'correlations_to_plot': correlations_to_plot,
        'autocorr_fns': autocorr_fns,
        'autocorr_time': autocorr_time,
        'variable_labels': variable_labels,
    })

    MCMCPlot.autocorr(**autocorr_plot_kwargs)
