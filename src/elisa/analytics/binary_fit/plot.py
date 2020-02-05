import numpy as np
from astropy import units as u
from scipy.interpolate import interp1d
from copy import copy, deepcopy

from elisa.binary_system import t_layer
from elisa import units as eu
from elisa.analytics.binary.models import central_rv_synthetic
from elisa.graphic.mcmc_graphics import Plot as MCMCPlot
from elisa.observer.observer import Observer
from elisa.graphic import graphics
from elisa.analytics.binary import params


PLOT_UNITS = {'asini': u.solRad, 'argument_of_periastron': u.degree, 'gamma': u.km/u.s, 'primary_minimum_time': u.d}


class RVPlot(object):
    """
    Universal plot interface for RVFit class.
    """
    def __init__(self, instance):
        self.rv_fit = instance

    def model(self, fit_params=None, start_phase=-0.6, stop_phase=0.6, number_of_points=300,
                    y_axis_unit=eu.km / eu.s):
        """
        prepares data for plotting the model described by fit params or calculated by last run of fitting procedure

        :param fit_params: dict; {fit_parameter: {value: float, unit: astropy.unit.Unit, ...(fitting method dependent)}
        :param start_phase: float;
        :param stop_phase: float;
        :param number_of_points: int;
        :param y_axis_unit: astropy.unit.Unit;
        :return:
        """
        plot_result_kwargs = dict()
        fit_params = self.rv_fit.rv_fit_params if fit_params is None else fit_params

        if fit_params is None:
            raise ValueError('')

        # converting to phase space
        x_data, y_data, yerr = dict(), dict(), dict()
        for component, data in self.rv_fit.radial_velocities.items():
            x_data[component] = data.x_data if data.x_unit is u.dimensionless_unscaled else \
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
        Plots complete correlation plot from supplied parameters. Usefull only for MCMC method

        :param flat_chain: numpy.ndarray; flattened chain of all parameters, use only if you want display your own chain
        :param variable_labels: list; list of variables during a MCMC run, which is used to identify columns in
        `flat_chain`, , use only if you want display your own chain
        :param normalization: Dict[str, Tuple(float, float)]; {var_name: (min_boundary, max_boundary), ...} dictionary
        of boundaries defined by user for each variable needed to reconstruct real values from normalized `flat_chain`,
        use only if you want display your own chain
        :param quantiles: iterable; A list of fractional quantiles to show on the 1-D histograms as vertical dashed
        lines.
        :param truths: Union[bool, list]; if true, fit results are used to indicate position of found values. If False,
        none are shown. If list is supplied, it functions the same as in corner.corner function.
        :param show_titles: bool; If True, labels above histogram with name of the variable, value, errorss and units
        are displayed
        :param plot_units: dict; Units in which to display the output {variable_name: unit, ...}
        :return:
        """
        corner_plot_kwargs = dict()

        flat_chain = copy(self.rv_fit.flat_chain) if flat_chain is None else copy(flat_chain)
        variable_labels = self.rv_fit.variable_labels if variable_labels is None else variable_labels
        labels = [params.PARAMS_KEY_TEX_MAP[label] for label in variable_labels]
        normalization = self.rv_fit.normalization if normalization is None else normalization
        quantiles = [0.16, 0.5, 0.84] if quantiles is None else quantiles

        # renormalizing flat chain to meaningful values
        flat_chain = params.renormalize_flat_chain(flat_chain, variable_labels, normalization)

        # transforming units
        fit_params = deepcopy(self.rv_fit.fit_params)
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

    def traces(self, traces_to_plot=None, flat_chain=None, variable_labels=None, normalization=None, plot_units=None,
               truths=False):
        """
        Plots traces of defined parameters.

        :param traces_to_plot: list; names of variables which traces will be displayed
        :param flat_chain: numpy.ndarray; flattened chain of all parameters
        :param variable_labels: list; list of variables during a MCMC run, which is used to identify columns in
        `flat_chain`
        :param normalization: Dict[str, Tuple(float, float)]; {var_name: (min_boundary, max_boundary), ...} dictionary
        of boundaries defined by user for each variable needed to reconstruct real values from normalized `flat_chain`,
        use only if you want display your own chain
        :param plot_units: dict; Units in which to display the output {variable_name: unit, ...}
        :param truths: bool; if true, fit results are used to indicate position of found values. If False,
        none are shown. It will not work with a
        custom chain. (if `flat_chain` is not None).
        :return:
        """
        traces_plot_kwargs = dict()

        flat_chain = copy(self.rv_fit.flat_chain) if flat_chain is None else copy(flat_chain)
        variable_labels = self.rv_fit.variable_labels if variable_labels is None else variable_labels
        normalization = self.rv_fit.normalization if normalization is None else normalization

        flat_chain = params.renormalize_flat_chain(flat_chain, variable_labels, normalization)

        # transforming units
        fit_params = deepcopy(self.rv_fit.fit_params)
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






