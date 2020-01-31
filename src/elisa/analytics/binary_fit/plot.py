import numpy as np
from astropy import units as u
from scipy.interpolate import interp1d

from elisa.binary_system import t_layer
from elisa import units as eu
from elisa.analytics.binary.models import central_rv_synthetic
from elisa.observer.observer import Observer
from elisa.graphic import graphics


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

        :param fit_params: dict; {fit_parameter: {value: float, unit: astropy.unit.Unit}
        :param start_phase: float;
        :param stop_phase: float;
        :param number_of_points: int;
        :param y_axis_unit: astropy.unit.Unit;
        :return:
        """
        plot_result_kwargs = dict()
        fit_params = self.rv_fit.rv_fit_params if fit_params is None else fit_params

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
