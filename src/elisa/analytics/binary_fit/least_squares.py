import functools
import numpy as np

from typing import Dict
from scipy import interpolate
from abc import ABCMeta
from scipy.optimize import least_squares

from . shared import (
    AbstractRVFit, AbstractLCFit,
    lc_r_squared, rv_r_squared, r_squared
)

from .. import RVData, LCData
from .. models import rv as rv_model
from .. models import lc as lc_model
from .. tools.utils import time_layer_resolver
from .. params import parameters
from ... observer.utils import normalize_light_curve
from ... logger import getPersistentLogger
from ... import const
from ... import settings
from ... binary_system.system import BinarySystem
from ... binary_system.curves.community import RadialVelocitySystem


logger = getPersistentLogger('analytics.binary_fit.least_squares')


def logger_decorator(suppress_logger=False):
    def do(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not suppress_logger:
                logger.debug(f'current xn value: {kwargs}')
            return func(*args, **kwargs)
        return wrapper
    return do


class LightCurveFit(AbstractLCFit, metaclass=ABCMeta):
    MORPHOLOGY = None

    def model_to_fit(self, xn):
        """
        Model to find minimum.

        :param xn: Iterable[float];
        :return: float;
        """
        diff = 1.0 / self.interp_treshold

        xn = parameters.vector_renormalizer(xn, self.fitable.keys(), self.normalization)
        kwargs = parameters.prepare_properties_set(xn, self.fitable.keys(), self.constrained, self.fixed)
        phases, kwargs = time_layer_resolver(self.x_data_reduced, pop=False, **kwargs)

        fit_xs = np.linspace(np.min(phases) - diff, np.max(phases) + diff, num=self.interp_treshold + 2) \
            if np.shape(phases)[0] > self.interp_treshold else phases
        args = fit_xs, self.discretization, self.observer
        fn = lc_model.synthetic_binary

        try:
            synthetic = logger_decorator()(fn)(*args, **kwargs)
            synthetic, _ = normalize_light_curve(synthetic, kind='average')
        except Exception as e:
            logger.error(f'your initial parameters lead during fitting to invalid binary system, exectpion: {str(e)}')
            return const.MAX_USABLE_FLOAT

        if np.shape(fit_xs) != np.shape(phases):
            synthetic = {
                band: interpolate.interp1d(fit_xs, curve, kind='cubic')(phases)
                for band, curve in synthetic.items()
            }
        else:
            synthetic = {band: val[self.x_data_reducer[band]] for band, val in synthetic.items()}

        residuals = np.sum([np.sum(np.power((synthetic[band] - self.y_data[band]) / self.y_err[band], 2))
                            for band in synthetic])

        r2 = r_squared(synthetic, self.y_data)
        logger.info(f'current R2: {r2}')

        return residuals

    def fit(self, data: Dict[str, LCData], x0: parameters.BinaryInitialParameters,
            discretization=5.0, interp_treshold=None, **kwargs):
        """
        Fit method using non-linear least squares.
        Based on https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

        :param data: elisa.analytics.dataset.base.LCData;
        :param x0: List[Dict]; initial state (metadata included)
        :param discretization: float; discretization of objects
        :param interp_treshold: int; data binning treshold
        :param kwargs: optional arguments for least_squares function (see documentation for
                       scipy.optimize.least_squares method)
        :return: Dict;
        """
        self.set_up(x0, data, passband=data.keys(), discretization=discretization, morphology=self.MORPHOLOGY,
                    interp_treshold=settings.MAX_CURVE_DATA_POINTS if interp_treshold is None else interp_treshold,
                    observer_system_cls=BinarySystem)
        initial_vector = parameters.vector_normalizer(self.initial_vector, self.fitable.keys(), self.normalization)

        # evaluate least squares from scipy
        logger.info("fitting started...")
        result = least_squares(self.model_to_fit, initial_vector, jac=kwargs.get('jac', '2-point'), bounds=(0, 1),
                               method=kwargs.get('method', 'trf'), ftol=kwargs.get('ftol', 1e-8),
                               xtol=kwargs.get('xtol', 1e-8), gtol=kwargs.get('gtol', 1e-8),
                               x_scale=kwargs.get('x_scale', 1.0), loss=kwargs.get('loss', 'linear'),
                               f_scale=kwargs.get('f_scale', 1.0), diff_step=kwargs.get('diff_step', None),
                               tr_solver=kwargs.get('tr_solver', None), tr_options=kwargs.get('tr_options', {}),
                               jac_sparsity=kwargs.get('jac_sparsity', None), max_nfev=kwargs.get('max_nfev', None),
                               verbose=kwargs.get('verbose', 2), args=kwargs.get('args', ()),
                               kwargs=kwargs.get('kwargs', {}))
        logger.info("fitting finished")

        result = parameters.vector_renormalizer(result.x, self.fitable.keys(), self.normalization)
        # put all together
        result_dict = {lbl: {
            'value': result[i],
            'fixed': False,
            'unit': self.fitable[lbl].to_dict()['unit'],
            'min': self.fitable[lbl].min,
            'max': self.fitable[lbl].max,
        } for i, lbl in enumerate(self.fitable.keys())}

        result_dict.update({lbl: {
            "value": val.value,
            'fixed': True,
            'unit': val.to_dict()['unit']
        } for lbl, val in self.fixed.items()})
        result_dict = self.eval_constrained_results(result_dict, self.constrained)

        r_squared_args = (self.x_data_reduced, self.y_data, self.observer.passband, discretization,
                          self.x_data_reducer, 1.0 / self.interp_treshold, self.interp_treshold,
                          self.observer.system_cls)

        r_dict = {key: value['value'] for key, value in result_dict.items()}
        r_squared_result = lc_r_squared(lc_model.synthetic_binary, *r_squared_args, **r_dict)
        result_dict["r_squared"] = {'value': r_squared_result, "unit": None}

        setattr(self, 'flat_result', result_dict)
        return parameters.serialize_result(result_dict)


class OvercontactLightCurveFit(LightCurveFit):
    MORPHOLOGY = 'over-contact'


class DetachedLightCurveFit(LightCurveFit):
    MORPHOLOGY = 'detached'


class CentralRadialVelocity(AbstractRVFit):
    def prepare_synthetic(self, xn):
        xn = parameters.vector_renormalizer(xn, self.fitable.keys(), self.normalization)
        kwargs = parameters.prepare_properties_set(xn, self.fitable.keys(), self.constrained, self.fixed)
        fn = rv_model.central_rv_synthetic
        synthetic = logger_decorator()(fn)(self.x_data_reduced, self.observer, **kwargs)
        return synthetic

    def central_rv_model_to_fit(self, xn):
        """
        Residual function.

        :param xn: numpy.array; current vector
        :return: numpy.array;
        """
        synthetic = self.prepare_synthetic(xn)
        return np.array([np.sum(np.power((synthetic[comp][self.x_data_reducer[comp]] - self.y_data[comp])
                                         / self.y_err[comp], 2)) for comp in synthetic.keys()])

    def fit(self, data: Dict[str, RVData], x0: parameters.BinaryInitialParameters, **kwargs):
        """
        Method to provide fitting of radial velocities curves.
        It can handle standadrd physical parameters `M_1`, `M_2` or astro community parameters `asini` and `q`.
        Based on non-linear least squares
        [https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html]

        :param data: elisa.analytics.dataset.base.RVData;
        :param x0: elisa.analytics.params.parameters.BinaryInitialParameters;
        :param kwargs: optional arguments for least_squares function (see documentation for
                       scipy.optimize.least_squares method)
        :return: Dict;
        """
        self.set_up(x0, data, observer_system_cls=RadialVelocitySystem)
        logger.info("fitting radial velocity light curve...")
        func = self.central_rv_model_to_fit
        initial_vector = parameters.vector_normalizer(self.initial_vector, self.fitable.keys(), self.normalization)
        result = least_squares(func, initial_vector, jac=kwargs.get('jac', '2-point'), bounds=(0, 1),
                               method=kwargs.get('method', 'trf'), ftol=kwargs.get('ftol', 1e-8),
                               xtol=kwargs.get('xtol', 1e-8), gtol=kwargs.get('gtol', 1e-8),
                               x_scale=kwargs.get('x_scale', 1.0), loss=kwargs.get('loss', 'linear'),
                               f_scale=kwargs.get('f_scale', 1.0), diff_step=kwargs.get('diff_step', None),
                               tr_solver=kwargs.get('tr_solver', None), tr_options=kwargs.get('tr_options', {}),
                               jac_sparsity=kwargs.get('jac_sparsity', None), max_nfev=kwargs.get('max_nfev', None),
                               verbose=kwargs.get('verbose', 0), args=kwargs.get('args', ()),
                               kwargs=kwargs.get('kwargs', {}))
        logger.info("fitting finished...")
        result = parameters.vector_renormalizer(result.x, self.fitable.keys(), self.normalization)

        # this relies on dict ordering (python >= 3.6)
        result_dict = {lbl: {
            'value': result[i],
            'fixed': False,
            'unit': self.fitable[lbl].to_dict()['unit'],
            'min': self.fitable[lbl].min,
            'max': self.fitable[lbl].max,
        } for i, lbl in enumerate(self.fitable.keys())}

        result_dict.update({lbl: {
            "value": val.value,
            'fixed': True,
            'unit': val.to_dict()['unit']
        } for lbl, val in self.fixed.items()})
        result_dict = self.eval_constrained_results(result_dict, self.constrained)

        r_squared_args = self.x_data_reduced, self.y_data, self.x_data_reducer, self.observer.system_cls
        r_dict = {key: value['value'] for key, value in result_dict.items()}

        r_squared_result = rv_r_squared(rv_model.central_rv_synthetic, *r_squared_args, **r_dict)
        result_dict["r_squared"] = {'value': r_squared_result, "unit": None}

        setattr(self, 'flat_result', result_dict)
        return parameters.serialize_result(result_dict)
