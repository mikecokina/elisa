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
from .. models import cost_fns
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
    """
    General class for solving inverse problem in case of LC data.
    """
    MORPHOLOGY = None

    def model_to_fit(self, xn):
        """
        Cost function minimized during solution of the inverse problem using the Least Squares method.

        :param xn: Iterable[float]; vector containing normalized values of model parameters optimized during fit
        :return: float; error weighted sum of squares of residuals
        """
        diff = 1.0 / self.interp_treshold

        xn = parameters.vector_renormalizer(xn, self.fitable.keys(), self.normalization)
        kwargs = parameters.prepare_properties_set(xn, self.fitable.keys(), self.constrained, self.fixed)
        phases, kwargs = time_layer_resolver(self.x_data_reduced, pop=False, **kwargs)

        if self.fit_xs is None:
            fit_xs = np.linspace(np.min(phases) - diff, np.max(phases) + diff, num=self.interp_treshold + 2) \
                if np.shape(phases)[0] > self.interp_treshold else phases
        else:
            fit_xs = self.fit_xs
        args = fit_xs, self.discretization, self.observer
        fn = lc_model.synthetic_binary

        try:
            synthetic = logger_decorator()(fn)(*args, **kwargs)
        except Exception as e:
            logger.error(f'your initial parameters lead during fitting to invalid binary system, exectpion: {str(e)}')
            return const.MAX_USABLE_FLOAT

        if np.shape(fit_xs) != np.shape(phases):
            synthetic = {
                band: interpolate.interp1d(fit_xs, curve, kind='cubic')(phases[self.x_data_reducer[band]])
                for band, curve in synthetic.items()
            }
        else:
            synthetic = {band: val[self.x_data_reducer[band]] for band, val in synthetic.items()}

        synthetic, _ = normalize_light_curve(synthetic, kind='average')

        residuals = cost_fns.wssr(self.y_data, self.y_err, synthetic)

        logger.info(f'current R2: {r_squared(synthetic, self.y_data)}')

        return residuals

    def fit(self, data: Dict[str, LCData], x0: parameters.BinaryInitialParameters,
            discretization=5.0, interp_treshold=None, samples="uniform", **kwargs):
        """
        Fit method using non-linear least squares.
        Based on `https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html`.

        :param data: Dict[elisa.analytics.dataset.base.LCData]; observational data in photometric filters
        :param x0: BinaryInitialParameters; initial state of model parameters
        :param discretization: float; discretization factor used for the primary component
        :param interp_treshold: int; Above this total number of datapoints, light curve will be interpolated
                                     using model containing `interp_treshold` equidistant points per epoch
        :param samples: Union[str, List]; 'uniform' (equidistant sampling in phase), 'adaptive'
                                          (equidistant sampling on curve) or list with phases in (0, 1) interval
        :param kwargs: optional arguments for least_squares function (see documentation for
                       scipy.optimize.least_squares method)
        :return: Dict; optimized model parameters in standard JSON format
        """
        self.set_up(x0, data, passband=data.keys(), discretization=discretization, morphology=self.MORPHOLOGY,
                    interp_treshold=settings.MAX_CURVE_DATA_POINTS if interp_treshold is None else interp_treshold,
                    observer_system_cls=BinarySystem, samples=samples)
        initial_vector = parameters.vector_normalizer(self.initial_vector, self.fitable.keys(), self.normalization)

        # evaluate least squares from scipy
        logger.info("fitting started...")
        result = least_squares(self.model_to_fit, initial_vector, jac=kwargs.get('jac', '2-point'), bounds=(0, 1),
                               method=kwargs.get('method', 'trf'), ftol=kwargs.get('ftol', 1e-7),
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
    """
    Optimization class for solving an inverse problem for overcontact systems.
    """
    MORPHOLOGY = 'over-contact'


class DetachedLightCurveFit(LightCurveFit):
    """
    Optimization class for solving an inverse problem for detached systems.
    """
    MORPHOLOGY = 'detached'


class CentralRadialVelocity(AbstractRVFit):
    """
    Class for fitting radial velocities using kinematic method.
    """
    def prepare_synthetic(self, xn):
        """
        Returns synthetic radial velocity observations for given set of normalized variable parameters.

        :param xn: List[float]; variable model parameters in normalized form
        :return: Dict[numpy.array]; synthetic RV observations
        """
        xn = parameters.vector_renormalizer(xn, self.fitable.keys(), self.normalization)
        kwargs = parameters.prepare_properties_set(xn, self.fitable.keys(), self.constrained, self.fixed)
        fn = rv_model.central_rv_synthetic
        synthetic = logger_decorator()(fn)(self.x_data_reduced, self.observer, **kwargs)
        return synthetic

    def central_rv_model_to_fit(self, xn):
        """
        Cost function minimized during solution of the inverse problem using the Least Squares method.

        :param xn: List[float]; variable model parameters in normalized form
        :return: numpy.array; error weighted sum of squares of residuals
        """
        synthetic = self.prepare_synthetic(xn)
        synthetic = {comp: synthetic[comp][self.x_data_reducer[comp]] for comp in synthetic}
        return cost_fns.wssr(self.y_data, self.y_err, synthetic)

    def fit(self, data: Dict[str, RVData], x0: parameters.BinaryInitialParameters, **kwargs):
        """
        Method to provide fitting of radial velocities curves.
        It can handle standard physical parameters including component masses `M_1`, `M_2` or astro community
        parameters containing `asini` and `q`. Optimizer based on non-linear least squares method:
        [https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html]

        :param data: elisa.analytics.dataset.base.RVData; radial velocity observations for primary and secondary
                                                          component
        :param x0: elisa.analytics.params.parameters.BinaryInitialParameters; initial state of model parameters
        :param kwargs: optional arguments for least_squares function (see documentation for
                       scipy.optimize.least_squares method)
        :return: Dict; optimized model parameters in standard JSON format
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
