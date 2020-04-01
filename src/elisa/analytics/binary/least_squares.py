import functools
import numpy as np
from scipy import interpolate

from abc import ABCMeta
from scipy.optimize import least_squares

from ...logger import getPersistentLogger
from ..binary import params
from ..binary import (
    utils as butils,
    models,
    shared
)

from elisa.analytics.binary.shared import (
    AbstractCentralRadialVelocityDataMixin,
    AbstractLightCurveDataMixin, AbstractFit)

from elisa.conf import config

logger = getPersistentLogger('analytics.binary.least_squares')


def logger_decorator(suppress_logger=False):
    def do(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not suppress_logger:
                logger.debug(f'current xn value: {kwargs}')
            return func(*args, **kwargs)
        return wrapper
    return do


class LightCurveFit(AbstractFit, AbstractLightCurveDataMixin, metaclass=ABCMeta):
    def model_to_fit(self, xn):
        """
        Model to find minimum.

        :param xn: Iterable[float];
        :return: float;
        """
        xn = params.param_renormalizer(xn, self.labels)
        kwargs = params.prepare_kwargs(xn, self.labels, self.constraint, self.fixed)

        phases, kwargs = models.rvt_layer_resolver(self.xs, **kwargs)
        fit_xs = np.linspace(np.min(phases) - self.diff, np.max(phases) + self.diff, num=self.interp_treshold + 2) \
            if np.shape(phases)[0] > self.interp_treshold else phases

        fn = models.synthetic_binary
        args = fit_xs, self.discretization, self.morphology, self.observer, False
        try:
            synthetic = logger_decorator()(fn)(*args, **kwargs)
            synthetic = butils.normalize_light_curve(synthetic, kind='average')

        except Exception as e:
            logger.error(f'your initial parmeters lead during fitting to invalid binary system')
            raise RuntimeError(f'your initial parmeters lead during fitting to invalid binary system: {str(e)}')

        if np.shape(fit_xs) != np.shape(phases):
            new_synthetic = dict()
            for fltr, curve in synthetic.items():
                f = interpolate.interp1d(fit_xs, curve, kind='cubic')
                new_synthetic[fltr] = f(phases)
            synthetic = new_synthetic

        residuals = np.array([np.sum(np.power(synthetic[band][self.xs_reverser[band]] - self.ys[band], 2)
                              / self.yerrs[band]) for band in synthetic])

        r2 = shared.r_squared(synthetic, self.ys)
        logger.info(f'current R2: {r2}')

        return residuals

    def fit(self, xs, ys, x0, discretization, yerr=None, interp_treshold=None, **kwargs):
        """
        Fit method using non-linear least squares.
        Based on https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

        :param xs: Dict[str, Iterable[float]]; {<passband>: <phases>}
        :param ys: Dict[str, Iterable[float]]; {<passband>: <fluxes>};
        :param x0: Dict[Dict]; initial state (metadata included)
        :param discretization: float; discretization of objects
        :param yerr: Union[numpy.array, float]; errors for each point of observation
        :param interp_treshold: int; data binning treshold
        :param kwargs: optional arguments for least_squares function (see documentation for
                       scipy.optimize.least_squares method)
        :return: Dict;
        """
        passband = list(ys.keys())
        # compute yerrs if not supplied
        yerrs = {c: butils.lightcurves_mean_error(ys[c]) if yerr[c] is None else yerr[c]
                 for c in xs.keys()}

        ys = butils.normalize_light_curve(ys, kind='average')
        self.xs, self.xs_reverser = params.xs_reducer(xs)
        self.ys, self.yerrs = ys, yerrs

        x0 = params.lc_initial_x0_validity_check(x0, self.morphology)
        x0_vector, labels, fixed, constraint, observer = params.fit_data_initializer(x0, passband=passband)

        self.discretization = discretization
        self.passband = passband
        self.labels, self.fixed, self.constraint = labels, fixed, constraint
        self.observer = observer

        self.interp_treshold = config.MAX_CURVE_DATA_POINTS if interp_treshold is None else interp_treshold
        self.diff = 1.0 / self.interp_treshold

        # evaluate least squares from scipy
        logger.info("fitting started...")
        result = least_squares(self.model_to_fit, x0_vector, jac=kwargs.get('jac', '2-point'), bounds=(0, 1),
                               method=kwargs.get('method', 'trf'), ftol=kwargs.get('ftol', 1e-8),
                               xtol=kwargs.get('xtol', 1e-8), gtol=kwargs.get('gtol', 1e-8),
                               x_scale=kwargs.get('x_scale', 1.0), loss=kwargs.get('loss', 'linear'),
                               f_scale=kwargs.get('f_scale', 1.0), diff_step=kwargs.get('diff_step', None),
                               tr_solver=kwargs.get('tr_solver', None), tr_options=kwargs.get('tr_options', {}),
                               jac_sparsity=kwargs.get('jac_sparsity', None), max_nfev=kwargs.get('max_nfev', None),
                               verbose=kwargs.get('verbose', 2), args=kwargs.get('args', ()),
                               kwargs=kwargs.get('kwargs', {}))
        logger.info("fitting finished")

        # put all together `floats`, `fixed` and `constraints`
        result = params.param_renormalizer(result.x, labels)
        result_dict = dict(zip(labels, result))
        result_dict.update(self.fixed)
        result_dict.update(params.constraints_evaluator(result_dict, self.constraint))

        result = {key: {"value": val} for key, val in result_dict.items()}

        # compute r_squared and append to result
        r_squared_args = self.xs, self.ys, self.passband, discretization, self.morphology, self.xs_reverser, \
                         self.diff, self.interp_treshold
        r_squared_result = shared.lc_r_squared(models.synthetic_binary, *r_squared_args, **result_dict)

        result["r_squared"] = {'value': r_squared_result}
        result = params.dict_to_user_format(result)
        return params.extend_result_with_units(result)


class OvercontactLightCurveFit(LightCurveFit):
    def __init__(self):
        super().__init__()
        self.morphology = 'over-contact'


class DetachedLightCurveFit(LightCurveFit):
    def __init__(self):
        super().__init__()
        self.morphology = 'detached'


class CentralRadialVelocity(AbstractCentralRadialVelocityDataMixin):
    def prep_params(self, xn):
        xn = params.param_renormalizer(xn, self.labels)
        kwargs = params.prepare_kwargs(xn, self.labels, self.constraint, self.fixed)
        fn = models.central_rv_synthetic
        synthetic = logger_decorator()(fn)(self.xs, self.observer, **kwargs)
        if self.on_normalized:
            synthetic = butils.normalize_rv_curve_to_max(synthetic)
        return synthetic

    def central_rv_model_to_fit(self, xn):
        """
        Residual function.

        :param xn: numpy.array; current vector
        :return: numpy.array;
        """
        synthetic = self.prep_params(xn)
        return np.array([np.sum(np.power((synthetic[comp][self.xs_reverser[comp]] - self.ys[comp])
                                         / self.yerrs[comp], 2)) for comp in synthetic.keys()])

    def fit(self, xs, ys, x0, yerr=None, on_normalized=False, **kwargs):
        """
        Method to provide fitting of radial velocities curves.
        It can handle standadrd physical parameters `M_1`, `M_2` or astro community parameters `asini` and `q`.
        Based on non-linear least squares
        [https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html]

        :param on_normalized: bool; if True, fitting is provided on normalized radial velocities curves
        :param xs: Iterable[float];
        :param ys: Dict; {'primary': np.array(), 'secondary': np.array()}
        :param x0: Dict; initial state (metadata included)
        :param yerr: Union[numpy.array, float]; errors for each point of observation
        :param kwargs: optional arguments for least_squares function (see documentation for
                       scipy.optimize.least_squares method)
        :return: Dict;
        """
        x0 = params.rv_initial_x0_validity_check(x0)
        yerrs = {c: butils.radialcurves_mean_error(ys[c]) if yerr[c] is None else yerr[c]
                 for c in xs.keys()}
        x0_vector, labels, fixed, constraint, observer = params.fit_data_initializer(x0)

        self.xs, self.xs_reverser = params.xs_reducer(xs)
        self.ys, self.yerrs = ys, yerrs
        self.labels, self.observer = labels, observer
        self.fixed, self.constraint = fixed, constraint

        logger.info("fitting radial velocity light curve...")
        func = self.central_rv_model_to_fit
        result = least_squares(func, x0_vector, jac=kwargs.get('jac', '2-point'), bounds=(0, 1),
                               method=kwargs.get('method', 'trf'), ftol=kwargs.get('ftol', 1e-8),
                               xtol=kwargs.get('xtol', 1e-8), gtol=kwargs.get('gtol', 1e-8),
                               x_scale=kwargs.get('x_scale', 1.0), loss=kwargs.get('loss', 'linear'),
                               f_scale=kwargs.get('f_scale', 1.0), diff_step=kwargs.get('diff_step', None),
                               tr_solver=kwargs.get('tr_solver', None), tr_options=kwargs.get('tr_options', {}),
                               jac_sparsity=kwargs.get('jac_sparsity', None), max_nfev=kwargs.get('max_nfev', None),
                               verbose=kwargs.get('verbose', 0), args=kwargs.get('args', ()),
                               kwargs=kwargs.get('kwargs', {}))
        logger.info("fitting finished...")

        result = params.param_renormalizer(result.x, labels)

        result_dict = dict(zip(labels, result))
        result_dict.update(self.fixed)
        result_dict.update(params.constraints_evaluator(result_dict, self.constraint))

        r_squared_args = self.xs, self.ys, on_normalized, self.xs_reverser
        r_squared_result = shared.rv_r_squared(models.central_rv_synthetic, *r_squared_args, **result_dict)

        result = {key: {"value": val} for key, val in result_dict.items()}
        result["r_squared"] = {'value': r_squared_result}
        result = params.dict_to_user_format(result)
        return params.extend_result_with_units(result)


binary_detached = DetachedLightCurveFit()
binary_overcontact = OvercontactLightCurveFit()
central_rv = CentralRadialVelocity()
