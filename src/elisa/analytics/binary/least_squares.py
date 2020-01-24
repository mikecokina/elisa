import functools
import numpy as np

from abc import ABCMeta
from scipy.optimize import least_squares

from ...conf.config import BINARY_COUNTERPARTS
from ...logger import getPersistentLogger
from ..binary import params
from ..binary import (
    utils as analutils,
    models,
    shared
)
from elisa.analytics.binary.shared import (
    AbstractCentralRadialVelocityDataMixin,
    AbstractLightCurveDataMixin)

logger = getPersistentLogger('analytics.binary.fit')


def logger_decorator(suppress_logger=False):
    def do(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not suppress_logger:
                logger.info(f'current xn value: {kwargs}')
            return func(*args, **kwargs)
        return wrapper
    return do


class LightCurveFit(AbstractLightCurveDataMixin, metaclass=ABCMeta):
    def model_to_fit(self, xn):
        """
        Model to find minimum.

        :param xn: Iterable[float];
        :return: float;
        """
        xn = params.param_renormalizer(xn, self.labels)
        kwargs = params.prepare_kwargs(xn, self.labels, self.constraint, self.fixed)
        fn = models.synthetic_binary
        args = self.xs, self.period, self.discretization, self.morphology, self.observer, False
        try:
            synthetic = logger_decorator()(fn)(*args, **kwargs)
            synthetic = analutils.normalize_lightcurve_to_max(synthetic)

        except Exception as e:
            logger.error(f'your initial parmeters lead during fitting to invalid binary system')
            raise RuntimeError(f'your initial parmeters lead during fitting to invalid binary system: {str(e)}')

        residua = np.array([np.sum(np.power(synthetic[band][self.xs_reverser[band]] - self.ys[band], 2)
                                   / self.yerrs[band]) for band in synthetic])

        return residua

    def fit(self, xs, ys, period, x0, discretization, yerrs=None, xtol=1e-8, ftol=1e-8, max_nfev=None,
            diff_step=None, f_scale=1.0):
        """
        Fit method using non-linear least squares.
        Based on https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

        :param xs: Dict[str, Iterable[float]]; {<passband>: <phases>}
        :param ys: Dict[str, Iterable[float]]; {<passband>: <fluxes>};
        :param period: float; sytem period
        :param x0: List[Dict]; initial state (metadata included)
        :param discretization: float; discretization of objects
        :param xtol: float; relative tolerance to consider solution
        :param yerrs: Union[numpy.array, float]; errors for each point of observation
        :param max_nfev: int; maximal iteration
               https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
        :param diff_step: Union[None, array];
               https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
        :param f_scale: float; optional
               https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
        :param ftol: float;
               https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
        :param xtol: float; tolerance of error to consider hitted solution as exact
               https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
        :return: Dict;
        """

        passband = list(ys.keys())
        # compute yerrs if not supplied
        yerrs = {band: analutils.lightcurves_mean_error(ys) for band in passband} if yerrs is None else yerrs

        self.xs, self.xs_reverser = params.xs_reducer(xs)
        self.ys, self.yerrs = ys, yerrs

        x0 = params.lc_initial_x0_validity_check(x0, self.morphology)
        x0, labels, fixed, constraint, observer = params.fit_data_initializer(x0, passband=passband)

        self.period = period
        self.discretization = discretization
        self.passband = passband
        self.labels, self.fixed, self.constraint = labels, fixed, constraint
        self.observer = observer

        # evaluate least squares from scipy
        logger.info("fitting circular synchronous system...")
        result = least_squares(self.model_to_fit, x0, bounds=(0, 1), max_nfev=max_nfev, xtol=xtol,
                               ftol=ftol, diff_step=diff_step, f_scale=f_scale)
        logger.info("fitting finished")

        # put all together `floats`, `fixed` and `constraints`
        result = params.param_renormalizer(result.x, labels)
        result_dict = dict(zip(labels, result))
        result_dict.update(self.fixed)
        result_dict.update(params.constraints_evaluator(result_dict, self.constraint))
        result = [{"param": key, "value": val} for key, val in result_dict.items()]

        # compute r_squared and append to result
        r_squared_args = self.xs, self.ys, period, self.passband, discretization, self.morphology, self.xs_reverser
        r_squared_result = shared.lc_r_squared(models.synthetic_binary, *r_squared_args, **result_dict)
        result.append({"r_squared": r_squared_result})

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
    def centarl_rv_model_to_fit(self, xn):
        """
        Residual function.

        :param xn: numpy.array; current vector
        :return: numpy.array;
        """
        xn = params.param_renormalizer(xn, self.labels)
        kwargs = params.prepare_kwargs(xn, self.labels, self.constraint, self.fixed)
        fn = models.central_rv_synthetic
        synthetic = logger_decorator()(fn)(self.xs, self.observer, **kwargs)
        if self.on_normalized:
            synthetic = analutils.normalize_rv_curve_to_max(synthetic)
        return np.array([np.sum(np.power((synthetic[comp][self.xs_reverser[comp]] - self.ys[comp])
                                         / self.yerrs[comp], 2)) for comp in BINARY_COUNTERPARTS])

    def fit(self, xs, ys, x0, yerrs=None, xtol=1e-8, ftol=1e-8, max_nfev=None, diff_step=None,
            f_scale=1.0, on_normalized=False):
        """
        Method to provide fitting of radial velocities curves.
        It can handle standadrd physical parameters `M_1`, `M_2` or astro community parameters `asini` and `q`.
        Based on non-linear least squares
        [https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html]

        :param on_normalized: bool; if True, fitting is provided on normalized radial velocities curves
        :param xs: Iterable[float];
        :param ys: Dict;
        :param x0: List[Dict]; initial state (metadata included)
        :param yerrs: Union[numpy.array, float]; errors for each point of observation
        :param max_nfev: int; maximal iteration
               https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
        :param diff_step: Union[None, array];
               https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
        :param f_scale: float; optional
               https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
        :param ftol: float;
               https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
        :param xtol: float; tolerance of error to consider hitted solution as exact
               https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
        :return: Dict;
        """
        x0 = params.rv_initial_x0_validity_check(x0)
        yerrs = {c: analutils.radialcurves_mean_error(ys) for c in BINARY_COUNTERPARTS} if yerrs is None else yerrs
        x0, labels, fixed, constraint, observer = params.fit_data_initializer(x0)

        self.xs, self.xs_reverser = params.xs_reducer(xs)
        self.ys, self.yerrs = ys, yerrs
        self.labels, self.observer = labels, observer
        self.fixed, self.constraint = fixed, constraint

        logger.info("fitting radial velocity light curve...")
        func = self.centarl_rv_model_to_fit
        result = least_squares(fun=func, x0=x0, bounds=(0, 1), max_nfev=max_nfev,
                               xtol=xtol, ftol=ftol, diff_step=diff_step, f_scale=f_scale)
        logger.info("fitting finished")

        result = params.param_renormalizer(result.x, labels)
        result_dict = dict(zip(labels, result))
        result_dict.update(self.fixed)
        result_dict.update(params.constraints_evaluator(result_dict, self.constraint))

        r_squared_args = self.xs, self.ys, on_normalized, self.xs_reverser
        r_squared_result = shared.rv_r_squared(models.central_rv_synthetic, *r_squared_args, **result_dict)

        # result = [{"param": key, "value": val} for key, val in result_dict.items()]
        result = {key: {"value": val} for key, val in result_dict.items()}
        result["r_squared"] = {'value': r_squared_result}
        return params.extend_result_with_units(result)


binary_detached = DetachedLightCurveFit()
binary_overcontact = OvercontactLightCurveFit()
central_rv = CentralRadialVelocity()
