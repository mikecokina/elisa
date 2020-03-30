import numpy as np
from scipy import interpolate

from abc import ABCMeta, abstractmethod
from ...binary_system.system import BinarySystem
from ...observer.observer import Observer
from ...analytics.binary import utils as analutils


class AbstractFit(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, *args, **kwargs):
        pass


class AbstractFitDataMixin(object):
    xs = list()
    ys = dict()
    yerrs = np.nan
    fixed = dict()
    constraint = dict()
    labels = list()
    observer = None
    xs_reverser = list()
    fit_xs = np.ndarray


class AbstractCentralRadialVelocityDataMixin(AbstractFitDataMixin):
    on_normalized = False


class AbstractLightCurveDataMixin(AbstractFitDataMixin):
    morphology = ''
    discretization = np.nan
    passband = ''
    period = np.nan


def lc_r_squared(synthetic, *args, **x):
    """
    Compute R^2 (coefficient of determination).

    :param synthetic: callable; synthetic method
    :param args: Tuple;
    :**args**:
        * **xs** * -- numpy.array; phases
        * **ys** * -- numpy.array; supplied fluxes (lets say fluxes from observation) normalized to max value
        * **period** * -- float;
        * **passband** * -- Union[str, List[str]];
        * **discretization** * -- flaot;
    :param x: Dict;
    :** x options**: kwargs of current parameters to compute binary system
    :return: float;
    """
    xs, ys, period, passband, discretization, morphology, xs_reverser, fit_xs = args

    observer = Observer(passband=passband, system=None)
    observer._system_cls = BinarySystem

    synthetic = synthetic(fit_xs, period, discretization, morphology, observer, False, **x)

    if np.shape(fit_xs) != np.shape(xs):
        new_synthetic = dict()
        for fltr, curve in synthetic.items():
            f = interpolate.interp1d(fit_xs, curve, kind='cubic')
            new_synthetic[fltr] = f(xs)
        synthetic = new_synthetic

    synthetic = {band: synthetic[band][xs_reverser[band]] for band in synthetic}

    synthetic = analutils.normalize_light_curve(synthetic, kind='average')
    return r_squared(synthetic, ys)


def rv_r_squared(synthetic, *args, **x):
    """
    Compute R^2 (coefficient of determination).

    :param synthetic: callable; synthetic method
    :param args: Tuple;
    :**args**:
        * **xs** * -- numpy.array; phases
        * **ys** * -- numpy.array; supplied fluxes (lets say fluxes from observation) normalized to max value
        * **period** * -- float;
        * **on_normalized** * -- bool;
    :param x: Dict;
    :** x options**: kwargs of current parameters to compute radial velocities curve
    :return: float;
    """
    xs, ys, on_normalized, xs_reverser = args

    observer = Observer(passband='bolometric', system=None)
    observer._system_cls = BinarySystem
    synthetic = synthetic(xs, observer, **x)
    synthetic = {comp: synthetic[comp][xs_reverser[comp]] for comp in synthetic}

    if on_normalized:
        synthetic = analutils.normalize_rv_curve_to_max(synthetic)

    return r_squared(synthetic, ys)


def r_squared(synthetic, observed):
    variability = np.sum([np.sum(np.power(observed[item] - np.mean(observed[item]), 2)) for item in observed])
    residual = np.sum([np.sum(np.power(synthetic[item] - observed[item], 2)) for item in observed])

    return 1 - (residual / variability)
