import numpy as np

from abc import ABCMeta, abstractmethod
from ...binary_system.system import BinarySystem
from ...observer.observer import Observer
from ...analytics.binary import (
    utils as analutils
)


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
    xs, ys, period, passband, discretization, morphology, xs_reverser = args
    observed_means = np.array([np.repeat(np.mean(ys[band]), len(ys[band])) for band in ys])
    variability = np.sum([np.sum(np.power(ys[band] - observed_means, 2)) for band in ys])

    observer = Observer(passband=passband, system=None)
    observer._system_cls = BinarySystem

    synthetic = synthetic(xs, period, discretization, morphology, observer, False, **x)
    synthetic = {band: synthetic[band][xs_reverser[band]] for band in synthetic}

    synthetic = analutils.normalize_lightcurve_to_max(synthetic)
    residual = np.sum([np.sum(np.power(synthetic[band] - ys[band], 2)) for band in ys])
    return 1.0 - (residual / variability)


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
    observed_means = np.array([np.repeat(np.mean(ys[comp]), len(ys[comp])) for comp in ys.keys()])
    variability = np.sum([np.sum(np.power(ys[comp] - observed_means, 2)) for comp in ys.keys()])

    observer = Observer(passband='bolometric', system=None)
    observer._system_cls = BinarySystem
    synthetic = synthetic(xs, observer, **x)
    synthetic = {comp: synthetic[comp][xs_reverser[comp]] for comp in synthetic}

    if on_normalized:
        synthetic = analutils.normalize_rv_curve_to_max(synthetic)

    residual = np.sum([np.sum(np.power(synthetic[comp] - ys[comp], 2)) for comp in ys.keys()])
    return 1.0 - (residual / variability)
