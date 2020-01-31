from ...logger import getLogger
from ... import (
    utils,
)
from elisa.analytics.binary.least_squares import central_rv

from elisa.analytics.binary_fit.plot import RVPlot

logger = getLogger('analytics.binary_fit.rv_fit')


class RVFit(object):
    MANDATORY_KWARGS = ['radial_velocities', ]
    OPTIONAL_KWARGS = []
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    def __init__(self, **kwargs):
        utils.invalid_kwarg_checker(kwargs, RVFit.ALL_KWARGS, RVFit)
        utils.check_missing_kwargs(self.__class__.MANDATORY_KWARGS, kwargs, instance_of=self.__class__)
        # kwargs = RVFit.transform_input(**kwargs)

        self.radial_velocities = None
        self.rv_fit_params = None

        self.plot = RVPlot(self)

        # values of properties
        logger.debug(f"setting properties of orbit")
        for kwarg in kwargs:
            setattr(self, kwarg, kwargs[kwarg])

    def fit(self, X0, method='least_squares', **kwargs):
        x_data, y_data, yerr = dict(), dict(), dict()
        for component, data in self.radial_velocities.items():
            x_data[component] = data.x_data
            y_data[component] = data.y_data
            yerr[component] = data.yerr
        if method == 'least_squares':
            self.rv_fit_params = central_rv.fit(xs=x_data, ys=y_data, x0=X0, yerr=yerr, **kwargs)

        elif method == 'mcmc':
            pass

        return self.rv_fit_params

