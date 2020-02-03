from ...logger import getLogger
from ... import (
    utils,
)
from elisa.analytics.binary.least_squares import central_rv as lstsqr_central_rv
from elisa.analytics.binary.mcmc import central_rv as mcmc_central_rv

from elisa.analytics.binary_fit.plot import RVPlot
from elisa.analytics.binary.mcmc import McMcMixin
from elisa.analytics.binary import params

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
        """
        Function encapsulates various fitting functions for fitting radial velocities
        :param X0: dict; starting values of the fit
        :param method: string;
        :param kwargs: dict;
        :return: dict: fit_params
        """
        x_data, y_data, yerr = dict(), dict(), dict()
        for component, data in self.radial_velocities.items():
            x_data[component] = data.x_data
            y_data[component] = data.y_data
            yerr[component] = data.yerr

        if method == 'least_squares':
            self.rv_fit_params = lstsqr_central_rv.fit(xs=x_data, ys=y_data, x0=X0, yerr=yerr, **kwargs)

        elif method == 'mcmc':
            self.rv_fit_params = mcmc_central_rv.fit(xs=x_data, ys=y_data, x0=X0, yerr=yerr, **kwargs)

            # filtering for free variable names
            # xfree = params.x0_to_variable_kwargs(X0)
            # result = McMcMixin.resolve_mcmc_result(sampler=sampler, labels=xfree)
            # # adding fixed and constrained values to the result dictionary
            # result.update()
            # self.rv_fit_params = result

        return self.rv_fit_params

