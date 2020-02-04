import numpy as np
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
        self.fit_params = None

        # MCMC specific variables
        self.flat_chain = None
        self.normalization = None
        self.variable_labels = None

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
            self.fit_params = lstsqr_central_rv.fit(xs=x_data, ys=y_data, x0=X0, yerr=yerr, **kwargs)

        elif method == 'mcmc':
            self.fit_params = mcmc_central_rv.fit(xs=x_data, ys=y_data, x0=X0, yerr=yerr, **kwargs)
            self.flat_chain = mcmc_central_rv.last_sampler.get_chain(flat=True)
            self.normalization = mcmc_central_rv.last_normalization
            self.variable_labels = mcmc_central_rv.labels

        return self.fit_params

    def load_chain(self, filename):
        """
        Function loads MCMC chain along with auxiliary data from json file created after each MCMC run

        :param filename: str; full name of the json file
        :return: Tuple[numpy.ndarray, list, Dict]; flattened mcmc chain, labels of variables in `flat_chain` columns,
        {var_name: (min_boundary, max_boundary), ...} dictionary of boundaries defined by user for each variable needed
        to reconstruct real values from normalized `flat_chain` array
        """
        filename = filename[:-5] if filename[-5:] == '.json' else filename
        data = McMcMixin.restore_flat_chain(fname=filename)
        self.flat_chain = np.array(data['flat_chain'])
        self.variable_labels = data['labels']
        self.normalization = data['normalization']

        # reproducing results from chain
        params.update_normalization_map(self.normalization)
        self.fit_params = McMcMixin.resolve_mcmc_result(flat_chain=self.flat_chain, labels=self.variable_labels)

        return self.flat_chain, self.variable_labels, self.normalization

