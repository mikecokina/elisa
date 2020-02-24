import numpy as np

from elisa.analytics.binary.mcmc import McMcMixin
from elisa.analytics.binary import params


def load_mcmc_chain(fit_instance, filename):
    filename = filename[:-5] if filename[-5:] == '.json' else filename
    data = McMcMixin.restore_flat_chain(fname=filename)
    fit_instance.flat_chain = np.array(data['flat_chain'])
    fit_instance.variable_labels = data['labels']
    fit_instance.normalization = data['normalization']

    # reproducing results from chain
    params.update_normalization_map(fit_instance.normalization)
    fit_instance.fit_params.update(McMcMixin.resolve_mcmc_result(flat_chain=fit_instance.flat_chain,
                                                                 labels=fit_instance.variable_labels))

    return fit_instance.flat_chain, fit_instance.variable_labels, fit_instance.normalization