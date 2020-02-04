import numpy as np

from corner import corner as _corner
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt

from elisa.analytics.binary import params


class Plot(object):
    @staticmethod
    def corner(flat_chain, labels, renorm, quantiles=None, **kwargs):
        """
        Evaluate mcmc corner plot.

        :param flat_chain: numpy.array; flatted array of parameters values in each mcmc step::

            [[param0_0, param1_0, ..., paramk_0],
            [param0_1, param1_1, ..., paramk_1], ...,
            [param0_b, param1_n, ..., paramk_n]]

        :param labels: Union[List, numpy.array]; labels of parameters in order of params in `flat_chain`
        :param quantiles: Union[List, numpy.array];
        :param renorm: Dict[str, Tuple(float, float)]; {var_name: (min_boundary, max_boundary), ...}
        """
        flat_chain = params.renormalize_flat_chain(flat_chain, labels, renorm)
        quantiles = [0.16, 0.5, 0.84] if quantiles is None else quantiles
        labels = [params.PARAMS_KEY_TEX_MAP[label] for label in labels]
        _corner(flat_chain, show_titles=True, labels=labels, quantiles=quantiles, **kwargs)
        plt.show()

    @staticmethod
    def paramtrace(traces_to_show=None, flat_chain=None, variable_labels=None, normalization=None):
        """
        Show value of parameter in mcmc chain.

        :param flat_chain: numpy.array; flatted array of parameters values in each mcmc step::

            [[param0_0, param1_0, ..., paramk_0],
            [param0_1, param1_1, ..., paramk_1], ...,
            [param0_b, param1_n, ..., paramk_n]]

        :param variable_labels: Union[List, numpy.array]; labels of parameters in order of params in `flat_chain`
        :param renorm: Dict[str, Tuple(float, float)];
        """
        flat_chain = params.renormalize_flat_chain(flat_chain, variable_labels, normalization)
        hash_map = {label: idx for idx, label in enumerate(variable_labels) if label in traces_to_show}

        height = len(traces_to_show)
        fig = plt.figure(figsize=(8, 2.5*height))

        gs = gridspec.GridSpec(height, 1)
        ax = []
        labels = [params.PARAMS_KEY_TEX_MAP[label] for label in traces_to_show]
        for idx, label in enumerate(traces_to_show):
            ax.append(fig.add_subplot(gs[idx]))
            # ax[-1].plot(flat_chain[:, hash_map[label]], label=labels[idx])
            ax[-1].scatter(np.arange(flat_chain.shape[0]), flat_chain[:, hash_map[label]], label=labels[idx], s=0.2)
            ax[-1].legend(loc=1)
            ax[-1].set_ylabel(labels[idx])

        ax[-1].set_xlabel('N')

        plt.subplots_adjust(right=1.0, top=1.0, hspace=0)
        plt.show()

    trace = paramtrace