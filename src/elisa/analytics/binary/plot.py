import numpy as np

from corner import corner as _corner
from matplotlib import pyplot as plt
from ...analytics.binary import params


class Plot(object):
    @staticmethod
    def _renormalize_flat_chain(flat_chain, labels, renorm=None):
        """
        Renormalize values in chain if renormalization Dict is supplied.
        """
        if renorm is not None:
            return [[params.renormalize_value(val, renorm[key][0], renorm[key][1])
                     for key, val in zip(labels, sample)] for sample in flat_chain]

    @staticmethod
    def corner(flat_chain, labels, plot_datapoints=True, quantiles=None, renorm=None, truths=None):
        """
        Evaluate mcmc corner plot.

        :param flat_chain: numpy.array; flatted array of parameters values in each mcmc step::

            [[param0_0, param1_0, ..., paramk_0],
            [param0_1, param1_1, ..., paramk_1], ...,
            [param0_b, param1_n, ..., paramk_n]]

        :param labels: Union[List, numpy.array]; labels of parameters in order of params in `flat_chain`
        :param plot_datapoints: bool;
        :param quantiles: Union[List, numpy.array];
        :param renorm: Dict[str, Tuple(float, float)];
        :param truths: List[float], real value to be ploted
        """
        flat_chain = Plot._renormalize_flat_chain(flat_chain, labels, renorm)
        quantiles = [0.16, 0.5, 0.84] if quantiles is None else quantiles
        labels = [params.PARAMS_KEY_TEX_MAP[label] for label in labels]
        _corner(flat_chain, show_titles=True, labels=labels, plot_datapoints=plot_datapoints,
                quantiles=quantiles, truths=truths)
        plt.show()

    @staticmethod
    def paramtrace(flat_chain, labels, param, renorm=None):
        """
        Show value of parameter in mcmc chain.

        :param flat_chain: numpy.array; flatted array of parameters values in each mcmc step::

            [[param0_0, param1_0, ..., paramk_0],
            [param0_1, param1_1, ..., paramk_1], ...,
            [param0_b, param1_n, ..., paramk_n]]

        :param labels: Union[List, numpy.array]; labels of parameters in order of params in `flat_chain`
        :param param: str; param to plot
        :param renorm: :param renorm: Dict[str, Tuple(float, float)];
        """
        hash_map = {label: idx for idx, label in enumerate(labels)}
        flat_chain = Plot._renormalize_flat_chain(flat_chain, labels, renorm)
        ys = np.array(flat_chain).T[hash_map[param]]
        xs = np.arange(len(ys))

        plt.plot(xs, ys)
        plt.xlabel('N')
        plt.ylabel(params.PARAMS_KEY_TEX_MAP[param])
        plt.legend(loc=1)
        plt.grid()
        plt.show()

    trace = paramtrace
