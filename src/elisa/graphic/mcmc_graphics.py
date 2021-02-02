import numpy as np
import matplotlib.gridspec as gridspec

from corner import corner as _corner
from matplotlib import pyplot as plt


class Plot(object):
    @staticmethod
    def corner(**kwargs):
        """
        Evaluate mcmc corner plot.

        """
        flat_chain = kwargs.pop('flat_chain')
        fit_params = kwargs.pop('fit_params')
        variable_labels = kwargs.pop('variable_labels')
        figure = _corner(flat_chain, **kwargs)

        # Extract the axes
        ndim = flat_chain.shape[1]
        axes = np.array(figure.axes).reshape((ndim, ndim))

        # Loop over the diagonal, adding units
        for i, label in enumerate(variable_labels):
            ax = axes[i, i]
            value = fit_params[label]['value']
            bottom = fit_params[label]["confidence_interval"]['min'] - value
            top = fit_params[label]["confidence_interval"]['max'] - value

            unit = fit_params[label]['unit']
            unit = '' if unit == 'dimensionless' or unit is None else unit
            if any(x in label for x in ['t_eff', 'argument_of_periastron']):
                title = r'{0}=${1:.0f}^{{{2:+.0f}}}_{{{3:+.0f}}}$ {4}'.format(kwargs['labels'][i], value, top, bottom,
                                                                              unit)
            else:
                title = r'{0}=${1:.2f}^{{{2:+.2f}}}_{{{3:+.2f}}}$ {4}'.format(kwargs['labels'][i], value, top, bottom,
                                                                              unit)
            ax.set_title(title)

        plt.show()

    @staticmethod
    def paramtrace(**kwargs):
        """
        Show traces of mcmc chain.
        """
        labels = kwargs['labels']
        hash_map = {label: idx for idx, label in enumerate(kwargs['variable_labels']) 
                    if label in kwargs['traces_to_plot']}

        height = len(kwargs['traces_to_plot'])
        fig = plt.figure(figsize=(8, 2.5*height))

        gs = gridspec.GridSpec(height, 1)
        ax = []

        plot_conter = 0
        for idx, label in enumerate(kwargs['variable_labels']):
            if label not in kwargs['traces_to_plot']:
                continue
            ax.append(fig.add_subplot(gs[plot_conter])) \
                if plot_conter == 0 else ax.append(fig.add_subplot(gs[plot_conter], sharex=ax[0]))
            plot_conter += 1
            ax[-1].scatter(np.arange(kwargs['flat_chain'].shape[0]), kwargs['flat_chain'][:, hash_map[label]],
                           label=labels[idx], s=0.2)
            ax[-1].legend(loc=1)

            unit = kwargs['fit_params'][label]['unit']
            unit = '' if unit == 'dimensionless' or unit is None else ' / [{0}]'.format(unit)
            ax[-1].set_ylabel(f'{labels[idx]}{unit}')

            if kwargs['truths']:
                ax[-1].axhline(kwargs['fit_params'][label]['value'], linestyle='dashed', color='black')
                ax[-1].axhline(kwargs['fit_params'][label]["confidence_interval"]['min'],
                               linestyle='dotted', color='black')
                ax[-1].axhline(kwargs['fit_params'][label]["confidence_interval"]['max'],
                               linestyle='dotted', color='black')

        ax[-1].set_xlabel('N')

        plt.subplots_adjust(right=1.0, top=1.0, hspace=0)
        plt.show()

    @staticmethod
    def autocorr(**kwargs):
        """
        Show autocorrelation function.
        """
        labels = kwargs['labels']
        hash_map = {label: idx for idx, label in enumerate(kwargs['variable_labels']) if label in
                    kwargs['correlations_to_plot']}

        height = len(kwargs['correlations_to_plot'])
        fig = plt.figure(figsize=(8, 2.5 * height))

        gs = gridspec.GridSpec(height, 1)
        ax = []

        for idx, label in enumerate(kwargs['variable_labels']):
            if label not in kwargs['correlations_to_plot']:
                continue

            ax.append(fig.add_subplot(gs[idx])) if idx == 0 else ax.append(fig.add_subplot(gs[idx], sharex=ax[0]))

            lbl = 'corr_time = {0:.2f}'.format(kwargs['autocorr_time'][idx])
            ax[-1].scatter(np.arange(kwargs['autocorr_fns'].shape[0]), kwargs['autocorr_fns'][:, hash_map[label]],
                           label=lbl, s=0.2)
            ax[-1].set_ylabel(f'{labels[idx]} correlation fn')
            ax[-1].legend()

        ax[-1].set_xlabel('N')

        plt.subplots_adjust(right=1.0, top=1.0, hspace=0)
        plt.show()
