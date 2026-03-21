from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from corner import corner as _corner
from matplotlib import gridspec
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

    from elisa.types import Float


class Plot:
    """Utility plotting helpers for MCMC diagnostics and posterior summaries."""

    @staticmethod
    def corner(
        *,
        flat_chain: NDArray[Float],
        fit_params: dict[str, dict[str, Any]],
        variable_labels: list[str],
        labels: list[str],
        **kwargs: object,
    ) -> None:
        """Evaluate an MCMC corner plot.

        The function renders a corner plot from a flattened MCMC chain and adds
        formatted parameter summaries to the diagonal panels using fitted values,
        confidence intervals, and units.

        :param flat_chain: Flattened MCMC chain with shape ``(n_samples, ndim)``.
        :type flat_chain: NDArray[Float]
        :param fit_params: Mapping containing fitted parameter values,
            confidence intervals, and units keyed by variable label.
        :type fit_params: dict[str, dict[str, object]]
        :param variable_labels: Internal variable identifiers corresponding to
            the columns of ``flat_chain``.
        :type variable_labels: list[str]
        :param labels: Display labels used in the plot.
        :type labels: list[str]
        :param kwargs: Additional keyword arguments forwarded to
            :func:`corner.corner`.
        :type kwargs: object
        :return: ``None``.
        :rtype: None
        """
        figure = _corner(flat_chain, labels=labels, **kwargs)

        ndim = flat_chain.shape[1]
        axes = np.array(figure.axes).reshape((ndim, ndim))

        for index, label in enumerate(variable_labels):
            ax = axes[index, index]
            value = fit_params[label]["value"]
            confidence_interval = fit_params[label]["confidence_interval"]
            bottom = confidence_interval["min"] - value
            top = confidence_interval["max"] - value

            unit = fit_params[label]["unit"]
            unit = "" if unit == "dimensionless" or unit is None else unit

            if any(item in label for item in ["t_eff", "argument_of_periastron"]):
                title = rf"{labels[index]}=${value:.0f}^{{{top:+.0f}}}_{{{bottom:+.0f}}}$ {unit}"
            elif "eccentricity" in label:
                title = rf"{labels[index]}=${value:.3f}^{{{top:+.3f}}}_{{{bottom:+.3f}}}$ {unit}"
            else:
                title = rf"{labels[index]}=${value:.3f}^{{{top:+.3f}}}_{{{bottom:+.3f}}}$ {unit}"

            ax.set_title(title)

        plt.show()

    @staticmethod
    def paramtrace(
        *,
        flat_chain: NDArray[Float],
        fit_params: dict[str, dict[str, Any]],
        variable_labels: list[str],
        traces_to_plot: list[str],
        labels: list[str],
        truths: bool,
    ) -> None:
        """Show traces of an MCMC chain.

        The function plots selected parameter traces from a flattened MCMC chain.
        Optionally, it overlays the fitted central value and confidence interval
        bounds for each plotted parameter.

        :param flat_chain: Flattened MCMC chain with shape ``(n_samples, ndim)``.
        :type flat_chain: NDArray[Float]
        :param fit_params: Mapping containing fitted parameter values,
            confidence intervals, and units keyed by variable label.
        :type fit_params: dict[str, dict[str, object]]
        :param variable_labels: Internal variable identifiers corresponding to
            the columns of ``flat_chain``.
        :type variable_labels: list[str]
        :param traces_to_plot: Variable identifiers to include in the trace
            plot.
        :type traces_to_plot: list[str]
        :param labels: Display labels used in the plot.
        :type labels: list[str]
        :param truths: Whether to plot the fitted value and confidence interval
            bounds.
        :type truths: bool
        :return: ``None``.
        :rtype: None
        """
        hash_map = {label: idx for idx, label in enumerate(variable_labels) if label in traces_to_plot}

        height = len(traces_to_plot)
        fig = plt.figure(figsize=(8, 2.5 * height))
        gs = gridspec.GridSpec(height, 1)
        axes: list[Axes] = []

        plot_counter = 0
        for idx, label in enumerate(variable_labels):
            if label not in traces_to_plot:
                continue

            if plot_counter == 0:
                axes.append(fig.add_subplot(gs[plot_counter]))
            else:
                axes.append(fig.add_subplot(gs[plot_counter], sharex=axes[0]))

            axes[-1].scatter(
                np.arange(flat_chain.shape[0]),
                flat_chain[:, hash_map[label]],
                label=labels[idx],
                s=0.2,
            )
            axes[-1].legend(loc=1)

            unit = fit_params[label]["unit"]
            unit = "" if unit == "dimensionless" or unit is None else f" / [{unit}]"
            axes[-1].set_ylabel(f"{labels[idx]}{unit}")

            if truths:
                axes[-1].axhline(
                    fit_params[label]["value"],
                    linestyle="dashed",
                    color="black",
                )
                axes[-1].axhline(
                    fit_params[label]["confidence_interval"]["min"],
                    linestyle="dotted",
                    color="black",
                )
                axes[-1].axhline(
                    fit_params[label]["confidence_interval"]["max"],
                    linestyle="dotted",
                    color="black",
                )

            plot_counter += 1

        axes[-1].set_xlabel("N")

        plt.subplots_adjust(right=1.0, top=1.0, hspace=0)
        plt.show()

    @staticmethod
    def autocorr(
        *,
        autocorr_fns: NDArray[Float],
        autocorr_time: NDArray[Float],
        variable_labels: list[str],
        correlations_to_plot: list[str],
        labels: list[str],
    ) -> None:
        """Show autocorrelation functions for selected variables.

        The function plots autocorrelation functions for selected parameters and
        annotates each subplot with the corresponding correlation time.

        :param autocorr_fns: Autocorrelation function values with shape
            ``(n_lags, ndim_selected)`` or compatible indexing layout.
        :type autocorr_fns: NDArray[Float]
        :param autocorr_time: Autocorrelation times for plotted variables.
        :type autocorr_time: NDArray[Float]
        :param variable_labels: Internal variable identifiers corresponding to
            the chain parameters.
        :type variable_labels: list[str]
        :param correlations_to_plot: Variable identifiers to include in the
            autocorrelation plot.
        :type correlations_to_plot: list[str]
        :param labels: Display labels used in the plot.
        :type labels: list[str]
        :return: ``None``.
        :rtype: None
        """
        hash_map = {label: idx for idx, label in enumerate(variable_labels) if label in correlations_to_plot}

        height = len(correlations_to_plot)
        fig = plt.figure(figsize=(8, 2.5 * height))
        gs = gridspec.GridSpec(height, 1)
        axes: list[Axes] = []

        plot_counter = 0
        for idx, label in enumerate(variable_labels):
            if label not in correlations_to_plot:
                continue

            if plot_counter == 0:
                axes.append(fig.add_subplot(gs[plot_counter]))
            else:
                axes.append(fig.add_subplot(gs[plot_counter], sharex=axes[0]))

            correlation_label = f"corr_time = {autocorr_time[idx]:.2f}"
            axes[-1].scatter(
                np.arange(autocorr_fns.shape[0]),
                autocorr_fns[:, hash_map[label]],
                label=correlation_label,
                s=0.2,
            )
            axes[-1].set_ylabel(f"{labels[idx]} correlation fn")
            axes[-1].legend()

            plot_counter += 1

        axes[-1].set_xlabel("N")

        plt.subplots_adjust(right=1.0, top=1.0, hspace=0)
        plt.show()
