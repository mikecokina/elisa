from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib import pyplot as plt

from elisa import units as u

if TYPE_CHECKING:
    from astropy.units import UnitBase
    from numpy.typing import NDArray


def display_observations(
    *,
    x_data: NDArray,
    x_unit: UnitBase,
    y_data: NDArray,
    y_err: NDArray | None,
    y_unit: UnitBase,
    plot_kwargs: dict[str, object],
) -> None:
    """Display observational data stored in a data set.

    This helper plots the provided observations either with vertical error bars
    or as a scatter plot, depending on whether ``y_err`` is available.

    :param x_data: Values on the x-axis.
    :type x_data: NDArray
    :param x_unit: Unit of the x-axis values.
    :type x_unit: astropy.units.UnitBase
    :param y_data: Values on the y-axis.
    :type y_data: NDArray
    :param y_err: Uncertainties of the y-axis values. If ``None``, a scatter
        plot is used instead of an error-bar plot.
    :type y_err: NDArray | None
    :param y_unit: Unit of the y-axis values.
    :type y_unit: astropy.units.UnitBase
    :param plot_kwargs: Keyword arguments forwarded to
        :func:`matplotlib.pyplot.errorbar` or
        :func:`matplotlib.pyplot.scatter`.
    :type plot_kwargs: dict[str, object]
    :return: ``None``.
    :rtype: None
    """
    if y_err is not None:
        plt.errorbar(
            x=x_data,
            y=y_data,
            yerr=y_err,
            linestyle="none",
            **plot_kwargs,
        )
    else:
        plt.scatter(
            x=x_data,
            y=y_data,
            **plot_kwargs,
        )

    x_label = (
        "Phase"
        if x_unit == u.dimensionless_unscaled
        else f"Time [{x_unit}]"
    )
    y_label = (
        "Flux"
        if y_unit == u.dimensionless_unscaled
        else f"Magnitude [{y_unit}]"
    )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.subplots_adjust(top=0.98, right=0.98)
    plt.show()
