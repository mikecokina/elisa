from __future__ import annotations

from typing import TYPE_CHECKING, Any

from elisa.graphic import data_set_graphics

if TYPE_CHECKING:
    from elisa.analytics.dataset.base import DataSet


class Plot:
    """Plot visualization interface for DataSet instances.

    This class provides methods for displaying and visualizing observational data
    from a DataSet instance.

    :param instance: The DataSet instance to be plotted.
    :type instance: DataSet
    """

    def __init__(self, instance: DataSet) -> None:
        """Initialize Plot instance with a DataSet.

        :param instance: The DataSet instance to associate with this Plot.
        :type instance: DataSet
        :returns: None.
        :rtype: None
        """
        self.data_set: DataSet = instance

    def display_observation(self, **kwargs: Any) -> None:
        """Plot the DataSet observation for visual examination.

        Displays observational data using matplotlib. The visualization automatically
        adapts based on whether the DataSet includes error information:
        uses scatter plot for data without errors, errorbar plot for data with errors.

        :param kwargs: Keyword arguments passed to matplotlib.pyplot.scatter
            (DataSet without errors) or matplotlib.pyplot.errorbar (DataSet with errors).
        :type kwargs: Any
        :returns: None.
        :rtype: None
        """
        obs_kwargs: dict[str, Any] = {
            "x_data": self.data_set.x_data,
            "y_data": self.data_set.y_data,
            "y_err": self.data_set.y_err,
            "x_unit": self.data_set.x_unit,
            "y_unit": self.data_set.y_unit,
            "plot_kwargs": kwargs,
        }
        data_set_graphics.display_observations(**obs_kwargs)
