from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from elisa.base.types import FLOAT

if TYPE_CHECKING:
    # Import the concrete Matplotlib 3D axes type for typing at check-time
    from mpl_toolkits.mplot3d.axes3d import Axes3D as _MplAxes3D
    from numpy.typing import NDArray

    from elisa.types import Axes3DProtocol, Float


def set_axes_equal(ax: Axes3DProtocol | _MplAxes3D) -> None:
    """Set equal scaling for all axes of a 3D plot.

    This ensures that geometric shapes such as spheres and cubes are displayed
    without distortion. It provides a workaround for Matplotlib's
    ``ax.set_aspect("equal")`` and ``ax.axis("equal")`` which do not operate
    correctly in 3D.

    :param ax: Matplotlib 3D axis instance.
    :type ax: elisa.graphic.utils.Axes3DProtocol
    :return: ``None``.
    :rtype: None
    """
    # Cast to the local protocol so type checkers / IDEs recognise
    # the 3D-specific methods on the axis object.
    ax3d = ax

    x_limits: NDArray[np.float64] = np.asarray(ax3d.get_xlim3d())
    y_limits: NDArray[np.float64] = np.asarray(ax3d.get_ylim3d())
    z_limits: NDArray[np.float64] = np.asarray(ax3d.get_zlim3d())

    x_range: Float = FLOAT(abs(x_limits[1] - x_limits[0]))
    x_middle: Float = FLOAT(np.mean(x_limits))

    y_range: Float = FLOAT(abs(y_limits[1] - y_limits[0]))
    y_middle: Float = FLOAT(np.mean(y_limits))

    z_range: Float = FLOAT(abs(z_limits[1] - z_limits[0]))
    z_middle: Float = FLOAT(np.mean(z_limits))

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence half of the maximum range is used as the plot radius.
    plot_radius: Float = FLOAT(0.5 * max([x_range, y_range, z_range]))

    ax3d.set_xlim3d((x_middle - plot_radius, x_middle + plot_radius))
    ax3d.set_ylim3d((y_middle - plot_radius, y_middle + plot_radius))
    ax3d.set_zlim3d((z_middle - plot_radius, z_middle + plot_radius))
