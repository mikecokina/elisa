from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from elisa import utils

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.base.container import StarContainer
    from elisa.base.spot import Spot
    from elisa.types import Float


def correct_component_mesh(star: StarContainer, com: Float, correction_factors: NDArray) -> StarContainer:
    """Correct underestimation of surface due to discretization.

    Apply discretization correction factors to the positions of the
    component (star) and its spots. Points are shifted to the centre of
    mass, scaled by the correction factor, and shifted back.

    :param star: StarContainer whose mesh will be corrected.
    :type star: elisa.base.container.StarContainer
    :param com: X-coordinate of the component centre of mass.
    :type com: elisa.types.Float
    :param correction_factors: Array-like of correction factors used by
        :func:`elisa.utils.discretization_correction_factor`.
    :type correction_factors: numpy.typing.NDArray | numpy.typing.NDArray
    :returns: The same ``star`` instance with updated ``points``.
    :rtype: elisa.base.container.StarContainer
    :raises TypeError: If required attributes are missing on ``star``.
    """
    # Basic validation of star interface
    if not hasattr(star, "points") or not hasattr(star, "discretization_factor"):
        msg = "`star` does not provide required mesh attributes"
        raise TypeError(msg)

    com_vector = np.array([com, 0.0, 0.0])

    disc = star.discretization_factor
    args = (disc, correction_factors)
    factor = utils.discretization_correction_factor(*args)
    centered_points = factor * (star.points - com_vector[None, :])
    star.points = centered_points + com_vector[None, :]

    if star.has_spots():
        for spot_ in star.spots.values():
            spot: Spot = spot_  # bypass static type checks for dynamic spot attributes
            if not hasattr(spot, "points") or not hasattr(spot, "discretization_factor"):
                # skip malformed spot definitions but continue processing others
                continue
            spot_disc = spot.discretization_factor
            args = (spot_disc, correction_factors)
            factor = utils.discretization_correction_factor(*args)
            centered_points = factor * (spot.points - com_vector[None, :])
            spot.points = centered_points + com_vector[None, :]

    return star


def symmetry_point_reduction(array: NDArray, base_symmetry_points_number: int) -> NDArray:
    """Return the part of an array corresponding to the base symmetry block.

    This utility extracts the first ``base_symmetry_points_number`` entries
    from ``array``. It is used when surface symmetries are exploited to
    reduce computational work to the fundamental block.

    :param array: Array-like distribution defined on surface points.
    :type array: numpy.typing.NDArray | numpy.typing.NDArray
    :param base_symmetry_points_number: Number of points contained in the
        base symmetry block (must be non-negative).
    :type base_symmetry_points_number: int
    :returns: Reduced array limited to the base symmetry points.
    :rtype: numpy.typing.NDArray
    :raises TypeError: If ``base_symmetry_points_number`` is not an int.
    :raises ValueError: If ``base_symmetry_points_number`` is negative.
    """
    if not isinstance(base_symmetry_points_number, int):
        msg = "`base_symmetry_points_number` must be an int"
        raise TypeError(msg)
    if base_symmetry_points_number < 0:
        msg = "`base_symmetry_points_number` must be non-negative"
        raise ValueError(msg)

    arr = np.asarray(array)
    return arr[:base_symmetry_points_number]
