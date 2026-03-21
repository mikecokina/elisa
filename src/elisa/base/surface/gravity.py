from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from elisa.base.container import StarContainer


def eval_args_for_magnitude_gradient(star_container: StarContainer) -> tuple[NDArray[np.floating], NDArray[np.integer]]:
    """Return points and faces to evaluate magnitude gradient on the surface.

    Depending on whether the surface has symmetry, return either the
    symmetric subset of points and faces or the full surface (excluding
    spots). The function does not modify the provided ``star_container``.

    :param star_container: Container describing the stellar surface.
    :type star_container: elisa.base.container.StarContainer
    :returns: Tuple of ``(points, faces)`` where ``points`` is an (N,3)
        array of coordinates and ``faces`` is an (M,3) array of triangle
        indices suitable for further geometric computations.
    :rtype: tuple[numpy.typing.NDArray, numpy.typing.NDArray]
    :raises TypeError: If ``star_container`` does not provide expected API.
    """
    # basic runtime validation
    if not hasattr(star_container, "symmetry_test") or not hasattr(star_container, "points"):
        msg = "`star_container` does not appear to be a valid StarContainer"
        raise TypeError(msg)

    if star_container.symmetry_test():
        points = star_container.symmetry_points()
        faces = star_container.symmetry_faces(star_container.faces)
    else:
        points, faces = star_container.points, star_container.faces

    return points, faces
