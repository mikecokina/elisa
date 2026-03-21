from __future__ import annotations

from typing import TYPE_CHECKING

from elisa import utils
from elisa.base.surface.coverage import surface_area_coverage
from elisa.logger import getLogger

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.single_system.container import SinglePositionContainer
    from elisa.types import Float


logger = getLogger("single_system.surface.coverage")


def compute_surface_coverage(system: SinglePositionContainer) -> dict[str, NDArray[Float]]:
    """Compute surface coverage of faces for a given rotational position.

    The function computes visible-triangle areas for faces that are visible
    in the supplied ``system`` container and expands them to a full per-face
    coverage distribution using :func:`elisa.base.surface.coverage.surface_area_coverage`.

    :param system: Single-position container describing the system and star.
    :type system: elisa.single_system.container.SinglePositionContainer
    :returns: Mapping with key ``'star'`` and value equal to per-face coverage array.
    :rtype: dict[str, numpy.typing.NDArray[elisa.types.Float]]
    """
    logger.debug("computing surface coverage for %s", system.position)
    star = system.star

    visible_face_points = star.points[star.faces[star.indices]]
    coverage_visible = utils.poly_areas(visible_face_points)
    coverage_full = surface_area_coverage(len(star.faces), star.indices, coverage_visible)

    return {"star": coverage_full}
