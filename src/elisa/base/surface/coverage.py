from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from elisa.base.error import raise_elisa_unexpected_error

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.types import NP_BOOL_, Float


def _selector_to_mask(sel: NDArray[NP_BOOL_], size: int, name: str) -> tuple[np.ndarray, int]:
    """Convert a selector (mask or indices) into a boolean mask of length ``size``.

    :param sel: Boolean mask or index array selecting elements.
    :type sel: numpy.typing.NDArray
    :param size: Expected length of resulting mask.
    :type size: int
    :param name: Name of the selector (used in error messages).
    :type name: str
    :returns: Tuple of (mask, selected_count).
    :rtype: tuple[numpy.ndarray, int]
    :raises ElisaUnexpectedError: If a selector violates its contract with the current mesh.
    """
    arr = np.asarray(sel)

    # Visibility code builds these selectors for the current mesh. Candidate
    # geometry may change which faces are visible, but cannot legitimately
    # change selector types or break their alignment with that mesh.
    # boolean mask case
    if arr.dtype == bool:
        if arr.shape != (size,):
            msg = f"Boolean {name!r} face selector has shape {arr.shape}; expected ({size},) for the current mesh."
            raise_elisa_unexpected_error(msg)
        mask = arr
    else:
        # treat as list of indices
        if arr.ndim != 1:
            msg = f"The {name!r} face indices must be one-dimensional; received shape {arr.shape}."
            raise_elisa_unexpected_error(msg)
        try:
            idx = arr.astype(int)
        except (TypeError, ValueError) as err:
            msg = f"`{name}` indices could not be interpreted as integers"
            raise_elisa_unexpected_error(msg, err)
        if idx.size == 0:
            mask = np.zeros(int(size), dtype=bool)
        else:
            if np.any(idx < 0) or np.any(idx >= size):
                msg = f"The {name!r} selector references face indices outside the current mesh of {size} faces."
                raise_elisa_unexpected_error(msg)
            mask = np.zeros(int(size), dtype=bool)
            mask[idx] = True

    sel_count = int(mask.sum())
    return mask, sel_count


def _validated_selected_face_areas(
        areas: NDArray | Float,
        selected_face_count: int,
        selection_name: str,
) -> NDArray | float:
    """Accept a broadcast scalar or one already computed area per selected face.

    :raises ElisaUnexpectedError: If computed areas do not align with selected faces.
    """
    arr = np.asarray(areas)
    if arr.shape == ():
        return float(arr)
    if arr.ndim != 1 or arr.shape[0] != selected_face_count:
        # Drift may change the mesh and area values, but the caller must keep
        # the selected faces and their computed areas aligned for that mesh.
        msg = (
            f"Computed {selection_name!r} face areas have shape {arr.shape}; "
            f"expected one value per selected face ({selected_face_count},)."
        )
        raise_elisa_unexpected_error(msg)
    return arr


def _assign_selected_face_areas(
        coverage: NDArray[np.floating],
        mask: np.ndarray,
        areas: NDArray | Float,
        selected_face_count: int,
        selection_name: str,
) -> None:
    """Scatter prepared areas into selected positions of the full coverage array."""
    coverage[mask] = _validated_selected_face_areas(areas, selected_face_count, selection_name)


def assemble_surface_coverage(
        size: int,
        visible: NDArray,
        visible_coverage: NDArray | Float,
        partial: NDArray | None = None,
        partial_coverage: NDArray | Float | None = None,
) -> NDArray[np.floating]:
    """Assemble already computed face areas into a full mesh coverage array.

    This function does not calculate intersections or geometric areas. Callers
    compute those first; this assembly only expands their selected-face values.

    Convert boolean masks or index arrays describing "visible" and
    optionally "partial" surface elements into a full coverage array of
    length ``size``. Invisible positions (not listed in ``visible`` or
    ``partial``) are filled with zeros.

    The function accepts either boolean masks or index arrays for
    ``visible`` and ``partial``. Coverage arrays may be scalars (applied
    to all selected positions) or arrays matching the number of selected
    positions.

    :param size: Length of the returned coverage array.
    :type size: int
    :param visible: Boolean mask or index array selecting fully visible
        surface elements.
    :type visible: numpy.typing.NDArray
    :param visible_coverage: Coverage value(s) for positions selected by
        ``visible``. It may be a scalar or array-like matching the number of
        selected elements.
    :type visible_coverage: numpy.typing.NDArray | elisa.types.Float
    :param partial: Optional boolean mask or index array selecting
        partially visible elements.
    :type partial: numpy.typing.NDArray | None
    :param partial_coverage: Coverage value(s) for positions selected by
        ``partial``. It may be a scalar or array-like matching the number of
        selected elements.
    :type partial_coverage: numpy.typing.NDArray | elisa.types.Float | None
    :returns: Coverage array with dtype float and length ``size``.
    :rtype: numpy.ndarray
    :raises ElisaUnexpectedError: If size, face selectors or prepared area shapes violate the mesh contract,
        or selected partial faces lack prepared coverage.
    """
    # Callers derive size from a mesh shape or len(); a negative or noninteger
    # face count violates the internal mesh contract, regardless of candidate values.
    if not isinstance(size, int):
        msg = f"Surface coverage requires an integer face count; received {size!r} ({type(size).__name__})."
        raise_elisa_unexpected_error(msg)
    if size < 0:
        msg = f"Surface coverage received a negative face count ({size}); mesh dimensions cannot be negative."
        raise_elisa_unexpected_error(msg)

    coverage: NDArray[np.floating] = np.zeros(int(size), dtype=float)

    # process visible selector
    vis_mask, vis_count = _selector_to_mask(visible, int(size), "visible")
    if vis_count > 0:
        _assign_selected_face_areas(coverage, vis_mask, visible_coverage, vis_count, "visible")

    # process partial selector if provided
    if partial is not None:
        part_mask, part_count = _selector_to_mask(partial, int(size), "partial")
        if part_count > 0:
            if partial_coverage is None:
                # The synthesis caller must compute partial-face coverage before
                # assembly. Degenerate geometry must fail numerically there;
                # omitting that result here violates the internal flow contract.
                msg = ("Selected partial faces require coverage computed "
                       "before surface coverage assembly; received None.")
                raise_elisa_unexpected_error(msg)
            _assign_selected_face_areas(coverage, part_mask, partial_coverage, part_count, "partial")

    return coverage


def surface_area_coverage(
        size: int,
        visible: NDArray,
        visible_coverage: NDArray | Float,
        partial: NDArray | None = None,
        partial_coverage: NDArray | Float | None = None,
) -> NDArray[np.floating]:
    """Compatibility wrapper for :func:`assemble_surface_coverage`."""
    return assemble_surface_coverage(size, visible, visible_coverage, partial, partial_coverage)
