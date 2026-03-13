from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.types import Float


def _selector_to_mask(sel: NDArray[np.bool], size: int, name: str) -> tuple[np.ndarray, int]:
    """Convert a selector (mask or indices) into a boolean mask of length ``size``.

    :param sel: Boolean mask or index array selecting elements.
    :type sel: numpy.typing.NDArray
    :param size: Expected length of resulting mask.
    :type size: int
    :param name: Name of the selector (used in error messages).
    :type name: str
    :returns: Tuple of (mask, selected_count).
    :rtype: tuple[numpy.ndarray, int]
    :raises ValueError: When indices are out of range or shapes are invalid.
    """
    arr = np.asarray(sel)

    # boolean mask case
    if arr.dtype == bool:
        if arr.shape != (size,):
            msg = f"Boolean `{name}` mask must have length equal to `size`"
            raise ValueError(msg)
        mask = arr
    else:
        # treat as list of indices
        if arr.ndim != 1:
            msg = f"`{name}` indices must be one-dimensional"
            raise ValueError(msg)
        try:
            idx = arr.astype(int)
        except (TypeError, ValueError) as err:
            msg = f"`{name}` indices could not be interpreted as integers"
            raise ValueError(msg) from err
        if idx.size == 0:
            mask = np.zeros(int(size), dtype=bool)
        else:
            if np.any(idx < 0) or np.any(idx >= size):
                msg = f"Indices in `{name}` are out of bounds for size {size}"
                raise ValueError(msg)
            mask = np.zeros(int(size), dtype=bool)
            mask[idx] = True

    sel_count = int(mask.sum())
    return mask, sel_count


def _assign_coverage(
        coverage: NDArray[np.floating],
        mask: np.ndarray,
        cov: NDArray | Float,
        expected_count: int, name: str,
) -> None:
    """Validate and assign coverage values to positions indicated by mask.

    :param coverage: The full coverage array to mutate.
    :param mask: Boolean mask indicating positions to assign.
    :param cov: Coverage scalar or array for selected positions.
    :param expected_count: Number of selected positions.
    :param name: Name used in error messages ('visible' or 'partial').
    :raises ValueError: If cov has inconsistent shape.
    """
    arr = np.asarray(cov)
    if arr.shape == ():
        coverage[mask] = float(arr)
    else:
        if arr.ndim != 1 or arr.shape[0] != expected_count:
            msg = f"Length of `{name}_coverage` does not match number of selected {name} elements"
            raise ValueError(msg)
        coverage[mask] = arr


def surface_area_coverage(
        size: int,
        visible: NDArray,
        visible_coverage: NDArray | Float,
        partial: NDArray | None = None,
        partial_coverage: NDArray | Float | None = None,
) -> NDArray[np.floating]:
    """Prepare an array with coverage values for surface areas.

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
    :raises TypeError: If ``size`` is not an int.
    :raises ValueError: If provided masks/indices or coverage arrays have
        inconsistent sizes or invalid values.
    """
    # Validate size
    if not isinstance(size, int):
        msg = "`size` must be an int"
        raise TypeError(msg)
    if size < 0:
        msg = "`size` must be non-negative"
        raise ValueError(msg)

    coverage: NDArray[np.floating] = np.zeros(int(size), dtype=float)

    # process visible selector
    vis_mask, vis_count = _selector_to_mask(visible, int(size), "visible")
    if vis_count > 0:
        _assign_coverage(coverage, vis_mask, visible_coverage, vis_count, "visible")

    # process partial selector if provided
    if partial is not None:
        part_mask, part_count = _selector_to_mask(partial, int(size), "partial")
        if part_count > 0:
            if partial_coverage is None:
                msg = "`partial_coverage` must be provided when `partial` is given"
                raise ValueError(msg)
            _assign_coverage(coverage, part_mask, partial_coverage, part_count, "partial")

    return coverage
