"""Coverage assembly preserves selection, broadcasting and overwrite semantics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from elisa.base.error import ElisaUnexpectedError
from elisa.base.surface.coverage import assemble_surface_coverage, surface_area_coverage

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


@pytest.mark.parametrize("assemble", [assemble_surface_coverage, surface_area_coverage])
@pytest.mark.parametrize(
    ("visible", "areas", "partial", "partial_areas", "expected"),
    [
        ([0, 2], [10.0, 20.0], [1], [3.0], [10.0, 3.0, 20.0, 0.0]),
        ([True, False, True, False], 2.0, [False, True, False, False], 3.0, [2.0, 3.0, 2.0, 0.0]),
        ([0, 1], [10.0, 20.0], [1], 3.0, [10.0, 3.0, 0.0, 0.0]),
        ([], [], [], None, [0.0, 0.0, 0.0, 0.0]),
        ([2, 0, 2], [10.0, 20.0], None, None, [10.0, 0.0, 20.0, 0.0]),
    ],
)
def test_assembly_and_compatibility_wrapper(
    assemble: Callable[..., NDArray[np.float64] | None],
    visible: list[bool] | list[int] | list[object],
    areas: float | list[float] | list[object],
    partial: list[bool] | list[int] | list[object] | None,
    partial_areas: float | list[float] | None,
    expected: list[float],
) -> None:
    result = assemble(4, visible, areas, partial, partial_areas)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.dtype(float)


@pytest.mark.parametrize("assemble", [assemble_surface_coverage, surface_area_coverage])
def test_area_count_validation_is_preserved(assemble: Callable[..., NDArray[np.float64] | None]) -> None:
    with pytest.raises(ElisaUnexpectedError, match="one value per selected face"):
        assemble(4, [0, 2], [1.0])


@pytest.mark.parametrize(
    "selector",
    [
        np.array([True, False]),
        np.array([[0, 1]]),
        np.array(["bad-index"]),
        np.array([-1]),
        np.array([4]),
    ],
)
def test_invalid_face_selectors_are_fatal_contracts(selector: NDArray[np.generic]) -> None:
    with pytest.raises(ElisaUnexpectedError):
        assemble_surface_coverage(4, selector, 1.0)
