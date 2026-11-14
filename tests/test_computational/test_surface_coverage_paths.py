"""Exact coverage regressions on real meshes, including eclipse contacts."""
from __future__ import annotations

from itertools import product
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from elisa import BinarySystem, const
from elisa.binary_system import dynamic
from elisa.binary_system import utils as bsutils
from elisa.binary_system.container import OrbitalPositionContainer
from elisa.binary_system.surface import coverage
from tests.test_computational.test_face_visibility import PARAMS
from tests.utils import ElisaTestCase

REFERENCE = Path(__file__).resolve().parents[1] / "data" / "coverage" / "surface_coverage_paths.npz"


def coverage_positions(binary: BinarySystem) -> list[tuple[float, bool]]:
    """Exercise both orderings, both size branches, contacts and adjacent exterior phases."""
    positions = [(0.20, False), (0.70, False), (0.01, True), (0.035, True), (0.493333333, True)]
    for index, boundary in enumerate(dynamic.get_eclipse_boundaries(binary, 1.0)):
        phase = (boundary - const.HALF_PI) / const.FULL_ARC
        outside = -1e-10 if index % 2 == 0 else 1e-10
        positions.extend(((phase, True), (phase + outside, False)))
    return positions


def positioned_container(binary: BinarySystem, phase: float) -> OrbitalPositionContainer:
    position = binary.calculate_orbital_motion(phase)[0]
    container = OrbitalPositionContainer.from_binary_system(binary, const.Position(0, 1.0, 0.0, 0.0, 0.0))
    container.build()
    return bsutils.move_sys_onpos(container, position)


@pytest.mark.requires_ld
class SurfaceCoveragePathsTestCase(ElisaTestCase):
    def test_original_coverage_and_output_options(self) -> None:
        """The fixture contains unmodified feature/optimizations results at SMA=1."""
        binary = BinarySystem.from_json(PARAMS)
        with np.load(REFERENCE) as reference:
            for index, (phase, in_eclipse) in enumerate(coverage_positions(binary)):
                container = positioned_container(binary, phase)
                if phase == 0.035:
                    primary = reference[f"{index}_primary"]
                    assert np.any((primary > 0) & (primary < container.primary.areas))
                for return_values, write, sma in product((False, True), (False, True), (1.0, 3.5)):
                    with self.subTest(phase=phase, eclipse=in_eclipse, returns=return_values, write=write, sma=sma):
                        sentinels = {}
                        for component in ("primary", "secondary"):
                            star = getattr(container, component)
                            sentinels[component] = star.coverage = np.array([-1.0])
                        result = coverage.compute_surface_coverage(
                            container, sma, in_eclipse=in_eclipse,
                            return_values=return_values, write_to_containers=write,
                        )
                        if not return_values:
                            assert result is None
                        for component in ("primary", "secondary"):
                            expected = reference[f"{index}_{component}"] * sma**2
                            stored = getattr(container, component).coverage
                            if return_values:
                                np.testing.assert_array_equal(result[component], expected)
                            if write:
                                np.testing.assert_array_equal(stored, expected)
                                if return_values:
                                    assert stored is result[component]
                            else:
                                assert stored is sentinels[component]

    def test_non_eclipse_skips_projections(self) -> None:
        binary = BinarySystem.from_json(PARAMS)
        for phase in (0.20, 0.70):
            container = positioned_container(binary, phase)
            # Patch only after constructing and positioning the real mesh.
            with (
                patch.object(coverage.utils, "get_visible_projection") as visible_projection,
                patch.object(coverage.utils, "plane_projection") as plane_projection,
            ):
                result = coverage.compute_surface_coverage(container, 2.0, in_eclipse=False)
            visible_projection.assert_not_called()
            plane_projection.assert_not_called()
            for component in ("primary", "secondary"):
                star = getattr(container, component)
                expected = np.zeros_like(star.areas)
                expected[star.indices] = star.areas[star.indices]
                np.testing.assert_array_equal(result[component], expected * 4.0)
                assert np.any(result[component] == 0)
                assert np.any(result[component] > 0)
