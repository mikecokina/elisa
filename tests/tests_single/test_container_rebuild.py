"""Full single-star rebuilds must merge regenerated spot geometry each time."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from elisa import units as u
from elisa.base.star import Star
from elisa.single_system.system import SingleSystem
from tests.tests_pulse.helpers import STAR_PARAMS, SYSTEM_PARMAS
from tests.utils import ElisaTestCase


class SingleContainerRebuildTestCase(ElisaTestCase):
    @staticmethod
    def make_system(*, spotted: bool, pulsating: bool = False) -> SingleSystem:
        modes = (
            [
                {
                    "l": 2,
                    "m": 1,
                    "amplitude": 10 * u.m / u.s,
                    "frequency": 1 / u.d,
                    "horizontal_to_radial_amplitude_ratio": 1,
                },
            ]
            if pulsating
            else []
        )
        star = Star(pulsations=modes, **STAR_PARAMS)
        if spotted:
            star.spots = [
                {"longitude": 35, "latitude": 70, "angular_radius": 20, "temperature_factor": 0.9},
            ]
        return SingleSystem(star=star, **SYSTEM_PARMAS)

    def check_rebuild(self, *, spotted: bool, pulsating: bool = False) -> None:
        system = self.make_system(spotted=spotted, pulsating=pulsating)
        container = system.build_container(phase=0.17)
        equilibrium = system.build_container(phase=0.17, build_pulsations=False).star

        for iteration in range(3):
            with self.subTest(iteration=iteration):
                container.build()
                fresh = system.build_container(phase=0.17).star
                star = container.star
                assert container.is_flat()
                if spotted or pulsating:
                    assert star.is_flat()
                assert len(star.points) == len(fresh.points)
                assert len(star.faces) == len(fresh.faces)
                # Identical inputs and deterministic builders: only roundoff is allowed.
                assert_allclose(star.points, fresh.points, rtol=1e-14, atol=0)
                assert_array_equal(star.faces, fresh.faces)
                assert_allclose(star.areas, fresh.areas, rtol=1e-14, atol=0)
                assert_allclose(star.areas.sum(), fresh.areas.sum(), rtol=1e-14, atol=0)
                # Independently reconstruct physical triangle areas from the final mesh.
                triangles = star.points[star.faces]
                areas = 0.5 * np.linalg.norm(
                    np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
                    axis=1,
                )
                assert_allclose(star.areas, areas, rtol=1e-14, atol=0)
                if spotted:
                    spot = equilibrium.spots[0]
                    assert len(spot.points) > 0
                    assert len(spot.faces) > 0
                    offset = len(equilibrium.points) - len(spot.points)
                    assert_array_equal(star.faces[-len(spot.faces) :], spot.faces + offset)
                    reference = star.pulsation_equilibrium_points if pulsating else star.points
                    assert_allclose(reference[-len(spot.points) :], spot.points, rtol=1e-14, atol=0)
                if pulsating:
                    assert_allclose(star.pulsation_equilibrium_points, equilibrium.points, rtol=1e-14, atol=0)
                    directions = equilibrium.points / np.linalg.norm(equilibrium.points, axis=1)[:, None]
                    assert_allclose(
                        star.pulsation_directions,
                        directions @ star.pulsation_rotation,
                        rtol=1e-14,
                        atol=1e-14,
                    )
                    displacement = sum(mode.complex_displacement.real for mode in star.pulsations.values())
                    assert_allclose(star.points, equilibrium.points + displacement, rtol=1e-14, atol=0)

    def test_spotted_rebuild_matches_fresh(self) -> None:
        self.check_rebuild(spotted=True)

    def test_unspotted_rebuild_matches_fresh(self) -> None:
        self.check_rebuild(spotted=False)

    def test_pulsating_spotted_rebuild_matches_fresh(self) -> None:
        self.check_rebuild(spotted=True, pulsating=True)
