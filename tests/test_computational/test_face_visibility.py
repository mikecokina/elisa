from __future__ import annotations

# The tests package selects Astropy constants before test-module imports.
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.testing as npt
import pytest

from elisa import BinarySystem, const
from elisa.binary_system import utils as bsutils
from elisa.binary_system.container import OrbitalPositionContainer
from elisa.binary_system.surface.coverage import compute_surface_coverage
from tests import set_astropy_units
from tests.utils import (
    ElisaTestCase,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


set_astropy_units()

PARAMS = {
    "system": {
        "inclination": "1.4392852253406623 rad",
        "period": 10.1,
        "argument_of_periastron": 90.0,
        "gamma": 0.0,
        "eccentricity": 0.0,
        "primary_minimum_time": 0.0,
        "phase_shift": 0.0,
        "semi_major_axis": 10.5,
        "mass_ratio": 3.333333333333334,
    },
    "primary": {
        "surface_potential": 103.3333833458323,
        "synchronicity": 1.0,
        "t_eff": 35000,
        "gravity_darkening": 1.0,
        "albedo": 1.0,
        "metallicity": 0.0,
        "discretization_factor": 5,
    },
    "secondary": {
        "surface_potential": 13.272225833478668,
        "synchronicity": 1.0,
        "t_eff": 10000,
        "gravity_darkening": 1.0,
        "albedo": 1.0,
        "metallicity": 0.0,
        "discretization_factor": 10,
    },
}


@pytest.mark.requires_ld
class VisibilityTestCase(ElisaTestCase):
    """testing whether right number of triangles and total surface area is visible to the observer."""

    def setUp(self) -> None:
        super().setUp()
        self.lc_base_path = str(
            Path(__file__).resolve().parent.joinpath("../data", "light_curves"),
        )

    def eval_coverage(self, phase: float, in_eclipse: bool = True) -> dict[str, NDArray[np.float64]]:
        bs = BinarySystem.from_json(PARAMS)

        position = bs.calculate_orbital_motion(phase)[0]
        container = OrbitalPositionContainer.from_binary_system(bs, const.Position(0, 1.0, 0.0, 0.0, 0.0))
        container.build()
        container = bsutils.move_sys_onpos(container, position)
        return compute_surface_coverage(
            container,
            bs.semi_major_axis,
            in_eclipse=in_eclipse,
            return_values=True,
            write_to_containers=False,
        )

    def test_visibility_out_of_eclipse(self) -> None:
        retval = self.eval_coverage(0.20, in_eclipse=False)

        expected_values = {"primary": 1674, "secondary": 416}
        primary_vis_faces = np.count_nonzero(retval["primary"])
        secondary_vis_faces = np.count_nonzero(retval["secondary"])
        assert (primary_vis_faces) == (expected_values["primary"])
        assert (secondary_vis_faces) == (expected_values["secondary"])

        expected_sums = {"primary": 3.3529083384789416e16, "secondary": 2.093212371790947e19}
        vis_area_p = np.sum(retval["primary"])
        vis_area_s = np.sum(retval["secondary"])
        assert (vis_area_p) == (expected_sums["primary"]), "Visible area must match exactly"
        assert (vis_area_s) == (expected_sums["secondary"]), "Visible area must match exactly"

    def test_visibility_similar_comp_eclipse(self) -> None:
        retval = self.eval_coverage(0.01)

        expected_values = {"primary": 0, "secondary": 415}
        primary_vis_faces = np.count_nonzero(retval["primary"])
        secondary_vis_faces = np.count_nonzero(retval["secondary"])
        assert (primary_vis_faces) == (expected_values["primary"])
        assert (secondary_vis_faces) == (expected_values["secondary"])

        expected_sums = {"primary": 0.0, "secondary": 2.0890992190449648e19}
        vis_area_p = np.sum(retval["primary"])
        vis_area_s = np.sum(retval["secondary"])

        npt.assert_allclose(vis_area_p, expected_sums["primary"], rtol=1e-10, atol=0.0)
        npt.assert_allclose(vis_area_s, expected_sums["secondary"], rtol=1e-10, atol=0.0)

    def test_visibility_dissimilar_comp_eclipse(self) -> None:
        retval = self.eval_coverage(0.493333333)

        expected_values = {"primary": 1674, "secondary": 416}
        primary_vis_faces = np.count_nonzero(retval["primary"])
        secondary_vis_faces = np.count_nonzero(retval["secondary"])
        assert (primary_vis_faces) == (expected_values["primary"])
        assert (secondary_vis_faces) == (expected_values["secondary"])

        expected_sums = {"primary": 3.352908353932797e16, "secondary": 2.0909670552161714e19}
        vis_area_p = np.sum(retval["primary"])
        vis_area_s = np.sum(retval["secondary"])

        npt.assert_allclose(vis_area_p, expected_sums["primary"], rtol=1e-10, atol=0.0)
        npt.assert_allclose(vis_area_s, expected_sums["secondary"], rtol=1e-10, atol=0.0)
