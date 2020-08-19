import numpy as np

from numpy.testing import assert_array_equal
from unittests.utils import (
    ElisaTestCase,
    prepare_binary_system,
    BINARY_SYSTEM_PARAMS,
    normalize_lv_for_unittests,
    load_radial_curve
)
from elisa.binary_system.curves import (
    rv
)
from elisa import umpy as up


class BinaryRadialCurvesTestCase(ElisaTestCase):

    def setUp(self):
        self.phases = up.arange(-0.2, 1.25, 0.05)

    def test_circular_detached(self):
        s = prepare_binary_system(BINARY_SYSTEM_PARAMS["detached-physical"])
        phases, rvdict = rv.radial_velocity(s, position_method=s.calculate_orbital_motion, phases=self.phases)
        obtained_rvp, obtained_rvs = normalize_lv_for_unittests(rvdict['primary'], rvdict['secondary'])
        expected = load_radial_curve("detahed.circ.json")

        expected_rvp, expected_rvs = -1 * np.array(expected["primary"]), -1 * np.array(expected["secondary"])
        expected_rvp, expected_rvs = normalize_lv_for_unittests(expected_rvp, expected_rvs)

        assert_array_equal(np.round(expected_rvp, 4), np.round(obtained_rvp, 4))
        assert_array_equal(np.round(expected_rvs, 4), np.round(obtained_rvs, 4))

    def test_eccentric_detached(self):
        s = prepare_binary_system(BINARY_SYSTEM_PARAMS["detached.ecc"])
        phases, rvdict = rv.radial_velocity(s, position_method=s.calculate_orbital_motion, phases=self.phases)
        obtained_rvp, obtained_rvs = normalize_lv_for_unittests(rvdict['primary'], rvdict['secondary'])
        expected = load_radial_curve("detahed.ecc.json")

        expected_rvp, expected_rvs = -1 * np.array(expected["primary"]), -1 * np.array(expected["secondary"])
        expected_rvp, expected_rvs = normalize_lv_for_unittests(expected_rvp, expected_rvs)

        assert_array_equal(np.round(expected_rvp, 4), np.round(obtained_rvp, 4))
        assert_array_equal(np.round(expected_rvs, 4), np.round(obtained_rvs, 4))
