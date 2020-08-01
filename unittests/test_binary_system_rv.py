import numpy as np

from numpy.testing import assert_array_equal, assert_array_less
from unittests.utils import (
    ElisaTestCase,
    prepare_binary_system,
    BINARY_SYSTEM_PARAMS,
    SPOTS_META,
    normalize_lv_for_unittests,
    load_radial_curve
)
from elisa.binary_system.curves import (
    rv
)
from elisa import umpy as up
from astropy import units as u


class BinaryRadialCurvesTestCase(ElisaTestCase):

    def setUp(self):
        self.phases = up.arange(-0.2, 1.25, 0.05)

    def test_circular_detached(self):
        s = prepare_binary_system(BINARY_SYSTEM_PARAMS["detached-physical"])
        rvdict = rv.com_radial_velocity(s, position_method=s.calculate_orbital_motion, phases=self.phases)
        obtained_rvp, obtained_rvs = normalize_lv_for_unittests(rvdict['primary'], rvdict['secondary'])
        expected = load_radial_curve("detahed.circ.json")

        expected_rvp, expected_rvs = -1 * np.array(expected["primary"]), -1 * np.array(expected["secondary"])
        expected_rvp, expected_rvs = normalize_lv_for_unittests(expected_rvp, expected_rvs)

        assert_array_equal(np.round(expected_rvp, 4), np.round(obtained_rvp, 4))
        assert_array_equal(np.round(expected_rvs, 4), np.round(obtained_rvs, 4))

    def test_eccentric_detached(self):
        s = prepare_binary_system(BINARY_SYSTEM_PARAMS["detached.ecc"])
        rvdict = rv.com_radial_velocity(s, position_method=s.calculate_orbital_motion, phases=self.phases)
        obtained_rvp, obtained_rvs = normalize_lv_for_unittests(rvdict['primary'], rvdict['secondary'])
        expected = load_radial_curve("detahed.ecc.json")

        expected_rvp, expected_rvs = -1 * np.array(expected["primary"]), -1 * np.array(expected["secondary"])
        expected_rvp, expected_rvs = normalize_lv_for_unittests(expected_rvp, expected_rvs)

        assert_array_equal(np.round(expected_rvp, 4), np.round(obtained_rvp, 4))
        assert_array_equal(np.round(expected_rvs, 4), np.round(obtained_rvs, 4))

    def test_rv_consistency_circular_detached(self):
        binary_kwargs = BINARY_SYSTEM_PARAMS["detached-physical"]
        binary_kwargs['inclination'] = 70 * u.deg

        s = prepare_binary_system(binary_kwargs, spots_primary=SPOTS_META['primary'])
        rvdict1 = s.compute_rv(position_method=s.calculate_orbital_motion, phases=self.phases, method='point_mass')
        rvdict2 = s.compute_rv(position_method=s.calculate_orbital_motion, phases=self.phases, method='radiometric')

        rvdict1['primary'], rvdict1['secondary'] = normalize_lv_for_unittests(rvdict1['primary'], rvdict1['secondary'])
        rvdict2['primary'], rvdict2['secondary'] = normalize_lv_for_unittests(rvdict2['primary'], rvdict2['secondary'])

        desired_delta = 0.02
        assert_array_less(np.abs(rvdict1['primary'] - rvdict2['primary']), desired_delta * np.ones(self.phases.shape))
        assert_array_less(np.abs(rvdict2['secondary'] - rvdict2['secondary']),
                          desired_delta * np.ones(self.phases.shape))

    def test_rv_consistency_circular_contact(self):
        binary_kwargs = BINARY_SYSTEM_PARAMS["over-contact"]
        binary_kwargs['inclination'] = 10 * u.deg

        s = prepare_binary_system(binary_kwargs, spots_primary=SPOTS_META['primary'])
        rvdict1 = s.compute_rv(position_method=s.calculate_orbital_motion, phases=self.phases, method='point_mass')
        rvdict2 = s.compute_rv(position_method=s.calculate_orbital_motion, phases=self.phases, method='radiometric')

        rvdict1['primary'], rvdict1['secondary'] = normalize_lv_for_unittests(rvdict1['primary'],
                                                                              rvdict1['secondary'])
        rvdict2['primary'], rvdict2['secondary'] = normalize_lv_for_unittests(rvdict2['primary'],
                                                                              rvdict2['secondary'])

        desired_delta = 0.033
        assert_array_less(np.abs(rvdict1['primary'] - rvdict2['primary']),
                          desired_delta * np.ones(self.phases.shape))
        assert_array_less(np.abs(rvdict2['secondary'] - rvdict2['secondary']),
                          desired_delta * np.ones(self.phases.shape))
