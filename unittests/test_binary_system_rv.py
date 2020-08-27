import numpy as np
import os.path as op

from importlib import reload

from numpy.testing import assert_array_equal, assert_array_less
from unittests.utils import (
    ElisaTestCase,
    prepare_binary_system,
    BINARY_SYSTEM_PARAMS,
    SPOTS_META,
    normalize_lv_for_unittests,
    load_radial_curve
)
from elisa.observer.observer import Observer
from elisa.binary_system.curves import (
    rv,
    rvmp
)
from elisa.conf import config
from elisa import umpy as up, const
from astropy import units as u


TOL = 1e-3


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


class ComputeRadiometricRVTestCase(ElisaTestCase):
    params = {
        'detached': {
            "primary_mass": 2.0, "secondary_mass": 1.0,
            "primary_surface_potential": 5.0, "secondary_surface_potential": 5.0,
            "primary_synchronicity": 1.0, "secondary_synchronicity": 1.0,
            "argument_of_periastron": const.HALF_PI * u.rad, "gamma": 0.0, "period": 5.0,
            "eccentricity": 0.0, "inclination": const.HALF_PI * u.rad, "primary_minimum_time": 0.0,
            "phase_shift": 0.0,
            "primary_t_eff": 6500, "secondary_t_eff": 6500,
            "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
            "primary_albedo": 1.0, "secondary_albedo": 1.0,
        },
        'detached-async-ecc': {
            "primary_mass": 2.0, "secondary_mass": 1.0,
            "primary_surface_potential": 5.0, "secondary_surface_potential": 5.0,
            "primary_synchronicity": 0.8, "secondary_synchronicity": 1.2,
            "argument_of_periastron": const.HALF_PI * u.rad, "gamma": 0.0, "period": 5.0,
            "eccentricity": 0.3, "inclination": const.HALF_PI * u.rad, "primary_minimum_time": 0.0,
            "phase_shift": 0.0,
            "primary_t_eff": 6500, "secondary_t_eff": 6500,
            "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
            "primary_albedo": 1.0, "secondary_albedo": 1.0,
        },
        'detached-async': {
            "primary_mass": 2.0, "secondary_mass": 1.0,
            "primary_surface_potential": 5.0, "secondary_surface_potential": 5.0,
            "primary_synchronicity": 0.8, "secondary_synchronicity": 1.2,
            "argument_of_periastron": const.HALF_PI * u.rad, "gamma": 0.0, "period": 5.0,
            "eccentricity": 0.0, "inclination": const.HALF_PI * u.rad, "primary_minimum_time": 0.0,
            "phase_shift": 0.0,
            "primary_t_eff": 6500, "secondary_t_eff": 6500,
            "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
            "primary_albedo": 1.0, "secondary_albedo": 1.0,
        },
        'over-contact': {
            "primary_mass": 2.0, "secondary_mass": 1.0,
            "primary_surface_potential": 2.7,
            "secondary_surface_potential": 2.7,
            "primary_synchronicity": 1.0, "secondary_synchronicity": 1.0,
            "argument_of_periastron": 90 * u.deg, "gamma": 0.0, "period": 1.0,
            "eccentricity": 0.0, "inclination": 90.0 * u.deg, "primary_minimum_time": 0.0,
            "phase_shift": 0.0,
            "primary_t_eff": 6000, "secondary_t_eff": 6000,
            "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
            "primary_albedo": 1.0, "secondary_albedo": 1.0
        },
        'eccentric': {
            "primary_mass": 1.0, "secondary_mass": 1.0,
            "primary_surface_potential": 8,
            "secondary_surface_potential": 8,
            "primary_synchronicity": 1.0, "secondary_synchronicity": 1.0,
            "argument_of_periastron": 223 * u.deg, "gamma": 0.0, "period": 3.0,
            "eccentricity": 0.3, "inclination": 90.0 * u.deg, "primary_minimum_time": 0.0,
            "phase_shift": 0.0,
            "primary_t_eff": 6000, "secondary_t_eff": 6000,
            "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
            "primary_albedo": 1.0, "secondary_albedo": 1.0
        }
    }

    def setUp(self):
        # raise unittest.SkipTest(message)
        self.lc_base_path = op.join(op.dirname(op.abspath(__file__)), "data", "light_curves")
        self.base_path = op.join(op.dirname(op.abspath(__file__)), "data", "radial_curves")
        self.law = config.LIMB_DARKENING_LAW

        config.LD_TABLES = op.join(self.lc_base_path, "limbdarkening")
        config.CK04_ATM_TABLES = op.join(self.lc_base_path, "atmosphere")
        config.ATM_ATLAS = "ck04"
        config._update_atlas_to_base_dir()
        config.RV_LAMBDA_INTERVAL = (5500, 5600)

    def tearDown(self):
        config.LIMB_DARKENING_LAW = self.law
        reload(rv)

    def do_comparison(self, bs, rv_file):
        o = Observer(system=bs)

        start_phs, stop_phs, step = -0.2, 1.2, 0.1

        obtained = o.rv(from_phase=start_phs, to_phase=stop_phs, phase_step=step, method='radiometric')
        obt_phs = obtained[0]
        obt_p = np.array(obtained[1]['primary'])
        obt_s = np.array(obtained[1]['secondary'])
        obt_p = obt_p[~np.isnan(obt_p)]
        obt_s = obt_s[~np.isnan(obt_s)]

        expected = load_radial_curve(rv_file)
        exp_phs = expected['phases']
        exp_p = np.array(expected['primary'])
        exp_s = np.array(expected['secondary'])
        exp_p = exp_p[~np.isnan(exp_p)]
        exp_s = exp_s[~np.isnan(exp_s)]

        self.assertTrue(np.all(up.abs(obt_phs - np.round(exp_phs, 3)) < TOL))
        self.assertTrue(np.all((up.abs(obt_p - exp_p) / np.abs(np.max(obt_p))) < TOL))
        self.assertTrue(np.all((up.abs(obt_s - exp_s) / np.abs(np.max(obt_s))) < TOL))

    def test_light_curve_pass_on_all_ld_law(self):
        """
        no assert here, it just has to pass without error
        """
        bs = prepare_binary_system(self.params["detached"])
        start_phs, stop_phs, step = -0.2, 1.2, 0.1

        laws = config.LD_LAW_TO_FILE_PREFIX.keys()

        for law in laws:
            config.LIMB_DARKENING_LAW = law
            reload(rvmp)
            o = Observer(system=bs)
            o.rv(from_phase=start_phs, to_phase=stop_phs, phase_step=step, method='radiometric')

    def test_circular_synchronous_detached_system(self):
        config.LIMB_DARKENING_LAW = "linear"
        reload(rv)

        bs = prepare_binary_system(self.params["detached"])
        self.do_comparison(bs, "detached.circ.sync.json")

    def test_circular_synchronous_overcontact_system(self):
        config.LIMB_DARKENING_LAW = "linear"
        reload(rv)

        bs = prepare_binary_system(self.params["over-contact"])
        self.do_comparison(bs, "overcontact.circ.sync.json")

    def test_circular_spotty_synchronous_detached_system(self):
        config.LIMB_DARKENING_LAW = "linear"
        reload(rv)

        bs = prepare_binary_system(self.params["detached"],
                                   spots_primary=SPOTS_META["primary"],
                                   spots_secondary=SPOTS_META["secondary"])

        self.do_comparison(bs, "detached.circ.spotty.sync.json")

    def test_circular_spotty_synchronous_overcontact_system(self):
        config.LIMB_DARKENING_LAW = "linear"
        reload(rv)

        bs = prepare_binary_system(self.params["over-contact"],
                                   spots_primary=SPOTS_META["primary"],
                                   spots_secondary=SPOTS_META["secondary"])

        self.do_comparison(bs, "overcontact.circ.spotty.sync.json")

    def test_cicular_spotty_asynchronous_detached_system(self):
        config.MAX_SPOT_D_LONGITUDE = up.pi / 45.0
        reload(rv)

        bs = prepare_binary_system(self.params["detached-async"],
                                   spots_primary=SPOTS_META["primary"])

        self.do_comparison(bs, "detached.circ.spotty.async.json")
