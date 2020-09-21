import numpy as np
import os.path as op
import os
import tempfile

from copy import copy
from os import cpu_count
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
from elisa.observer import observer
from elisa.observer import mp_manager
from elisa.binary_system.curves import (
    rv,
    rv_point,
    c_appx_router
)
from elisa.conf import config
from elisa import umpy as up, const
from elisa import units as u


TOL = 1e-3

PARAMS = {
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
        "eccentricity": 0.3, "inclination": 85.0 * u.deg, "primary_minimum_time": 0.0,
        "phase_shift": 0.0,
        "primary_t_eff": 6000, "secondary_t_eff": 6000,
        "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
        "primary_albedo": 1.0, "secondary_albedo": 1.0
    }
}


def reload_modules():
    reload(mp_manager)
    reload(observer)
    reload(rv)
    reload(rv_point)
    reload(c_appx_router)


class ResetClass(ElisaTestCase):
    lc_base_path = None
    default_law = None

    def reset_config(self):
        config.CK04_ATM_TABLES = op.join(self.lc_base_path, "atmosphere")
        config.LD_TABLES = op.join(self.lc_base_path, "limbdarkening")
        config.LIMB_DARKENING_LAW = self.default_law
        config.ATM_ATLAS = "ck04"
        config.NUMBER_OF_PROCESSES = -1
        config.POINTS_ON_ECC_ORBIT = 118
        config.MAX_RELATIVE_D_R_POINT = 3e-3
        config.MAX_SUPPLEMENTAR_D_DISTANCE = 1e-1
        config.MAX_SPOT_D_LONGITUDE = np.pi / 180.0
        config._update_atlas_to_base_dir()
        reload_modules()


class BinaryRadialCurvesTestCase(ResetClass):
    def setUp(self):
        self.phases = up.arange(-0.2, 1.25, 0.05)
        self.lc_base_path = op.join(op.dirname(op.abspath(__file__)), "data", "light_curves")
        self.default_law = "cosine"
        self.reset_config()

    def tearDown(self):
        self.reset_config()

    def do_comparison(self, system, file):
        rvdict = rv.com_radial_velocity(system, position_method=system.calculate_orbital_motion, phases=self.phases)
        obtained_rvp, obtained_rvs = normalize_lv_for_unittests(rvdict['primary'], rvdict['secondary'])
        expected = load_radial_curve(file)

        expected_rvp, expected_rvs = -1 * np.array(expected["primary"]), -1 * np.array(expected["secondary"])
        expected_rvp, expected_rvs = normalize_lv_for_unittests(expected_rvp, expected_rvs)

        assert_array_equal(np.round(expected_rvp, 4), np.round(obtained_rvp, 4))
        assert_array_equal(np.round(expected_rvs, 4), np.round(obtained_rvs, 4))

    def test_circular_detached(self):
        s = prepare_binary_system(BINARY_SYSTEM_PARAMS["detached-physical"])
        self.do_comparison(s, "detahed.circ.json")

    def test_eccentric_detached(self):
        s = prepare_binary_system(BINARY_SYSTEM_PARAMS["detached.ecc"])
        self.do_comparison(s, "detahed.ecc.json")


class BinaryRadialCurvesConsistencyTestCase(ResetClass):
    def setUp(self):
        self.lc_base_path = op.join(op.dirname(op.abspath(__file__)), "data", "light_curves")
        self.default_law = "cosine"
        self.reset_config()

    def tearDown(self):
        self.reset_config()

    @staticmethod
    def check_consistency(binary_kwargs, desired_delta, spots_primary=None, spots_secondary=None, phases=None):
        system = prepare_binary_system(binary_kwargs, spots_primary=spots_primary, spots_secondary=spots_secondary)

        o = observer.Observer(system=system)
        _, rvdict1 = o.rv(phases=phases, method='radiometric')
        _, rvdict2 = o.rv(phases=phases, method='point_mass')

        rvdict1['primary'], rvdict1['secondary'] = normalize_lv_for_unittests(rvdict1['primary'], rvdict1['secondary'])
        rvdict2['primary'], rvdict2['secondary'] = normalize_lv_for_unittests(rvdict2['primary'], rvdict2['secondary'])

        assert_array_less(np.abs(rvdict1['primary'] - rvdict2['primary']), desired_delta * np.ones(phases.shape))
        assert_array_less(np.abs(rvdict2['secondary'] - rvdict2['secondary']), desired_delta * np.ones(phases.shape))

        # from matplotlib import pyplot as plt
        # # plt.plot(self.phases, rvdict1['primary']-rvdict2['primary'], c='r')
        # plt.plot(self.phases, rvdict1['primary'], c='r')
        # plt.plot(self.phases, rvdict2['primary'], c='r', linestyle='dashed')
        # # plt.plot(self.phases, rvdict1['secondary']-rvdict2['secondary'], c='b')
        # plt.plot(self.phases, rvdict1['secondary'], c='b')
        # plt.plot(self.phases, rvdict2['secondary'], c='b', linestyle='dashed')
        # plt.show()

    def test_rv_consistency_circular_detached(self):
        binary_kwargs = copy(BINARY_SYSTEM_PARAMS["detached-physical"])
        binary_kwargs['inclination'] = 70 * u.deg
        phases = np.array([0.15, 0.2, 0.25, 0.3, 0.4, 0.7, 0.75, 0.8, 0.85])
        self.check_consistency(binary_kwargs, 0.02, spots_primary=SPOTS_META['primary'], phases=phases)

    def test_rv_consistency_circular_contact(self):
        binary_kwargs = copy(BINARY_SYSTEM_PARAMS["over-contact"])
        binary_kwargs['inclination'] = 10 * u.deg
        phases = np.array([0.15, 0.2, 0.25, 0.3, 0.4, 0.7, 0.75, 0.8, 0.85])
        self.check_consistency(binary_kwargs, 0.033, spots_primary=SPOTS_META['primary'], phases=phases)

    def test_rv_consistency_eccentric_approx_zero(self):
        config.POINTS_ON_ECC_ORBIT = -1
        config.MAX_RELATIVE_D_R_POINT = 0.0
        reload_modules()

        phases = np.array([0.15, 0.2, 0.25, 0.3, 0.4, 0.7, 0.75, 0.8, 0.85])
        binary_kwargs = copy(BINARY_SYSTEM_PARAMS["detached-physical"])
        binary_kwargs['inclination'] = 70.0 * u.deg
        binary_kwargs['eccentricity'] = 0.3
        self.check_consistency(binary_kwargs, 0.005, phases=phases)


class ComputeRadiometricRVTestCase(ResetClass):
    def setUp(self):
        # raise unittest.SkipTest(message)
        self.lc_base_path = op.join(op.dirname(op.abspath(__file__)), "data", "light_curves")
        self.default_law = "cosine"
        self.reset_config()

    def tearDown(self):
        self.reset_config()

    def do_comparison(self, bs, rv_file, start_phs=-0.2, stop_phs=1.2, step=0.1):
        o = observer.Observer(system=bs)

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
        bs = prepare_binary_system(PARAMS["detached"])
        start_phs, stop_phs, step = -0.2, 1.2, 0.1

        laws = config.LD_LAW_TO_FILE_PREFIX.keys()

        config_template = "[computational]\nnumber_of_processes={}\n[physics]\nlimb_darkening_law={}"
        config_path = op.join(tempfile.gettempdir(), "elisa.ini")

        try:
            for cpu_core in [-1, cpu_count()]:
                for law in laws:
                    os.environ["ELISA_CONFIG"] = config_path
                    config_data = config_template.format(cpu_core, law)

                    with open(config_path, "w") as f:
                        f.write(config_data)

                    config.read_and_update_config(config_path)
                    config.LD_TABLES = op.join(self.lc_base_path, "limbdarkening")
                    config.CK04_ATM_TABLES = op.join(self.lc_base_path, "atmosphere")
                    config.ATM_ATLAS = "ck04"
                    config._update_atlas_to_base_dir()

                    reload_modules()

                    o = observer.Observer(system=bs)
                    o.rv(from_phase=start_phs, to_phase=stop_phs, phase_step=step, method='radiometric')
        finally:
            os.remove(config_path)

    def test_circular_synchronous_detached_system(self):
        config.LIMB_DARKENING_LAW = "linear"
        reload_modules()

        bs = prepare_binary_system(PARAMS["detached"])
        self.do_comparison(bs, "detached.circ.sync.json")

    def test_circular_synchronous_overcontact_system(self):
        config.LIMB_DARKENING_LAW = "linear"
        reload_modules()

        bs = prepare_binary_system(PARAMS["over-contact"])
        self.do_comparison(bs, "overcontact.circ.sync.json")

    def test_circular_spotty_synchronous_detached_system(self):
        config.LIMB_DARKENING_LAW = "linear"
        reload_modules()

        bs = prepare_binary_system(PARAMS["detached"],
                                   spots_primary=SPOTS_META["primary"],
                                   spots_secondary=SPOTS_META["secondary"])
        self.do_comparison(bs, "detached.circ.spotty.sync.json")

    def test_circular_spotty_synchronous_overcontact_system(self):
        config.LIMB_DARKENING_LAW = "linear"
        reload_modules()

        bs = prepare_binary_system(PARAMS["over-contact"],
                                   spots_primary=SPOTS_META["primary"],
                                   spots_secondary=SPOTS_META["secondary"])
        self.do_comparison(bs, "overcontact.circ.spotty.sync.json")

    def test_cicular_spotty_asynchronous_detached_system(self):
        config.MAX_SPOT_D_LONGITUDE = up.pi / 45.0
        reload_modules()

        bs = prepare_binary_system(PARAMS["detached-async"], spots_primary=SPOTS_META["primary"])
        self.do_comparison(bs, "detached.circ.spotty.async.json")

    def test_eccentric_synchronous_detached_system_no_approximation(self):
        config.POINTS_ON_ECC_ORBIT = -1
        config.MAX_RELATIVE_D_R_POINT = 0.0
        reload_modules()

        bs = prepare_binary_system(PARAMS["eccentric"])

        self.do_comparison(bs, "detached.ecc.sync.json")

    def test_eccentric_system_approximation_one(self):
        config.POINTS_ON_ECC_ORBIT = 5
        config.MAX_RELATIVE_D_R_POINT = 0.0
        reload_modules()

        bs = prepare_binary_system(PARAMS["eccentric"])

        self.do_comparison(bs, "detached.ecc.appx_one.json")

    def test_eccentric_system_approximation_two(self):
        config.POINTS_ON_ECC_ORBIT = int(1e6)
        config.MAX_RELATIVE_D_R_POINT = 0.05
        config.MAX_SUPPLEMENTAR_D_DISTANCE = 0.05
        reload_modules()

        bs = prepare_binary_system(PARAMS["eccentric"])

        self.do_comparison(bs, "detached.ecc.appx_two.json", start_phs=-0.2, stop_phs=1.2, step=0.05)

    def test_eccentric_system_approximation_three(self):
        config.POINTS_ON_ECC_ORBIT = int(1e6)
        config.MAX_RELATIVE_D_R_POINT = 0.05
        reload_modules()

        bs = prepare_binary_system(PARAMS["eccentric"])

        self.do_comparison(bs, "ecc.appx_three.json", -0.0, 0.01, 0.002)

    def test_eccentric_spotty_asynchronous_detached_system(self):
        bs = prepare_binary_system(PARAMS["detached-async-ecc"], spots_primary=SPOTS_META["primary"])
        self.do_comparison(bs, "ecc.spotty.async.json")


class CompareSingleVsMultiprocess(ResetClass):
    def setUp(self):
        # raise unittest.SkipTest(message)
        self.lc_base_path = op.join(op.dirname(op.abspath(__file__)), "data", "light_curves")
        self.base_path = op.join(op.dirname(op.abspath(__file__)), "data", "radial_curves")
        self.default_law = "cosine"
        self.reset_config()

    def tearDown(self):
        self.reset_config()

    def do_comparison(self, system, start_phs=-0.2, stop_phs=1.2, step=0.1, tol=1e-8):
        config.NUMBER_OF_PROCESSES = -1
        reload_modules()

        o = observer.Observer(system=system)

        sp_res = o.rv(from_phase=start_phs, to_phase=stop_phs, phase_step=step, method='radiometric')
        sp_p = np.array(sp_res[1]['primary'])
        sp_s = np.array(sp_res[1]['secondary'])
        sp_p = sp_p[~np.isnan(sp_p)]
        sp_s = sp_s[~np.isnan(sp_s)]

        config.NUMBER_OF_PROCESSES = cpu_count()
        reload_modules()

        mp_res = o.rv(from_phase=start_phs, to_phase=stop_phs, phase_step=step, method='radiometric')
        mp_p = np.array(mp_res[1]['primary'])
        mp_s = np.array(mp_res[1]['secondary'])
        mp_p = mp_p[~np.isnan(mp_p)]
        mp_s = mp_s[~np.isnan(mp_s)]

        self.assertTrue(np.all((up.abs(sp_p - mp_p) / np.abs(np.max(sp_p))) < tol))
        self.assertTrue(np.all((up.abs(sp_s - mp_s) / np.abs(np.max(sp_s))) < tol))

    def test_circular_sync_rv(self):
        config.LIMB_DARKENING_LAW = "linear"
        reload_modules()

        bs = prepare_binary_system(PARAMS["detached"])
        self.do_comparison(bs)

    def test_circular_spotty_async_rv(self):
        config.MAX_SPOT_D_LONGITUDE = up.pi / 45.0
        reload_modules()

        bs = prepare_binary_system(PARAMS["detached-async"], spots_primary=SPOTS_META["primary"])
        self.do_comparison(bs)

    def test_eccentric_system_no_approximation(self):
        config.POINTS_ON_ECC_ORBIT = -1
        config.MAX_RELATIVE_D_R_POINT = 0.0
        reload_modules()

        bs = prepare_binary_system(PARAMS["eccentric"])
        self.do_comparison(bs)

    def test_eccentric_system_approximation_one(self):
        config.POINTS_ON_ECC_ORBIT = 5
        config.MAX_RELATIVE_D_R_POINT = 0.0
        reload_modules()

        bs = prepare_binary_system(PARAMS["eccentric"])
        self.do_comparison(bs, tol=1e-4)

    def test_eccentric_system_approximation_two(self):
        config.POINTS_ON_ECC_ORBIT = int(1e6)
        config.MAX_RELATIVE_D_R_POINT = 0.05
        config.MAX_SUPPLEMENTAR_D_DISTANCE = 0.05
        reload_modules()

        bs = prepare_binary_system(PARAMS["eccentric"])
        self.do_comparison(bs, tol=1e-3)

    def test_eccentric_system_approximation_three(self):
        config.POINTS_ON_ECC_ORBIT = int(1e6)
        config.MAX_RELATIVE_D_R_POINT = 0.05
        reload_modules()

        bs = prepare_binary_system(PARAMS["eccentric"])
        self.do_comparison(bs, -0.0, 0.01, 0.002, tol=1e-4)

    def test_eccentric_spotty_asynchronous_detached_system(self):
        bs = prepare_binary_system(PARAMS["detached-async-ecc"], spots_primary=SPOTS_META["primary"])
        self.do_comparison(bs)
