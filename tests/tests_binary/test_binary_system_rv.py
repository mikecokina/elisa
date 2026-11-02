# keep it first
# due to stupid astropy units/constants implementation
from tests import set_astropy_units

import os
import pytest
import numpy as np
import os.path as op
from unittest import skip

from copy import copy
from numpy.testing import assert_array_equal, assert_array_less
from tests.utils import (
    ElisaTestCase,
    prepare_binary_system,
    BINARY_SYSTEM_PARAMS,
    SPOTS_META,
    normalize_rv_for_unittests,
    load_radial_curve,
    APPROX_SETTINGS
)
from elisa.observer.observer import Observer
from elisa.binary_system.curves import rv
from elisa import settings
from elisa import umpy as up, const
from elisa import units as u
from elisa.binary_system.curves.community import RadialVelocitySystem

set_astropy_units()


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
        "primary_albedo": 1.0, "secondary_albedo": 1.0, 'primary_discretization_factor': 3.0
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


class BinaryRadialCurvesTestCase(ElisaTestCase):
    def setUp(self):
        super(BinaryRadialCurvesTestCase, self).setUp()
        self.phases = up.arange(-0.2, 1.25, 0.05)
        # self.lc_base_path = op.join(op.dirname(op.abspath(__file__)), "data", "light_curves")
        # self.default_law = "cosine"
        # self.reset_config()

    def do_comparison(self, system, file):
        rvdict = rv.kinematic_radial_velocity(system, position_method=system.calculate_orbital_motion, phases=self.phases)
        obtained_rvp, obtained_rvs = normalize_rv_for_unittests(rvdict['primary'], rvdict['secondary'])
        expected = load_radial_curve(file)

        expected_rvp, expected_rvs = -1 * np.array(expected["primary"]), -1 * np.array(expected["secondary"])
        expected_rvp, expected_rvs = normalize_rv_for_unittests(expected_rvp, expected_rvs)

        assert_array_equal(np.round(expected_rvp, 4), np.round(obtained_rvp, 4))
        assert_array_equal(np.round(expected_rvs, 4), np.round(obtained_rvs, 4))

    def test_circular_detached(self):
        s = prepare_binary_system(BINARY_SYSTEM_PARAMS["detached-physical"])
        self.do_comparison(s, "detahed.circ.json")

    def test_eccentric_detached(self):
        s = prepare_binary_system(BINARY_SYSTEM_PARAMS["detached.ecc"])
        self.do_comparison(s, "detahed.ecc.json")


@pytest.mark.requires_ld
@pytest.mark.requires_atmosphere
class BinaryRadialCurvesConsistencyTestCase(ElisaTestCase):
    def setUp(self):
        super(BinaryRadialCurvesConsistencyTestCase, self).setUp()
        self.lc_base_path = op.join(op.dirname(op.abspath(__file__)), "../data", "light_curves")
        # self.default_law = "cosine"
        # self.reset_config()

    @staticmethod
    def check_consistency(binary_kwargs, desired_delta, spots_primary=None, spots_secondary=None, phases=None):
        system = prepare_binary_system(binary_kwargs, spots_primary=spots_primary, spots_secondary=spots_secondary)

        o = Observer(system=system)
        _, rvdict1 = o.rv(phases=phases, method='radiometric')
        _, rvdict2 = o.rv(phases=phases, method='kinematic')

        rvdict1['primary'], rvdict1['secondary'] = normalize_rv_for_unittests(rvdict1['primary'], rvdict1['secondary'])
        rvdict2['primary'], rvdict2['secondary'] = normalize_rv_for_unittests(rvdict2['primary'], rvdict2['secondary'])

        assert_array_less(np.abs(rvdict1['primary'] - rvdict2['primary']), desired_delta * np.ones(phases.shape))
        assert_array_less(np.abs(rvdict2['secondary'] - rvdict2['secondary']), desired_delta * np.ones(phases.shape))

        # from matplotlib import pyplot as plt
        # # plt.plot(self.phases, rvdict1['primary']-rvdict2['primary'], c='r')
        # plt.plot(phases, rvdict1['primary'], c='r')
        # plt.plot(phases, rvdict2['primary'], c='r', linestyle='dashed')
        # # plt.plot(self.phases, rvdict1['secondary']-rvdict2['secondary'], c='b')
        # plt.plot(phases, rvdict1['secondary'], c='b')
        # plt.plot(phases, rvdict2['secondary'], c='b', linestyle='dashed')
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
        # reload_modules()
        settings.configure(**{"MAX_NU_SEPARATION": -1, "MAX_D_FLUX": 0.0})

        phases = np.array([0.15, 0.2, 0.25, 0.3, 0.4, 0.7, 0.75, 0.8, 0.85])
        binary_kwargs = copy(BINARY_SYSTEM_PARAMS["detached-physical"])
        binary_kwargs['inclination'] = 70.0 * u.deg
        binary_kwargs['eccentricity'] = 0.3
        self.check_consistency(binary_kwargs, 0.005, phases=phases)


@pytest.mark.requires_ld
@pytest.mark.requires_atmosphere
class ComputeRadiometricRVTestCase(ElisaTestCase):
    def setUp(self):
        super(ComputeRadiometricRVTestCase, self).setUp()
        self.lc_base_path = op.join(op.dirname(op.abspath(__file__)), "../data", "light_curves")


    def do_comparison(self, bs, rv_file, start_phs=-0.2, stop_phs=1.2, step=0.1):
        o = Observer(system=bs)

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

        # from matplotlib import pyplot as plt
        # # plt.plot(obt_phs, (obt_p-exp_p)/obt_p.max(), c='r')
        # plt.plot(obt_phs, obt_p, c='r')
        # plt.plot(exp_phs, exp_p, c='r', linestyle='dashed')
        # # plt.plot(obt_phs, (obt_s-exp_s)/obt_p.max(), c='b')
        # plt.plot(obt_phs, obt_s, c='b')
        # plt.plot(exp_phs, exp_s, c='b', linestyle='dashed')
        # plt.show()

        self.assertTrue(np.all(up.abs(obt_phs - np.round(exp_phs, 3)) < TOL))
        self.assertTrue(np.all((up.abs(obt_p - exp_p) / np.abs(np.max(obt_p))) < TOL))
        self.assertTrue(np.all((up.abs(obt_s - exp_s) / np.abs(np.max(obt_s))) < TOL))

    @pytest.mark.requires_ld(
        passbands=("Generic.Bessell.V", "bolometric"), laws=("linear", "logarithmic", "square_root")
    )
    @pytest.mark.requires_atmosphere(atlas="ck04")
    def test_light_curve_pass_on_all_ld_law(self):
        """
        no assert here, it just has to pass without error
        """
        bs = prepare_binary_system(PARAMS["detached"])
        start_phs, stop_phs, step = -0.2, 1.2, 0.1

        laws = settings.LD_LAW_TO_FILE_PREFIX.keys()

        # idea is to update configuration content for problematic values in default configuration file
        content_tempalte = "[physics]\n" \
                           "limb_darkening_law={ld_law}\n" \
                           "[computational]\n" \
                           "number_of_processes={cpu}\n"

        for cpu_core in [-1, 2]:
            for law in laws:
                settings.configure(**{"NUMBER_OF_PROCESSES": cpu_core,
                                      "LIMB_DARKENING_LAW": law,
                                      "ATM_ATLAS": "ck04"})
                self.write_default_support(ld_tables=settings.LD_TABLES, atm_tables=settings.CK04_ATM_TABLES)
                with open(self.CONFIG_FILE, "a") as f:
                    f.write(content_tempalte.format(ld_law=law, cpu=cpu_core))

                o = Observer(system=bs)
                o.rv(from_phase=start_phs, to_phase=stop_phs, phase_step=step, method='radiometric')

                if op.isfile(self.CONFIG_FILE):
                    os.remove(self.CONFIG_FILE)

    def test_circular_synchronous_detached_system(self):
        bs = prepare_binary_system(PARAMS["detached"])
        self.do_comparison(bs, "detached.circ.sync.json")

    def test_circular_synchronous_overcontact_system(self):
        bs = prepare_binary_system(PARAMS["over-contact"])
        self.do_comparison(bs, "overcontact.circ.sync.json")

    def test_circular_spotty_synchronous_detached_system(self):
        bs = prepare_binary_system(PARAMS["detached"],
                                   spots_primary=SPOTS_META["primary"],
                                   spots_secondary=SPOTS_META["secondary"])
        self.do_comparison(bs, "detached.circ.spotty.sync.json")

    def test_circular_spotty_synchronous_overcontact_system(self):
        bs = prepare_binary_system(PARAMS["over-contact"],
                                   spots_primary=SPOTS_META["primary"],
                                   spots_secondary=SPOTS_META["secondary"])
        self.do_comparison(bs, "overcontact.circ.spotty.sync.json")

    def test_cicular_spotty_asynchronous_detached_system(self):
        settings.configure(**{"MAX_SPOT_D_LONGITUDE": up.pi / 45.0})
        bs = prepare_binary_system(PARAMS["detached-async"], spots_primary=SPOTS_META["primary"])
        self.do_comparison(bs, "detached.circ.spotty.async.json")

    def test_eccentric_synchronous_detached_system_no_approximation(self):
        settings.configure(**APPROX_SETTINGS["no_approx"])
        bs = prepare_binary_system(PARAMS["eccentric"])
        self.do_comparison(bs, "detached.ecc.sync.json")

    def test_eccentric_system_approximation_one(self):
        settings.configure(**APPROX_SETTINGS["interp_approx"])
        bs = prepare_binary_system(PARAMS["eccentric"])
        self.do_comparison(bs, "detached.ecc.appx_one.json", start_phs=-0.2, stop_phs=1.2, step=0.007)

    def test_eccentric_system_approximation_two(self):
        settings.configure(**APPROX_SETTINGS["symm_counterparts"])
        ec_params = PARAMS["eccentric"].copy()
        ec_params["argument_of_periastron"] = 90 * u.deg
        bs = prepare_binary_system(ec_params)
        self.do_comparison(bs, "detached.ecc.appx_two.json", start_phs=-0.2, stop_phs=1.2, step=0.05)

    def test_eccentric_system_approximation_three(self):
        settings.configure(**APPROX_SETTINGS["similar_neighbours"])
        bs = prepare_binary_system(PARAMS["eccentric"])
        self.do_comparison(bs, "ecc.appx_three.json", -0.0, 0.01, 0.002)

    def test_eccentric_spotty_asynchronous_detached_system(self):
        bs = prepare_binary_system(PARAMS["detached-async-ecc"], spots_primary=SPOTS_META["primary"])
        self.do_comparison(bs, "ecc.spotty.async.json")


@pytest.mark.requires_ld
@pytest.mark.requires_atmosphere
class CompareSingleVsMultiprocess(ElisaTestCase):
    def setUp(self):
        super(CompareSingleVsMultiprocess, self).setUp()
        self.lc_base_path = op.join(op.dirname(op.abspath(__file__)), "../data", "light_curves")
        self.base_path = op.join(op.dirname(op.abspath(__file__)), "../data", "radial_curves")

        self.write_default_support(ld_tables=settings.LD_TABLES, atm_tables=settings.CK04_ATM_TABLES)

    def tearDown(self):
        super(CompareSingleVsMultiprocess, self).tearDown()

    def do_comparison(self, system, start_phs=-0.2, stop_phs=1.2, step=0.1, tol=1e-8):
        settings.configure(**{"NUMBER_OF_PROCESSES": -1})

        o = Observer(system=system)

        sp_res = o.rv(from_phase=start_phs, to_phase=stop_phs, phase_step=step, method='radiometric')
        sp_p = np.array(sp_res[1]['primary'])
        sp_s = np.array(sp_res[1]['secondary'])
        sp_p = sp_p[~np.isnan(sp_p)]
        sp_s = sp_s[~np.isnan(sp_s)]

        settings.configure(**{"NUMBER_OF_PROCESSES":  2})

        mp_res = o.rv(from_phase=start_phs, to_phase=stop_phs, phase_step=step, method='radiometric')
        mp_p = np.array(mp_res[1]['primary'])
        mp_s = np.array(mp_res[1]['secondary'])
        mp_p = mp_p[~np.isnan(mp_p)]
        mp_s = mp_s[~np.isnan(mp_s)]

        self.assertTrue(np.all((up.abs(sp_p - mp_p) / np.abs(np.max(sp_p))) < tol))
        self.assertTrue(np.all((up.abs(sp_s - mp_s) / np.abs(np.max(sp_s))) < tol))

    def test_circular_sync_rv(self):
        bs = prepare_binary_system(PARAMS["detached"])
        self.do_comparison(bs)

    def test_circular_spotty_async_rv(self):
        settings.configure(**{"MAX_SPOT_D_LONGITUDE": up.pi / 45.0})

        # write config for stupid windows multiprocessing
        with open(self.CONFIG_FILE, "a") as f:
            f.write(f"[computational]"
                    f"\nmax_spot_d_longitude={settings.MAX_SPOT_D_LONGITUDE}"
                    f"\n")

        bs = prepare_binary_system(PARAMS["detached-async"], spots_primary=SPOTS_META["primary"])
        self.do_comparison(bs)

    def test_eccentric_system_no_approximation(self):
        settings.configure(**APPROX_SETTINGS["no_approx"])

        with open(self.CONFIG_FILE, "a") as f:
            f.write(f"[computational]"
                    f"\nmax_nu_separation={settings.MAX_NU_SEPARATION}"
                    f"\nmax_relative_d_r_point={settings.MAX_D_FLUX}"
                    f"\n")
        bs = prepare_binary_system(PARAMS["eccentric"])
        self.do_comparison(bs)

    def test_eccentric_system_approximation_one(self):
        settings.configure(**APPROX_SETTINGS["interp_approx"])

        with open(self.CONFIG_FILE, "a") as f:
            f.write(f"[computational]"
                    f"\nmax_nu_separation={settings.MAX_NU_SEPARATION}"
                    f"\nmax_relative_d_r_point={settings.MAX_D_FLUX}"
                    f"\n")

        bs = prepare_binary_system(PARAMS["eccentric"])
        self.do_comparison(bs, tol=1e-4)

    def test_eccentric_system_approximation_two(self):
        settings.configure(**APPROX_SETTINGS["symm_counterparts"])
        settings.configure(**{"MAX_D_FLUX": 0.0})

        with open(self.CONFIG_FILE, "a") as f:
            f.write(f"[computational]"
                    f"\nmax_nu_separation={settings.MAX_NU_SEPARATION}"
                    f"\nmax_relative_d_r_point={settings.MAX_D_FLUX}"
                    f"\n")

        bs = prepare_binary_system(PARAMS["eccentric"])
        self.do_comparison(bs, tol=2e-3)

    def test_eccentric_system_approximation_three(self):
        settings.configure(**APPROX_SETTINGS["similar_neighbours"])

        with open(self.CONFIG_FILE, "a") as f:
            f.write(f"[computational]"
                    f"\nmax_nu_separation={settings.MAX_NU_SEPARATION}"
                    f"\nmax_relative_d_r_point={settings.MAX_D_FLUX}")

        bs = prepare_binary_system(PARAMS["eccentric"])
        self.do_comparison(bs, -0.0, 0.01, 0.002, tol=1e-4)

    def test_eccentric_spotty_asynchronous_detached_system(self):
        bs = prepare_binary_system(PARAMS["detached-async-ecc"], spots_primary=SPOTS_META["primary"])
        self.do_comparison(bs)


@pytest.mark.requires_ld
@pytest.mark.requires_atmosphere
class CompareApproxVsExact(ElisaTestCase):
    def setUp(self):
        super(CompareApproxVsExact, self).setUp()
        self.lc_base_path = op.join(op.dirname(op.abspath(__file__)), "../data", "light_curves")
        self.base_path = op.join(op.dirname(op.abspath(__file__)), "../data", "radial_curves")

        self.write_default_support(ld_tables=settings.LD_TABLES, atm_tables=settings.CK04_ATM_TABLES)

    @skip
    def test_approximation_one(self):
        bs = prepare_binary_system(PARAMS["eccentric"])

        phase_step = 1.0/150
        settings.configure(**APPROX_SETTINGS["interp_approx"])
        settings.configure(**{"NUMBER_OF_PROCESSES": 4})
        o = Observer(system=bs)
        ap_res = o.rv(from_phase=-0.1, to_phase=1.1, phase_step=phase_step, method='radiometric')
        ap_p = np.array(ap_res[1]['primary'])
        ap_s = np.array(ap_res[1]['secondary'])
        ap_p = ap_p[~np.isnan(ap_p)]
        ap_s = ap_s[~np.isnan(ap_s)]

        settings.configure(**APPROX_SETTINGS["no_approx"])
        o = Observer(system=bs)
        ex_res = o.rv(from_phase=-0.1, to_phase=1.1, phase_step=phase_step, method='radiometric')
        ex_p = np.array(ex_res[1]['primary'])
        ex_s = np.array(ex_res[1]['secondary'])
        ex_p = ex_p[~np.isnan(ex_p)]
        ex_s = ex_s[~np.isnan(ex_s)]

        # import matplotlib.pyplot as plt
        # plt.plot(ex_res[0], (ex_p-ap_p)/ex_p.max(), label='primary')
        # plt.plot(ex_res[0], (ex_s-ap_s)/ex_p.max(), label='secondary')
        # plt.legend()
        # plt.show()

        self.assertTrue(np.all((up.abs(ex_p - ap_p) / np.abs(np.max(ex_p))) < 3e-3))
        self.assertTrue(np.all((up.abs(ex_s - ap_s) / np.abs(np.max(ex_s))) < 3e-3))

    @skip
    def test_approximation_two(self):
        bs = prepare_binary_system(PARAMS["eccentric"])

        settings.configure(**APPROX_SETTINGS["symm_counterparts"])
        settings.configure(**{"NUMBER_OF_PROCESSES": 4})
        o = Observer(system=bs)
        ap_res = o.rv(from_phase=-0.1, to_phase=1.1, phase_step=0.01, method='radiometric')
        ap_p = np.array(ap_res[1]['primary'])
        ap_s = np.array(ap_res[1]['secondary'])
        ap_p = ap_p[~np.isnan(ap_p)]
        ap_s = ap_s[~np.isnan(ap_s)]

        settings.configure(**APPROX_SETTINGS["no_approx"])
        o = Observer(system=bs)
        ex_res = o.rv(from_phase=-0.1, to_phase=1.1, phase_step=0.01, method='radiometric')
        ex_p = np.array(ex_res[1]['primary'])
        ex_s = np.array(ex_res[1]['secondary'])
        ex_p = ex_p[~np.isnan(ex_p)]
        ex_s = ex_s[~np.isnan(ex_s)]

        # import matplotlib.pyplot as plt
        # plt.plot(ex_res[0], ex_p / ex_s.max())
        # plt.plot(ex_res[0], ap_p / ex_s.max())
        # plt.plot(ex_res[0], ex_s / ex_s.max())
        # plt.plot(ex_res[0], ap_s / ex_s.max())
        # plt.plot(ex_res[0], (ex_p - ap_p) / ex_s.max(), label='primary')
        # plt.plot(ex_res[0], (ex_s - ap_s) / ex_s.max(), label='secondary')
        # plt.legend()
        # plt.show()

        self.assertTrue(np.all((up.abs(ex_p - ap_p) / np.abs(np.max(ex_s))) < 3e-3))
        self.assertTrue(np.all((up.abs(ex_s - ap_s) / np.abs(np.max(ex_s))) < 3e-3))

    @skip
    def test_approximation_three(self):
        bs = prepare_binary_system(PARAMS["eccentric"])

        settings.configure(**APPROX_SETTINGS["similar_neighbours"])
        settings.configure(**{"NUMBER_OF_PROCESSES": 4})
        o = Observer(system=bs)
        ap_res = o.rv(from_phase=-0.1, to_phase=0.6, phase_step=0.01, method='radiometric')
        ap_p = np.array(ap_res[1]['primary'])
        ap_s = np.array(ap_res[1]['secondary'])
        ap_p = ap_p[~np.isnan(ap_p)]
        ap_s = ap_s[~np.isnan(ap_s)]

        settings.configure(**APPROX_SETTINGS["no_approx"])
        o = Observer(system=bs)
        ex_res = o.rv(from_phase=-0.1, to_phase=0.6, phase_step=0.01, method='radiometric')
        ex_p = np.array(ex_res[1]['primary'])
        ex_s = np.array(ex_res[1]['secondary'])
        ex_p = ex_p[~np.isnan(ex_p)]
        ex_s = ex_s[~np.isnan(ex_s)]

        # import matplotlib.pyplot as plt
        # plt.plot(ex_res[0], (ex_p - ap_p) / ex_p.max(), label='primary')
        # plt.plot(ex_res[0], (ex_s - ap_s) / ex_p.max(), label='secondary')
        # plt.legend()
        # plt.show()

        self.assertTrue(np.all((up.abs(ex_p - ap_p) / np.abs(np.max(ex_p))) < 3e-3))
        self.assertTrue(np.all((up.abs(ex_s - ap_s) / np.abs(np.max(ex_s))) < 3e-3))


COMMUNITY_RV_TOL = 5e-3


class RadialVelocityObserverTestCase(ElisaTestCase):
    def setUp(self):
        super(RadialVelocityObserverTestCase, self).setUp()
        self.phases = up.arange(-0.2, 1.25, 0.05)

    def test_all_init_values_in_expected_units(self):
        init_kwargs = dict(
            eccentricity=0.1,
            argument_of_periastron=90 * u.deg,
            period=10.0,
            mass_ratio=0.5,
            asini=9 * u.solRad,
            gamma=11 * u.m / u.s
        )

        expected = np.round([0.1, const.HALF_PI, 10.0, 0.5, 9.0, 11], 4)
        o = RadialVelocitySystem(**init_kwargs)
        obtained = np.round([[o.eccentricity, o.argument_of_periastron,
                              o.period, o.mass_ratio, o.asini, o.gamma]], 4)
        self.assertTrue(np.all(np.abs(expected - obtained)) < COMMUNITY_RV_TOL)

    def test_rvs_from_binary_system_instance_are_same(self):
        s = prepare_binary_system(BINARY_SYSTEM_PARAMS["detached.ecc"])
        s.inclination = 1.1
        s.init()
        std_rvdict = rv.kinematic_radial_velocity(s, position_method=s.calculate_orbital_motion, phases=self.phases)

        asini = np.float64((s.semi_major_axis * np.sin(s.inclination) * u.m).to(u.solRad))

        rv_system = RadialVelocitySystem(eccentricity=s.eccentricity,
                                         argument_of_periastron=np.degrees(s.argument_of_periastron),
                                         period=s.period,
                                         mass_ratio=s.mass_ratio,
                                         asini=asini,
                                         gamma=s.gamma)
        o = Observer(passband='bolometric', system=rv_system)

        phases, com_rv_dict = o.observe.rv(phases=self.phases, method='kinematic')

        self.assertTrue(np.all(np.abs(std_rvdict['primary'] - com_rv_dict['primary']) < COMMUNITY_RV_TOL))
        self.assertTrue(np.all(np.abs(std_rvdict['secondary'] - com_rv_dict['secondary']) < COMMUNITY_RV_TOL))
