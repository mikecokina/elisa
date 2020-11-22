import json
import os.path as op
import numpy as np
import builtins

from matplotlib import pyplot as plt

from elisa import units, u, settings
from elisa.analytics import LCData, LCBinaryAnalyticsTask
from elisa.analytics.params.parameters import BinaryInitialParameters
from elisa.analytics.tools.bvi import elisa_bv_temperature
from elisa.binary_system import t_layer
from scipy.interpolate import Akima1DInterpolator

builtins._ASTROPY_SETUP_ = True

passbands = ["Generic.Bessell.B", "Generic.Bessell.V", "Generic.Bessell.R", "Generic.Bessell.I"]
color_map = {"Generic.Bessell.B": "b", "Generic.Bessell.V": "g", "Generic.Bessell.R": "r", "Generic.Bessell.I": "m"}

P = 0.37491730
T0 = 52500.2071
t0 = 2452500.2071
do_plot = False


def get_lc():
    fpath = op.join("lc.json")
    with open(fpath, "r") as f:
        return json.loads(f.read())


def phase_filter(phases, threshold=0.05):
    phase_mask = {band: [True] for band in passbands}
    for band in passbands:
        xs = phases[band]
        xs_roll = np.roll(xs, shift=1)
        diff = np.abs(xs - xs_roll)

        cum_sum = 0
        for _diff in diff[1:]:
            cum_sum += _diff
            if cum_sum >= threshold:
                cum_sum = 0
                phase_mask[band].append(True)
            else:
                phase_mask[band].append(False)
    return phase_mask


def mag_to_flx(ys):
    m2, f2 = 0.0, 10e10
    return {band: f2 * np.power(10, ((m2 - ys[band]) / 2.5)) for band in passbands}


def temperature_estimation(xs, ys):
    interp_b = Akima1DInterpolator(xs['Generic.Bessell.B'], ys['Generic.Bessell.B'])
    interp_v = Akima1DInterpolator(xs['Generic.Bessell.V'], ys['Generic.Bessell.V'])

    b_mag = np.array([interp_b(0.25), interp_b(0.75)])
    v_mag = np.array([interp_v(0.25), interp_v(0.75)])

    bv = b_mag - v_mag
    t1, t2 = elisa_bv_temperature(bv, morphology="over-contact")
    return t1, t2


def main():
    lc = get_lc()

    xs = {band: t_layer.jd_to_phase(t0, P, np.array(lc[band]["jd"])) for band in passbands}
    xs_argsort = {band: np.argsort(xs[band]) for band in passbands}
    xs = {band: xs[band][xs_argsort[band]] for band in passbands}

    ys_mag = {band: np.array(lc[band]["mag"])[xs_argsort[band]] for band in passbands}
    phase_mask = phase_filter(xs, threshold=0.01)

    ys_flx = mag_to_flx(ys_mag)
    ys_max = {band: max(np.array(ys_flx[band])) for band in passbands}
    ys = {band: ys_flx[band] / ys_max[band] for band in passbands}

    if do_plot:
        for band in passbands:
            plt.scatter(xs[band][phase_mask[band]], y=ys[band][phase_mask[band]], c=color_map[band])
        plt.show()

    yerrs_mag = {band: np.array(np.array(lc[band]["err"]))[xs_argsort[band]] for band in passbands}

    ys_mag_max = {band: ys_mag[band] - yerrs_mag[band] for band in passbands}
    ys_mag_min = {band: ys_mag[band] + yerrs_mag[band] for band in passbands}

    ys_flx_max = mag_to_flx(ys_mag_max)
    ys_flx_min = mag_to_flx(ys_mag_min)

    yerrs = {band: (ys_flx_max[band] - ys_flx_min[band]) / (2. * ys_max[band])
             for band in passbands}

    xs = {band: xs[band][phase_mask[band]] for band in passbands}
    ys = {band: ys[band][phase_mask[band]] for band in passbands}
    yerrs = {band: yerrs[band][phase_mask[band]] for band in passbands}

    data = {passband: LCData(**{
        "x_data": xs[passband],
        "y_data": ys[passband],
        "y_err": yerrs[passband],
        "x_unit": units.dimensionless_unscaled,
        "y_unit": units.dimensionless_unscaled,
        "passband": passband
    }) for passband in lc}

    lc_initial = {
        "system": {
            "semi_major_axis": {
                "value": 2.633,
                "constraint": "2.633 / sin(radians(system@inclination))"
            },
            "inclination": {
                "value": 85.0,
                "fixed": False,
                "min": 80,
                "max": 90
            },
            "argument_of_periastron": {
                "value": 0.0,
                "fixed": True
            },
            "mass_ratio": {
                "value": 0.353,
                "fixed": True
            },
            "eccentricity": {
                "value": 0.0,
                "fixed": True
            },
            "period": {
                "value": P,
                "fixed": True,
                "unit": units.d
            }
        },
        "primary": {
            "t_eff": {
                "value": 6250.0,
                "fixed": False,
                "min": 6000.0,
                "max": 6500.0,
                "unit": units.K
            },
            "surface_potential": {
                "value": 2.5,
                "fixed": False,
                "min": 2,
                "max": 3
            },
            "gravity_darkening": {
                "value": 0.5,
                "fixed": False,
                "min": 0.3,
                "max": 1.0
            },
            "albedo": {
                "value": 0.75,
                "fixed": False,
                "min": 0.5,
                "max": 1.0
            },
        },
        "secondary": {
            "t_eff": {
                "value": 6250.0,
                "fixed": False,
                "min": 6000.0,
                "max": 6500.0
            },
            "surface_potential": {
                'constraint': 'primary@surface_potential',
            },
            "gravity_darkening": {
                "value": 0.5,
                "fixed": False,
                "min": 0.3,
                "max": 1.0
            },
            "albedo": {
                "value": 0.75,
                "fixed": False,
                "min": 0.5,
                "max": 1.0
            },
        }
    }

    lc_initial = BinaryInitialParameters(**lc_initial)
    task = LCBinaryAnalyticsTask(data=data, method='mcmc', expected_morphology="over-contact")
    task.fit(x0=lc_initial, nsteps=3000, nwalkers=1000, save=True, fit_id="thesis_mcmc_rtmi", progress=True)
    task.save_result(op.join(settings.HOME, "thesis_mcmc_rtmi.result.json"))
    task.plot.model()
    task.plot.corner(truths=True)


if __name__ == '__main__':
    main()
