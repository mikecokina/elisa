

import json
import os.path as op
import numpy as np
import builtins

from matplotlib import pyplot as plt
from elisa.analytics.binary.mcmc import binary_overcontact
from elisa.binary_system import t_layer
from elisa.analytics.bvi import elisa_bv_temperature
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

    temperature = temperature_estimation(xs, ys_mag)
    print(temperature)

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

    lc_initial = [
        {
            'value': 2.63,
            'param': 'semi_major_axis',
            'constraint': '2.63 / sin(radians({inclination}))'
        },
        {
            'value': 5000.0,
            'param': 'p__t_eff',
            'fixed': False,
            'min': 4000.0,
            'max': 7000.0
        },
        {
            'value': 2.55,
            'param': 'p__surface_potential',
            'fixed': False,
            'min': 2.0,
            'max': 3.0
        },
        {
            'value': 2.55,
            'param': 's__surface_potential',
            'fixed': False,
            'min': 2.0,
            'max': 3.0
        },
        {
            'value': 5000.0,
            'param': 's__t_eff',
            'constraint': '{p__t_eff}',
        },
        {
            'value': 85.0,
            'param': 'inclination',
            'fixed': False,
            'min': 80,
            'max': 90
        },
        {
            'value': 0.32,
            'param': 'p__gravity_darkening',
            'fixed': True
        },
        {
            'value': 0.32,
            'param': 's__gravity_darkening',
            'fixed': True
        },
        {
            'value': 0.6,
            'param': 'p__albedo',
            'fixed': True
        },
        {
            'value': 0.6,
            'param': 's__albedo',
            'fixed': True
        },
        {
            'value': 0.0,
            'param': 'argument_of_periastron',
            'fixed': True
        },
        {
            'value': 0.35,
            'param': 'mass_ratio',
            'fixed': False,
            'min': 0.33,
            'max': 0.37
        },
        {
            'value': 0.0,
            'param': 'eccentricity',
            'fixed': True
        }
    ]

    binary_overcontact.fit(xs=xs, ys=ys, x0=lc_initial, period=P, discretization=5.0,
                           nwalkers=20, nsteps=20000, nsteps_burn_in=3000, yerrs=yerrs)

    # result = binary_overcontact.restore_flat_chain("2020-03-27T18.26.59")
    # binary_overcontact.plot.corner(result['flat_chain'], result['labels'], renorm=result['normalization'])


if __name__ == '__main__':
    main()
