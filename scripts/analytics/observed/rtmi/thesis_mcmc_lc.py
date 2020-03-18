

import json
import os.path as op
import numpy as np
import builtins

from matplotlib import pyplot as plt
from elisa.analytics.binary.mcmc import binary_overcontact
from elisa.binary_system import t_layer
from elisa.analytics.bvi import elisa_bv_temperature, pogsons_formula
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


def temperature_estimation(xs, ys):

    interp_b = Akima1DInterpolator(xs['Generic.Bessell.B'], ys['Generic.Bessell.B'])
    interp_v = Akima1DInterpolator(xs['Generic.Bessell.V'], ys['Generic.Bessell.V'])

    b_flux = np.array([interp_b(0.25), interp_b(0.75)])
    v_flux = np.array([interp_v(0.25), interp_v(0.75)])

    bv = pogsons_formula(b_flux, v_flux)
    t1, t2 = elisa_bv_temperature(bv, morphology="over-contact")
    return t1, t2


def main():
    lc = get_lc()

    ys_min = min([min(np.array(lc[band]["mag"])) for band in passbands])

    xs = {band: t_layer.jd_to_phase(t0, P, np.array(lc[band]["jd"])) for band in passbands}
    xs_argsort = {band: np.argsort(xs[band]) for band in passbands}
    xs = {band: xs[band][xs_argsort[band]] for band in passbands}

    ys = {band: np.array(2 + (-np.array(lc[band]["mag"]) / ys_min))[xs_argsort[band]] for band in passbands}
    yerrs = {band: np.array(np.array(lc[band]["err"]) / ys_min)[xs_argsort[band]] for band in passbands}

    phase_mask = phase_filter(xs, threshold=0.01)

    temperature = temperature_estimation(xs, ys)
    print(temperature)

    if do_plot:
        for band in passbands:
            plt.scatter(xs[band][phase_mask[band]], y=ys[band][phase_mask[band]], c=color_map[band])
        plt.show()

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
            'value': 6500.0,
            'param': 'p__t_eff',
            'fixed': False,
            'min': 5500.0,
            'max': 7500.0
        },
        {
            'value': 3.0,
            'param': 'p__surface_potential',
            'fixed': False,
            'min': 1.0,
            'max': 6.0
        },
        {
            'value': 3.0,
            'param': 's__surface_potential',
            'fixed': False,
            'min': 1.0,
            'max': 6.0
        },
        {
            'value': 4000.0,
            'param': 's__t_eff',
            'fixed': False,
            'min': 4000.0,
            'max': 6500.0
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
            'fixed': True
        },
        {
            'value': 0.0,
            'param': 'eccentricity',
            'fixed': True
        }
    ]

    binary_overcontact.fit(xs=xs, ys=ys, x0=lc_initial, period=P, discretization=5.0,
                           nwalkers=20, nsteps=20000, nsteps_burn_in=3000, yerrs=yerrs)


if __name__ == '__main__':
    main()
