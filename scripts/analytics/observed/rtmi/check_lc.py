import json
import numpy as np
import os.path as op

from elisa.analytics.binary import params
from elisa.analytics.binary.models import synthetic_binary
from elisa.analytics.binary.utils import normalize_lightcurve_to_max
from elisa.binary_system import t_layer
from elisa.observer.observer import Observer
from matplotlib import pyplot as plt


passbands = ["Generic.Bessell.B", "Generic.Bessell.V", "Generic.Bessell.R", "Generic.Bessell.I"]
# passbands = ["Generic.Bessell.B"]
color_map = {"Generic.Bessell.B": "b", "Generic.Bessell.V": "g", "Generic.Bessell.R": "r", "Generic.Bessell.I": "m"}
ys_shift = {"Generic.Bessell.B": 0.3, "Generic.Bessell.V": 0.2, "Generic.Bessell.R": 0.1, "Generic.Bessell.I": 0.0}

P = 0.37491730
t0 = 2452500.2071


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


def lc_sys(xs, p, a, i, q, t1, t2, omega_1, omega_2, a_1, a_2, g_1, g_2):

    observer = Observer(passband=passbands, system=None)
    observer._system_cls = '<class \'elisa.binary_system.system.BinarySystem\'>'

    kwargs = {
        "argument_of_periastron": 0.0,
        "eccentricity": 0.0,
        "inclination": i,
        "mass_ratio": q,
        "semi_major_axis": a,
        "p__t_eff": t1,
        "p__surface_potential": omega_1,
        "p__gravity_darkening": g_1,
        "p__albedo": a_1,
        "p__metallicity": 0.0,
        "p__synchronicity": 1.0,
        "s__t_eff": t2,
        "s__surface_potential": omega_2,
        "s__gravity_darkening": g_2,
        "s__albedo": a_2,
        "s__metallicity": 0.0,
        "s__synchronicity": 1.0,
        "gamma": 0.0
    }

    curve = synthetic_binary(xs, p, 3.0, "over-contact", observer, _raise_invalid_morphology=True, **kwargs)
    return curve


def mag_to_flx(ys):
    m2, f2 = 0.0, 10e10
    return {band: f2 * np.power(10, ((m2 - ys[band]) / 2.5)) for band in passbands}


def main():
    lc = get_lc()

    xs = {band: t_layer.jd_to_phase(t0, P, np.array(lc[band]["jd"])) for band in passbands}
    xs_argsort = {band: np.argsort(xs[band]) for band in passbands}
    xs = {band: xs[band][xs_argsort[band]] for band in passbands}

    ys_mag = {band: np.array(lc[band]["mag"])[xs_argsort[band]] for band in passbands}
    phase_mask = phase_filter(xs, threshold=0.01)

    # mag to flux ######################################################################################################
    ys_flx = mag_to_flx(ys_mag)
    ys_max = {band: max(np.array(ys_flx[band])) for band in passbands}
    ys = {band: ys_flx[band] / ys_max[band] for band in passbands}
    # ##################################################################################################################

    # error ############################################################################################################
    yerrs_mag = {band: np.array(np.array(lc[band]["err"]))[xs_argsort[band]] for band in passbands}

    ys_mag_max = {band: ys_mag[band] - yerrs_mag[band] for band in passbands}
    ys_mag_min = {band: ys_mag[band] + yerrs_mag[band] for band in passbands}

    ys_flx_max = mag_to_flx(ys_mag_max)
    ys_flx_min = mag_to_flx(ys_mag_min)

    yerrs = {band: (ys_flx_max[band] - ys_flx_min[band]) / (2. * ys_max[band]) for band in passbands}
    # ##################################################################################################################

    # phase mask #######################################################################################################
    xs = {band: xs[band][phase_mask[band]] for band in passbands}
    ys = {band: ys[band][phase_mask[band]] for band in passbands}
    yerrs = {band: yerrs[band][phase_mask[band]] for band in passbands}
    # ##################################################################################################################

    xs_syn, xs_reverser = params.xs_reducer(xs)

    p = dict(p=P, a=2.69, i=84.63, q=0.38, t1=6272.96, t2=6430.94, omega_1=2.536,
             omega_2=2.536, a_1=0.63, a_2=0.59, g_1=0.51, g_2=0.31)

    ys_syn = lc_sys(xs_syn, **p)
    ys_syn = normalize_lightcurve_to_max(ys_syn)

    for band in passbands:
        plt.scatter(xs[band] - 0.01, y=ys[band] - ys_shift[band], c=color_map[band])
        plt.plot(xs_syn, ys_syn[band] - ys_shift[band], c=color_map[band], label=band)

    plt.xlabel('Phase')
    plt.ylabel('Flux')
    plt.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0))
    plt.show()

    lhood = -0.5 * np.sum(np.array([np.sum(np.power((ys_syn[band][xs_reverser[band]] - ys[band])
                                                    / yerrs[band], 2)) for band in ys_syn]))
    print(lhood)


if __name__ == '__main__':
    main()
