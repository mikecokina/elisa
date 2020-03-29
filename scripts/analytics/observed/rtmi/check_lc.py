import json
import numpy as np
import os.path as op

from elisa.analytics.binary.models import synthetic_binary
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

    ys = {band: np.array(lc[band]["mag"])[xs_argsort[band]] for band in passbands}
    phase_mask = phase_filter(xs, threshold=0.01)

    xs_syn = np.arange(0.0, 1.0, 0.01)
    ys_syn = lc_sys(xs_syn,
                    p=P, a=2.6440, i=84.10, q=0.37, t1=4400, t2=4500, omega_1=2.568,
                    omega_2=2.568, a_1=0.6, a_2=0.6, g_1=0.32, g_2=0.32)

    ys_syn_max = {band: max(np.array(ys_syn[band])) for band in passbands}
    ys_syn = {band: ys_syn[band] / ys_syn_max[band] for band in passbands}

    ys = mag_to_flx(ys)
    ys_max = {band: max(np.array(ys[band])) for band in passbands}
    ys = {band: ys[band] / ys_max[band] for band in passbands}

    for band in passbands:
        plt.scatter(xs[band] - 0.01, y=ys[band] - ys_shift[band], c=color_map[band])
        plt.plot(xs_syn, ys_syn[band] - ys_shift[band], c=color_map[band])

    plt.xlabel('Phase')
    plt.ylabel('Flux')
    plt.show()



    # xs = {
    #     "primary": np.array(rv["primary"]["jd"]),
    #     "secondary": np.array(rv["secondary"]["jd"]),
    # }
    #
    # ys = {
    #     "primary": np.array(rv["primary"]["vr"]) * 1e3,
    #     "secondary": np.array(rv["secondary"]["vr"]) * 1e3,
    # }
    #
    # yerr = {
    #     "primary": np.array(rv["primary"]["err"]) * 1e3,
    #     "secondary": np.array(rv["secondary"]["err"]) * 1e3,
    # }
    #
    # xs["primary"] = t_layer.jd_to_phase(T0, P, xs["primary"])
    # xs["secondary"] = t_layer.jd_to_phase(T0, P, xs["secondary"])
    #
    # xs_syn = np.arange(0.0, 1.5, 0.005)
    # rv = lc_sys(e=0.0, omega=0.0, p=P, q=0.35, asini=2.63, gamma=-14718.46, xs=xs_syn)
    #
    # plt.scatter(xs["primary"], ys["primary"] / 1e3, c="g", label="observation")
    # plt.scatter(xs["primary"], ys["secondary"] / 1e3, c="g")
    #
    # phase_mask = xs["primary"] <= 0.5
    #
    # plt.scatter(xs["primary"][phase_mask] + 1, ys["primary"][phase_mask] / 1e3, c="g")
    # plt.scatter(xs["primary"][phase_mask] + 1, ys["secondary"][phase_mask] / 1e3, c="g")
    #
    # plt.plot(xs_syn, rv["primary"] / 1e3, c="r", label="fit")
    # plt.plot(xs_syn, rv["secondary"] / 1e3, c="r", )
    #
    # plt.xlabel('Phase')
    # plt.ylabel(r'Radial Velocity/($km/s$)')
    # plt.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0))
    # plt.show()


if __name__ == '__main__':
    main()
