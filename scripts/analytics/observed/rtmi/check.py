import json
import numpy as np
import os.path as op

from elisa.binary_system import t_layer
from elisa.binary_system.curves.community import RadialVelocitySystem
from elisa.observer.observer import Observer
from matplotlib import pyplot as plt


P = 0.37491730
T0 = 52500.2071


def get_rv():
    fpath = op.join("rv.json")
    with open(fpath, "r") as f:
        return json.loads(f.read())


def rv_sys(e, omega, p, q, asini, gamma, xs):

    data = dict(
        eccentricity=e,
        argument_of_periastron=omega,
        period=p,
        mass_ratio=q,
        asini=asini,
        gamma=gamma
    )
    system = RadialVelocitySystem(**data)
    observer = Observer(passband='bolometric', system=system)
    _, rv = observer.observe.rv(phases=xs, normalize=False)
    return rv


def main():
    rv = get_rv()
    xs = {
        "primary": np.array(rv["primary"]["jd"]),
        "secondary": np.array(rv["secondary"]["jd"]),
    }

    ys = {
        "primary": np.array(rv["primary"]["vr"]) * 1e3,
        "secondary": np.array(rv["secondary"]["vr"]) * 1e3,
    }

    yerr = {
        "primary": np.array(rv["primary"]["err"]) * 1e3,
        "secondary": np.array(rv["secondary"]["err"]) * 1e3,
    }

    xs["primary"] = t_layer.jd_to_phase(T0, P, xs["primary"])
    xs["secondary"] = t_layer.jd_to_phase(T0, P, xs["secondary"])

    xs_syn = np.arange(0.0, 1.5, 0.005)
    rv = rv_sys(e=0.0, omega=0.0, p=P, q=0.35, asini=2.63, gamma=-14718.46, xs=xs_syn)

    plt.scatter(xs["primary"], ys["primary"] / 1e3, c="g", label="observation")
    plt.scatter(xs["primary"], ys["secondary"] / 1e3, c="g")

    phase_mask = xs["primary"] <= 0.5

    plt.scatter(xs["primary"][phase_mask] + 1, ys["primary"][phase_mask] / 1e3, c="g")
    plt.scatter(xs["primary"][phase_mask] + 1, ys["secondary"][phase_mask] / 1e3, c="g")

    plt.plot(xs_syn, rv["primary"] / 1e3, c="r", label="fit")
    plt.plot(xs_syn, rv["secondary"] / 1e3, c="r", )

    plt.xlabel('Phase')
    plt.ylabel(r'Radial Velocity/($km/s$)')
    plt.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0))
    plt.show()


if __name__ == '__main__':
    main()
