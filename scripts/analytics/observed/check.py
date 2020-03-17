import json
import numpy as np
import os.path as op

from elisa.binary_system import t_layer
from elisa.binary_system.curves.community import RadialVelocitySystem
from elisa.observer.observer import Observer
from matplotlib import pyplot as plt


P = 2.47028375753
T0 = 54953.900333


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

    xs_syn = np.arange(min(xs["primary"]), max(xs["primary"]), 0.005)
    rv = rv_sys(e=0.0324, omega=226.7555, p=P, q=1.073, asini=11.52, gamma=-24016.3067, xs=xs_syn)

    plt.scatter(xs["primary"], ys["primary"], c="g", label="observation")
    plt.scatter(xs["primary"], ys["secondary"], c="g")

    plt.plot(xs_syn, rv["primary"], c="r", label="fit")
    plt.plot(xs_syn, rv["secondary"], c="r", )

    plt.xlabel('Phase')
    plt.ylabel(r'Radial Velocity/($m/s$)')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
