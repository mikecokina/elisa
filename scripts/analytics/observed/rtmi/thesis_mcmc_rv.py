import json
import os.path as op
import numpy as np
import builtins

from matplotlib import pyplot as plt
from elisa.analytics.binary.mcmc import central_rv
from elisa.analytics.binary.models import central_rv_synthetic
from elisa.analytics.binary.shared import rv_r_squared
from elisa.binary_system import t_layer

builtins._ASTROPY_SETUP_ = True


P = 0.37491730
T0 = 52500.2071
t0 = 2452500.2071
do_plot = False


def get_rv():
    fpath = op.join("rv.json")
    with open(fpath, "r") as f:
        return json.loads(f.read())


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

    if do_plot:
        plt.scatter(x=xs["primary"], y=ys["primary"], c="b")
        plt.scatter(x=xs["secondary"], y=ys["secondary"], c="r")
        plt.show()

    rv_initial = [
        {
            'value': 0.0,
            'param': 'eccentricity',
            'fixed': True
        },
        {
            'value': 5.0,
            'param': 'asini',
            'fixed': False,
            'min': 1.0,
            'max': 15.0

        },
        {
            'value': 1.0,
            'param': 'mass_ratio',
            'fixed': False,
            'min': 0.0,
            'max': 1.5
        },
        {
            'value': 0.0,
            'param': 'argument_of_periastron',
            'fixed': True
        },
        {
            'value': 0.0,
            'param': 'gamma',
            'fixed': False,
            'min': -50000.0,
            'max': 50000.0
        },
        {
            'value': P,
            'param': 'period',
            'fixed': True
        }
    ]

    # central_rv.fit(xs=xs, ys=ys, x0=rv_initial, nwalkers=20, nsteps=50000, nsteps_burn_in=5000, yerrs=yerr)
    # result = central_rv.restore_flat_chain(central_rv.last_fname)
    # central_rv.plot.corner(result['flat_chain'], result['labels'], renorm=result['normalization'])

    result = {
        "eccentricity": 0.0,
        "asini": 2.64,
        "mass_ratio": 0.35,
        "argument_of_periastron": 0.0,
        "gamma": -14717.23,
        "period": P,
    }
    r_squared_args = xs["primary"], ys, False, {"primary": np.arange(0, len(xs["primary"]), 1),
                                                "secondary": np.arange(0, len(xs["secondary"]), 1)}
    r_squared_result = rv_r_squared(central_rv_synthetic, *r_squared_args, **result)
    print(r_squared_result)


if __name__ == '__main__':
    main()
