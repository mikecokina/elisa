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


P = 2.47028375753
T0 = 54953.900333
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
            'value': 0.14,
            'param': 'eccentricity',
            'fixed': False,
            'min': 0.0,
            'max': 0.3
        },
        {
            'value': 11.04,
            'param': 'asini',
            'fixed': False,
            'min': 10.0,
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
            'value': 258.83,
            'param': 'argument_of_periastron',
            'fixed': False,
            'min': 180,
            'max': 360
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
            "eccentricity": 0.03234252471144975,
            "asini": 11.528396223783147,
            "mass_ratio": 1.0731164055537552,
            "argument_of_periastron": 226.28671717951596,
            "gamma": -24032.146612564076,
            "period": 2.47028375753,
    }

    r_squared_args = xs["primary"], ys, False, {"primary": np.arange(0, len(xs["primary"]), 1),
                                     "secondary": np.arange(0, len(xs["secondary"]), 1)}
    r_squared_result = rv_r_squared(central_rv_synthetic, *r_squared_args, **result)
    print(r_squared_result)


if __name__ == '__main__':
    main()
