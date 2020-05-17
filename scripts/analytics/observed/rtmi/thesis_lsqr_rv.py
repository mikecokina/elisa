"""
[
    {
        "param": "eccentricity",
        "value": 0.0324368140709541,
        "unit": "dimensionless"
    },
    {
        "param": "asini",
        "value": 11.525057216386077,
        "unit": "solRad"
    },
    {
        "param": "mass_ratio",
        "value": 1.0733059076858549,
        "unit": "dimensionless"
    },
    {
        "param": "argument_of_periastron",
        "value": 226.75552803002495,
        "unit": "degrees"
    },
    {
        "param": "gamma",
        "value": -24016.306698751767,
        "unit": "m/s"
    },
    {
        "param": "period",
        "value": 2.47028375753,
        "unit": "days"
    },
    {
        "r_squared": 0.9965987925987034
    }
]

"""

import json
import os.path as op
import numpy as np

from matplotlib import pyplot as plt
from elisa.analytics.binary.least_squares import central_rv
from elisa.binary_system import t_layer


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

    result = central_rv.fit(xs=xs, ys=ys, x0=rv_initial, xtol=1e-10, yerrs=yerr)
    print(json.dumps(result, indent=4))


if __name__ == '__main__':
    main()
