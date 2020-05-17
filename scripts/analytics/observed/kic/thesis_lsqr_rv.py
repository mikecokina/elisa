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

from elisa.analytics.binary.least_squares import central_rv
from elisa.binary_system import t_layer


P = 2.47028375753
T0 = 54953.900333


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

    rv_initial = [
        {
            'value': 0.15,
            'param': 'eccentricity',
            'fixed': False,
            'min': 0.0,
            'max': 0.3
        },
        {
            'value': 12.5,
            'param': 'asini',
            'fixed': False,
            'min': 10.0,
            'max': 15.0

        },
        {
            'value': 1.0,
            'param': 'mass_ratio',
            'fixed': False,
            'min': 1.0,
            'max': 1.5
        },
        {
            'value': 220.0,
            'param': 'argument_of_periastron',
            'fixed': False,
            'min': 180,
            'max': 360
        },
        {
            'value': 0.0,
            'param': 'gamma',
            'fixed': False,
            'min': -50000,
            'max': 0.0
        },
        {
            'value': P,
            'param': 'period',
            'fixed': True,
        }
    ]

    result = central_rv.fit(xs=xs, ys=ys, x0=rv_initial, xtol=1e-10, yerrs=yerr)
    print(json.dumps(result, indent=4))


if __name__ == '__main__':
    main()
