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

from elisa import units
from elisa.analytics import RVData, RVBinaryAnalyticsTask
from elisa.analytics.params.parameters import BinaryInitialParameters
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

    data = {comp: RVData(**{
        "x_data": xs[comp],
        "y_data": ys[comp],
        "y_err": yerr[comp],
        "x_unit": units.dimensionless_unscaled,
        "y_unit": units.m / units.s

    }) for comp in ["primary", "secondary"]}

    rv_initial = {
        "system": {
            "eccentricity": {
                "value": 0.0,
                "fixed": True,
            },
            "asini": {
                "value": 5.0,
                "fixed": False,
                "min": 1.0,
                "max": 15.0
            },
            "mass_ratio": {
                "value": 1.0,
                "fixed": False,
                "min": 0.1,
                "max": 1.5
            },
            "argument_of_periastron": {
                "value": 0.0,
                'fixed': True
            },
            "gamma": {
                "value": 0.0,
                "fixed": False,
                'min': -50000,
                'max': 50000
            },
            "period": {
                "value": P,
                "fixed": True
            }
        }
    }

    rv_initial = BinaryInitialParameters(**rv_initial)
    task = RVBinaryAnalyticsTask(data=data, method='least_squares')
    task.fit(x0=rv_initial)

    task.plot.model()


if __name__ == '__main__':
    main()
