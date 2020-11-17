import json
import os.path as op
import numpy as np
import builtins

from matplotlib import pyplot as plt

from elisa import units
from elisa.analytics import RVData, RVBinaryAnalyticsTask
from elisa.analytics.params.parameters import BinaryInitialParameters
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
                "value": 0.14,
                "fixed": False,
                'min': 0.0,
                'max': 0.3
            },
            "asini": {
                "value": 11.04,
                "fixed": False,
                'min': 10.0,
                'max': 15.0
            },
            "mass_ratio": {
                "value": 1.0,
                "fixed": False,
                'min': 0.1,
                'max': 1.5
            },
            "argument_of_periastron": {
                'value': 258.83,
                "fixed": False,
                'min': 180,
                'max': 360
            },
            "gamma": {
                "value": 0.0,
                "fixed": False,
                'min': -50000.0,
                'max': 50000.0
            },
            "period": {
                'value': P,
                'fixed': True
            }
        }
    }

    rv_initial = BinaryInitialParameters(**rv_initial)
    task = RVBinaryAnalyticsTask(data=data, method='mcmc')
    task.fit(x0=rv_initial, nsteps=8000, burn_in=4000, save=True, fit_id="mcmc_rv_fit_no_period", progress=True)
    task.plot.model()
    task.plot.corner(truths=True)


if __name__ == '__main__':
    main()
