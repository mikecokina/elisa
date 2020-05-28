import json
import os.path as op

import numpy as np

from elisa.analytics import RVData, RVBinaryAnalyticsTask
from elisa.analytics.params.parameters import BinaryInitialParameters
from elisa.binary_system import t_layer
from elisa import units

np.random.seed(1)
DATA = op.join(op.abspath(op.dirname(__file__)), "data")


def get_rv():
    fpath = op.join(DATA, "rv.json")
    with open(fpath, "r") as f:
        return json.loads(f.read())


def main():
    period, t0, phases = 4.5, 12.0, np.arange(-0.6, 0.62, 0.02)
    jd = t_layer.phase_to_jd(t0, period, phases)

    rv = get_rv()
    u = np.random.normal
    n = len(rv["primary"])

    sigma = 2000
    rv = {comp: u(val, sigma, n) for comp, val in rv.items()}
    rv_err = {comp: sigma * np.ones(val.shape) for comp, val in rv.items()}

    data = {comp: RVData(**{
        "x_data": jd,
        "y_data": rv[comp],
        "y_err": rv_err[comp],
        "x_unit": units.d,
        "y_unit": units.m / units.s

    }) for comp in rv}

    result = {
        "system": {
            "eccentricity": {
                "value": 0.2,
                "fixed": False,
                "min": 0.0,
                "max": 0.5
            },
            "asini": {
                "value": 15.0,
                "fixed": False,
                "min": 10.0,
                "max": 20.0
            },
            "mass_ratio": {
                "value": 0.6,
                "fixed": False,
                "min": 0.1,
                "max": 1.0
            },
            "argument_of_periastron": {
                "value": 0.0,
                "fixed": True
            },
            "gamma": {
                "value": 30000.0,
                "fixed": False,
                "min": 10000.0,
                "max": 50000.0
            },
            "period": {
                "value": 4.5,
                "fixed": False,
                "unit": units.d,
                "min": 4.4,
                "max": 4.6
            },
            "primary_minimum_time": {
                'value': 11.1,
                'fixed': False,
                'min': 11.1,
                'max': 12.1
            }
        }
    }

    task = RVBinaryAnalyticsTask(data=data, method='mcmc')
    task.set_result(result=result)
    task.load_chain("mcmc_rv_fit_no_period", discard=0)
    task.plot.model()
    task.plot.corner(truths=True)
    task.plot.traces()
    task.plot.autocorrelation()


if __name__ == '__main__':
    main()