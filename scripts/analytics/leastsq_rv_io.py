import json
import os.path as op

import numpy as np
from astropy import units as au

from elisa.analytics import RVData, RVBinaryAnalyticsTask
from elisa.utils import random_sign

np.random.seed(1)
DATA = op.join(op.abspath(op.dirname(__file__)), "data")


def get_rv():
    fpath = op.join(DATA, "rv.json")
    with open(fpath, "r") as f:
        return json.loads(f.read())


def main():
    phases = np.arange(-0.6, 0.62, 0.02)
    rv = get_rv()
    u = np.random.normal
    n = len(phases)

    sigma = 2000
    rv = {comp: u(val, sigma, n) for comp, val in rv.items()}
    rv_err = {comp: sigma * np.ones(val.shape) for comp, val in rv.items()}

    data = {comp: RVData(**{
        "x_data": phases,
        "y_data": rv[comp],
        "y_err": rv_err[comp],
        "x_unit": au.dimensionless_unscaled,
        "y_unit": au.m / au.s

    }) for comp in rv}

    result = {
        "system": {
            "eccentricity": {
                "value": 0.002663944961117104,
                "fixed": False,
                "unit": None,
                "min": 0.0,
                "max": 0.5
            },
            "gamma": {
                "value": 19676.555153842037,
                "fixed": False,
                "unit": "m / s",
                "min": 10000.0,
                "max": 50000.0
            },
            "mass_ratio": {
                "value": 0.5089890168783162,
                "fixed": False,
                "unit": None,
                "min": 0.1,
                "max": 10.0
            },
            "asini": {
                "value": 16.445535068958847,
                "fixed": False,
                "unit": "solRad",
                "min": 10.0,
                "max": 20.0
            },
            "argument_of_periastron": {
                "value": 0.0,
                "fixed": True,
                "unit": "deg"
            },
            "period": {
                "value": 4.5,
                "fixed": True,
                "unit": "d"
            }
        }
    }

    task = RVBinaryAnalyticsTask(data=data, method='least_squares')
    task.set_result(result)
    task.result_summary()
    task.plot.model()


if __name__ == '__main__':
    main()
