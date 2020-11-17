import json
import os.path as op
import numpy as np

from astropy import units as au
from elisa.analytics import RVData, RVBinaryAnalyticsTask
from elisa.analytics.params.parameters import BinaryInitialParameters

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
    _max = np.max(list(rv.values()))
    rv = {comp: u(val, sigma, n) for comp, val in rv.items()}
    rv_err = {comp: sigma * np.ones(val.shape) for comp, val in rv.items()}

    data = {comp: RVData(**{
        "x_data": phases,
        "y_data": rv[comp],
        "y_err": rv_err[comp],
        "x_unit": au.dimensionless_unscaled,
        "y_unit": au.m / au.s

    }) for comp in rv}

    rv_initial = {
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
                "value": 3,
                "fixed": False,
                "min": 0.1,
                "max": 10
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
                "fixed": True
            }
        }
    }

    rv_initial = BinaryInitialParameters(**rv_initial)
    task = RVBinaryAnalyticsTask(data=data, method='mcmc')
    task.fit(x0=rv_initial, nsteps=1000, burn_in=100, save=True, fit_id="mcmc_rv_fit")


if __name__ == '__main__':
    main()
