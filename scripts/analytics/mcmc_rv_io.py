import json
import os.path as op

import numpy as np
from elisa import units as au

from elisa.analytics import RVData, RVBinaryAnalyticsTask
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

    result = {
        "system": {
            "eccentricity": {
                "value": 0.006390258902292462,
                "confidence_interval": {
                    "min": 0.000211201272860275,
                    "max": 0.01594528143175709
                },
                "fixed": False,
                "min": 0.0,
                "max": 0.5,
                "unit": None
            },
            "gamma": {
                "value": 18611.537085134805,
                "confidence_interval": {
                    "min": 18033.005328711417,
                    "max": 20043.024452146725
                },
                "fixed": False,
                "min": 10000.0,
                "max": 50000.0,
                "unit": "m / s"
            },
            "mass_ratio": {
                "value": 0.5333972280611943,
                "confidence_interval": {
                    "min": 0.48716297200917325,
                    "max": 0.5694281211593984
                },
                "fixed": False,
                "min": 0.1,
                "max": 10.0,
                "unit": None
            },
            "asini": {
                "value": 17.370367486603328,
                "confidence_interval": {
                    "min": 16.960580880518137,
                    "max": 17.87235430231025
                },
                "fixed": False,
                "min": 10.0,
                "max": 20.0,
                "unit": "solRad"
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
        },
        "r_squared": {
            "value": 0.9933561106448899,
            "unit": None
        }
    }

    task = RVBinaryAnalyticsTask(data=data, method='mcmc')
    task.set_result(result=result)
    task.load_chain("mcmc_rv_fit", discard=2000)
    task.plot.model()
    task.plot.corner(truths=True)
    task.plot.traces()
    task.plot.autocorrelation()


if __name__ == '__main__':
    main()
