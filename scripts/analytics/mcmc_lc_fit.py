import json
import os.path as op

import numpy as np
from elisa.analytics import LCBinaryAnalyticsTask
from elisa.analytics import LCData
from elisa.analytics.params.parameters import BinaryInitialParameters
from elisa import units

np.random.seed(1)
DATA = op.join(op.abspath(op.dirname(__file__)), "data")


def get_lc():
    fpath = op.join(DATA, "lc.json")
    with open(fpath, "r") as f:
        return json.loads(f.read())


def main():
    lc = get_lc()
    phases = {band: np.arange(-0.6, 0.62, 0.02) for band in lc}
    n = len(lc["Generic.Bessell.V"])

    sigma = 0.004
    bias = {passband: np.random.normal(0, sigma, n) for passband, curve in lc.items()}
    lc = {passband: curve + bias[passband] for passband, curve in lc.items()}
    lc_err = {passband: sigma * np.ones(curve.shape) for passband, curve in lc.items()}

    data = {passband: LCData(**{
        "x_data": phases[passband],
        "y_data": lc[passband],
        "y_err": lc_err[passband],
        "x_unit": units.dimensionless_unscaled,
        "y_unit": units.dimensionless_unscaled,
        "passband": passband
    }) for passband in lc}

    lc_initial = {
        "system": {
            "semi_major_axis": {
                "value": 16.515,
                "constraint": "16.515 / sin(radians(system@inclination))"
            },
            "inclination": {
                "value": 85.0,
                "fixed": False,
                "min": 80,
                "max": 90
            },
            "argument_of_periastron": {
                "value": 0.0,
                "fixed": True
            },
            "mass_ratio": {
                "value": 0.5,
                "fixed": True
            },
            "eccentricity": {
                "value": 0.0,
                "fixed": True
            },
            "period": {
                "value": 4.5,
                "fixed": True,
                "unit": units.d
            }
        },
        "primary": {
            "t_eff": {
                "value": 8307.0,
                "fixed": False,
                "min": 7800.0,
                "max": 8800.0,
                "unit": units.K
            },
            "surface_potential": {
                "value": 3.0,
                "fixed": False,
                "min": 3,
                "max": 5
            },
            "gravity_darkening": {
                "value": 0.32,
                "fixed": True
            },
            "albedo": {
                "value": 0.6,
                "fixed": True
            },
        },
        "secondary": {
            "t_eff": {
                "value": 4000.0,
                "fixed": False,
                "min": 4000.0,
                "max": 7000.0
            },
            "surface_potential": {
                "value": 5.0,
                "fixed": False,
                "min": 5.0,
                "max": 7.0
            },
            "gravity_darkening": {
                "value": 0.32,
                "fixed": True
            },
            "albedo": {
                "value": 0.6,
                "fixed": True
            }
        }
    }
    lc_initial = BinaryInitialParameters(**lc_initial)
    task = LCBinaryAnalyticsTask(data=data, method='mcmc', expected_morphology="detached")
    task.fit(x0=lc_initial, nsteps=100, mwalkers=100, save=True, fit_id="mcmc_lc_fit", progress=True)


if __name__ == '__main__':
    main()
