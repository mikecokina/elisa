import json
import os.path as op

import numpy as np

from elisa import units
from elisa.analytics import LCBinaryAnalyticsTask
from elisa.analytics import LCData
from elisa.utils import random_sign

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

    result = {
        "primary": {
            "t_eff": {
                "value": 8099.941757469695,
                "fixed": False,
                "unit": "K",
                "min": 7800.0,
                "max": 8800.0
            },
            "surface_potential": {
                "value": 3.9079468419211425,
                "fixed": False,
                "unit": None,
                "min": 3.0,
                "max": 5.0
            },
            "albedo": {
                "value": 0.6,
                "fixed": True,
                "unit": None
            },
            "gravity_darkening": {
                "value": 0.32,
                "fixed": True,
                "unit": None
            }
        },
        "secondary": {
            "t_eff": {
                "value": 5970.458040348652,
                "fixed": False,
                "unit": "K",
                "min": 4000.0,
                "max": 7000.0
            },
            "surface_potential": {
                "value": 5.9166442699971284,
                "fixed": False,
                "unit": None,
                "min": 5.0,
                "max": 7.0
            },
            "albedo": {
                "value": 0.6,
                "fixed": True,
                "unit": None
            },
            "gravity_darkening": {
                "value": 0.32,
                "fixed": True,
                "unit": None
            }
        },
        "system": {
            "inclination": {
                "value": 83.56583703194174,
                "fixed": False,
                "unit": "deg",
                "min": 80.0,
                "max": 90.0
            },
            "eccentricity": {
                "value": 0.0,
                "fixed": True,
                "unit": None
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
            },
            "mass_ratio": {
                "value": 0.5,
                "fixed": True,
                "unit": None
            },
            "semi_major_axis": {
                "value": 16.61968275372717,
                "constraint": "16.515 / sin(radians(system@inclination))",
                "unit": "solRad"
            }
        },
        "r_squared": {
            "value": 0.9943566454789291,
            "unit": None
        }
    }

    task = LCBinaryAnalyticsTask(data=data, method='least_squares', expected_morphology="detached")
    task.set_result(result)
    task.plot.model()


if __name__ == '__main__':
    main()
