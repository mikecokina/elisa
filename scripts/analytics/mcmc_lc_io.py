import json
import os.path as op

import numpy as np
from astropy import units as au

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
        "x_unit": au.dimensionless_unscaled,
        "y_unit": au.dimensionless_unscaled,
        "passband": passband
    }) for passband in lc}

    result = {
        "primary": {
            "t_eff": {
                "value": 8230.84100260351,
                "confidence_interval": {
                    "min": 7926.453297478265,
                    "max": 8591.643787140914
                },
                "fixed": False,
                "min": 7800.0,
                "max": 8800.0,
                "unit": "K"
            },
            "surface_potential": {
                "value": 3.9337264746233775,
                "confidence_interval": {
                    "min": 3.697563664749108,
                    "max": 4.199838559400843
                },
                "fixed": False,
                "min": 3.0,
                "max": 5.0,
                "unit": None
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
                "value": 5983.733468261532,
                "confidence_interval": {
                    "min": 5393.912953493938,
                    "max": 6474.502604079509
                },
                "fixed": False,
                "min": 4000.0,
                "max": 7000.0,
                "unit": "K"
            },
            "surface_potential": {
                "value": 6.112664406249967,
                "confidence_interval": {
                    "min": 5.728751677639996,
                    "max": 6.477440272860319
                },
                "fixed": False,
                "min": 5.0,
                "max": 7.0,
                "unit": None
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
                "value": 85.33146246324937,
                "confidence_interval": {
                    "min": 82.21719586922225,
                    "max": 88.54153499552233
                },
                "fixed": False,
                "min": 80.0,
                "max": 90.0,
                "unit": "deg"
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
                "value": 16.569975351859675,
                "constraint": "16.515 / sin(radians(system@inclination))",
                "unit": "solRad"
            }
        }
    }
    task = LCBinaryAnalyticsTask(data=data, method='mcmc')
    task.set_result(result)
    task.load_chain("mcmc_lc_fit")
    # task.plot.model()
    # task.plot.corner(truths=True)
    # task.plot.traces()
    # task.plot.autocorrelation()
    task.result_summary()


if __name__ == '__main__':
    main()
