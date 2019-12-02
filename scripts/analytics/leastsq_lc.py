import json
import numpy as np
import os.path as op

from elisa.analytics.binary.least_squares import binary_detached
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
    u = np.random.uniform
    n = len(lc["Generic.Bessell.B"])

    _max = np.max(list(lc.values()))
    bias = {"Generic.Bessell.B": np.random.uniform(0, _max * 0.008, n) * np.array([random_sign() for _ in range(n)]),
            "Generic.Bessell.V": np.random.uniform(0, _max * 0.008, n) * np.array([random_sign() for _ in range(n)]),
            "Generic.Bessell.R": np.random.uniform(0, _max * 0.008, n) * np.array([random_sign() for _ in range(n)])}
    lc = {comp: val + bias[comp] for comp, val in lc.items()}

    lc_initial = [
            {
                'value': 0.5,
                'param': 'mass_ratio',
                'fixed': True
            },
            {
                'value': 16.54321389,
                'param': 'semi_major_axis',
                'fixed': True
            },
            {
                'value': 4800.0,
                'param': 'p__t_eff',
                'fixed': True
            },
            {
                'value': 5.0,
                'param': 'p__surface_potential',
                'fixed': True
            },
            {
                'value': 6700.0,
                'param': 's__t_eff',
                'fixed': True
            },
            {
                'value': 6.0,
                'param': 's__surface_potential',
                'fixed': False,
                'min': 4.0,
                'max': 10.0
            },
            {
                'value': 0.32,
                'param': 'p__gravity_darkening',
                'fixed': True
            },
            {
                'value': 0.32,
                'param': 's__gravity_darkening',
                'fixed': True
            },
            {
                'value': 0.6,
                'param': 'p__albedo',
                'fixed': True
            },
            {
                'value': 0.6,
                'param': 's__albedo',
                'fixed': True
            },
            {
                'value': 85.0,
                'param': 'inclination',
                'fixed': True
            }
        ]

    result = binary_detached.fit(xs=phases, ys=lc, period=4.5, discretization=5.0, x0=lc_initial, yerrs=None)
    print(json.dumps(result, indent=4))


if __name__ == '__main__':
    main()
