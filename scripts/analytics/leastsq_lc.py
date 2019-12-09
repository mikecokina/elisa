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
    n = len(lc["Generic.Bessell.V"])

    _max = np.max(list(lc.values()))
    bias = {"Generic.Bessell.B": np.random.uniform(0, _max * 0.004, n) * np.array([random_sign() for _ in range(n)]),
            "Generic.Bessell.V": np.random.uniform(0, _max * 0.004, n) * np.array([random_sign() for _ in range(n)]),
            "Generic.Bessell.R": np.random.uniform(0, _max * 0.004, n) * np.array([random_sign() for _ in range(n)])}
    lc = {comp: val + bias[comp] for comp, val in lc.items()}

    lc_initial = [
        {
            'value': 16.515,
            'param': 'semi_major_axis',
            'constraint': '16.515 / sin(radians({inclination}))'
        },
        {
            'value': 8307.0,
            'param': 'p__t_eff',
            'fixed': False,
            'min': 7500.0,
            'max': 8500.0
        },
        {
            'value': 6.5,
            'param': 'p__surface_potential',
            'fixed': False,
            'min': 3,
            'max': 10
        },
        {
            'value': 5500.0,
            'param': 's__t_eff',
            'fixed': False,
            'min': 4000.0,
            'max': 8000.0
        },
        {
            'value': 7,
            'param': 's__surface_potential',
            'fixed': False,
            'min': 4.0,
            'max': 10.0
        },
        {
            'value': 85.0,
            'param': 'inclination',
            'fixed': False,
            'min': 80,
            'max': 90
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
            'value': 0.0,
            'param': 'argument_of_periastron',
            'fixed': True
        },
        {
            'value': 0.5,
            'param': 'mass_ratio',
            'fixed': True
        },
        {
            'value': 0.0,
            'param': 'eccentricity',
            'fixed': True
        }
    ]

    result = binary_detached.fit(xs=phases, ys=lc, period=4.5, discretization=5.0, x0=lc_initial, yerrs=None)
    print(json.dumps(result, indent=4))


if __name__ == '__main__':
    main()
