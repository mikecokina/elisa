import json
import numpy as np
import os.path as op

from elisa.analytics.binary.least_squares import central_rv
from elisa.conf.config import BINARY_COUNTERPARTS
from elisa.utils import random_sign

np.random.seed(1)
DATA = op.join(op.abspath(op.dirname(__file__)), "data")


def get_rv():
    fpath = op.join(DATA, "rv.json")
    with open(fpath, "r") as f:
        return json.loads(f.read())


def main():
    phases = np.arange(-0.6, 0.62, 0.02)
    xs = {comp: phases for comp in BINARY_COUNTERPARTS}
    rv = get_rv()
    u = np.random.uniform
    n = len(rv["primary"])

    _max = np.max(list(rv.values()))
    bias = {"primary": u(0, _max * 0.05, n) * np.array([random_sign() for _ in range(n)]),
            "secondary": u(0, _max * 0.05, n) * np.array([random_sign() for _ in range(n)])}
    rv = {comp: val + bias[comp] for comp, val in rv.items()}

    rv_initial = [
        {
            'value': 0.0,
            'param': 'eccentricity',
            'fixed': True
        },
        {
            'value': 15.0,
            'param': 'asini',
            'fixed': False,
            'min': 10.0,
            'max': 20.0

        },
        {
            'value': 3,
            'param': 'mass_ratio',
            'fixed': False,
            'min': 0,
            'max': 10
        },
        {
            'value': 0.0,
            'param': 'argument_of_periastron',
            'fixed': True
        },
        {
            'value': 30000.0,
            'param': 'gamma',
            'fixed': False,
            'min': 10000.0,
            'max': 50000.0
        },
        {
            'value': 4.5,
            'param': 'period',
            'fixed': True
        }
    ]

    result = central_rv.fit(xs=xs, ys=rv, x0=rv_initial, xtol=1e-10, yerrs=None)
    print(json.dumps(result, indent=4))


if __name__ == '__main__':
    main()