import json
import numpy as np
import os.path as op

from elisa.analytics.binary.mcmc import central_rv
from elisa.binary_system import t_layer
from elisa.conf.config import BINARY_COUNTERPARTS
from elisa.utils import random_sign


np.random.seed(1)
DATA = op.join(op.abspath(op.dirname(__file__)), "data")


def get_rv():
    fpath = op.join(DATA, "rv.json")
    with open(fpath, "r") as f:
        return json.loads(f.read())


def main():
    period, t0, phases = 4.5, 12.0, np.arange(-0.6, 0.62, 0.02)
    jd = t_layer.phase_to_jd(t0, period, phases)
    xs = {comp: jd for comp in BINARY_COUNTERPARTS}

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
            'value': 4.4,
            'param': 'period',
            'fixed': False,
            'min': 4.4,
            'max': 4.6
        },
        {
            'value': 11.1,
            'param': 'primary_minimum_time',
            'fixed': False,
            'min': 11.1,
            'max': 12.1
        }
    ]

    central_rv.fit(xs=xs, ys=rv, x0=rv_initial, nwalkers=20, nsteps=10000, nsteps_burn_in=1000, yerrs=None)

    result = central_rv.restore_flat_chain(central_rv.last_fname)
    central_rv.plot.corner(result['flat_chain'], result['labels'], renorm=result['normalization'])


if __name__ == '__main__':
    main()
