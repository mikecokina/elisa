import numpy as np
from elisa.analytics.binary.least_squares import central_rv

np.random.seed(1)


def main():
    phases = np.arange(-0.6, 0.62, 0.02)
    rv = {
        'primary': np.array([]),
        'secondary': np.array([])
    }

    # _max = np.max(list(rv.values()))
    # bias = {"primary": np.random.uniform(0, _max * 0.05, len(rv["primary"]))
    #                    * np.array([random_sign() for _ in range(len(rv["primary"]))]),
    #         "secondary": np.random.uniform(0, _max * 0.05, len(rv["primary"]))
    #                      * np.array([random_sign() for _ in range(len(rv["primary"]))])}
    # rv = {comp: val + bias[comp] for comp, val in rv.items()}

    rv_initial = [
        {
            'value': 0.2,
            'param': 'eccentricity',
            'fixed': False,
            'min': 0.0,
            'max': 0.5

        },
        {
            'value': 10.0,
            'param': 'asini',
            'fixed': False,
            'min': 1.0,
            'max': 20.0

        },
        {
            'value': 1.0,
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
            'value': 20000.0,
            'param': 'gamma',
            'fixed': True
        }
    ]

    rv_initial = [
        {
            'value': 0.0,
            'param': 'eccentricity',
            'fixed': True
        },
        {
            'value': 85.0,
            'param': 'inclination',
            'fixed': True
        },
        {
            'value': 2.0,
            'param': 'p__mass',
            'fixed': False,
            'min': 1.0,
            'max': 20.0

        },
        {
            'value': 1.0,
            'param': 's__mass',
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
            'value': 20000.0,
            'param': 'gamma',
            'fixed': False
        }
    ]

    result = central_rv.fit(xs=phases, ys=rv, period=0.6, x0=rv_initial, yerrs=None)



if __name__ == '__main__':
    main()