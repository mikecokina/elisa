import json
from elisa.binary_system import system
from elisa.observer.observer import Observer
import matplotlib.pyplot as plt
import numpy as np


def get_params(filename):
    with open(filename, "r") as f:
        return json.loads(f.read())


data = get_params('data/wide_binary.json')
surface_discredizations = [10, 5, 3, 2, 1]

curves = [None for _ in surface_discredizations]
phases = None
for ii, alpha in enumerate(surface_discredizations):
    data['primary']['discretization_factor'] = alpha
    binary = system.BinarySystem.from_json(data)

    o = Observer(passband=[  # defining passbands at which calculate a light curve
        # 'Generic.Bessell.U',
        # 'Generic.Bessell.B',
        'Generic.Bessell.V',
        # 'Generic.Bessell.R',
        # 'Generic.Bessell.I',
    ],
        system=binary)  # specifying the binary system to use in light curve synthesis
    phases, curves[ii] = o.lc(
        from_phase=-0.5,
        to_phase=0.5,
        phase_step=0.005,
        # phase_step=0.01,
        # normalize=True,
    )
    curves[ii] = curves[ii]['Generic.Bessell.V']

    y_data = curves[ii] / np.mean(curves[ii])
    mean = np.mean(abs(y_data-1))
    std = np.std(y_data, ddof=1)
    print(f'std estim: {std}')

    in_1_sigma = abs(y_data - 1) <= mean + std
    in_2_sigma = abs(y_data - 1) <= mean + (2.0 * std)
    in_3_sigma = abs(y_data - 1) <= mean + (3.0 * std)

    # print(f'sigma: {in_1_sigma}, 2sigma: {in_2_sigma}, 3sigma: {in_3_sigma}')
    print(f'sigma: {np.sum(in_1_sigma) / len(in_1_sigma)}')
    print(f'factor: {alpha}, mean noise: {mean}, low: {min(y_data - 1)}, high: {max(y_data - 1)}')
    print('--------------------------------------------------------------------')
    plt.plot(phases, y_data, label='factor: {0}, mean noise: {1:.2E}'.format(alpha, mean))

plt.xlabel('Phase')
plt.ylabel('Normalized flux')
plt.legend()
plt.show()
