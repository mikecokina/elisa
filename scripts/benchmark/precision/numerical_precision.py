import json
from elisa.binary_system import system
from elisa.observer.observer import Observer
import matplotlib.pyplot as plt
import numpy as np


def get_params(filename):
    with open(filename, "r") as f:
        return json.loads(f.read())


data = get_params('data/wide_binary.json')
surface_discredizations = [10, 5, 3]

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
    print(f'factor: {alpha}, mean noise: {mean}')
    plt.plot(phases, y_data, label='factor: {0}, mean noise: {1:.2E}'.format(alpha, mean))

plt.xlabel('Phase')
plt.ylabel('Normalized flux')
plt.legend()
plt.show()
