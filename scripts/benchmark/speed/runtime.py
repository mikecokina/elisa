import json
from time import time
from elisa.binary_system import system
from elisa.observer.observer import Observer
from elisa.conf import settings
import matplotlib.pyplot as plt
import numpy as np
from elisa.binary_system.container import OrbitalPositionContainer


def get_params(filename):
    with open(filename, "r") as f:
        return json.loads(f.read())


def get_data(data, phs):
    binary = system.BinarySystem.from_json(data)
    o = Observer(passband=[  # defining passbands at which calculate a light curve
        # 'Generic.Bessell.U',
        'Generic.Bessell.B',
        'Generic.Bessell.V',
        'Generic.Bessell.R',
        # 'Generic.Bessell.I',
    ],
        system=binary)  # specifying the binary system to use in light curve synthesis

    _ = o.lc(
        phases=phs
    )

settings.LIMB_DARKENING_LAW = 'logarithmic'

data_circ = get_params('data/test_binary_circ.json')
data_ecc = get_params('data/test_binary_ecc.json')
outfl1 = 'benchmark_circ.dat'
outfl2 = 'benchmark_ecc.dat'

surface_discredizations = [10, 7, 5, 3]
n_phases = np.arange(50, 310, 50)
# n_phases = np.arange(50, 60, 50)
# N = 1
N = 5

#ecc = False
ecc = True

f = open(outfl1, 'w')
f.write(f'#Alpha    n_phases     time\n')

if ecc:
    g = open(outfl2, 'w')
    g.write(f'#Alpha    n_phases     time\n')

print(f'#Alpha    n_phases    time')
for ii, alpha in enumerate(surface_discredizations):
    data_circ['primary']['discretization_factor'] = alpha
    data_ecc['primary']['discretization_factor'] = alpha

    for jj, phs in enumerate(n_phases):
        start_time = time()
        for kk in range(N):
            phases = np.linspace(-0.5, 0.5, num=phs)
            get_data(data_circ, phases)
        elapsed = np.round((time() - start_time) / N, 2)
        f.write('{:>5} {:>10} {:>10}\n'.format(alpha, phs, elapsed))
        print('circular {:>10} {:>10} {:>10}'.format(alpha, phs, elapsed))

        # binary = system.BinarySystem.from_json(data_circ)
        # position = binary.calculate_orbital_motion(0.0)[0]
        # container = OrbitalPositionContainer.from_binary_system(binary, position)
        # container.build()
        #
        # print(f'{alpha} {container.primary.faces.shape[0]} {container.secondary.faces.shape[0]}')

        if not ecc:
            continue

        start_time = time()
        for kk in range(N):
            phases = np.linspace(-0.5, 0.5, num=phs)
            get_data(data_ecc, phases)

        elapsed = np.round((time() - start_time) / N, 2)
        g.write('{:>5} {:>10} {:>10}\n'.format(alpha, phs, elapsed))
        print('eccentric {:>10} {:>10} {:>10}'.format(alpha, phs, elapsed))


f.close()
if ecc:
    g.close()
# plt.xlabel('Phase')
# plt.ylabel('Normalized flux')
# plt.legend()
# plt.show()
