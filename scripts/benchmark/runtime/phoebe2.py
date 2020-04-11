import json
import phoebe
import numpy as np
import time

logger = phoebe.logger(clevel = 'WARNING')

ntriangles = {
    "2": (20324, 7544),
    "5": (3376, 1132),
    "7": (1604, 564),
    "10": (848, 288),
}


def get_binary(eccentricity, triangles):
    b = phoebe.Bundle()
    b.add_component('star', component='primary')
    b.add_component('star', component='secondary')
    b.add_orbit('binary')
    b.set_hierarchy('orbit:binary(star:primary, star:secondary)')
    b.set_value('incl@orbit', 85)
    b.set_value('ecc@orbit', eccentricity)
    b.set_value('period@orbit', 3)
    b.set_value('sma@orbit', 12.628465471719581)
    b.set_value("per0@binary@component", 210.0)
    # b.set_value('mass@primary@component', 2.0)
    # b.set_value('mass@secondary@component', 1.0)
    b.set_value("q@binary@component", 0.5)
    b.set_value('teff@primary', 5500)
    b.set_value('teff@secondary', 5500)
    # b.set_value('atm@primary@phoebe01@compute', 'ck2004')
    # b.set_value('atm@secondary@phoebe01@compute', 'ck2004')
    b.set_value("pot@primary@component", 5.0)
    b.set_value("pot@secondary@component", 5.0)
    b.set_value('irrad_method@phoebe01@compute', 'wilson')
    b.set_value('ntriangles@primary@phoebe01@compute', triangles[0])
    b.set_value('ntriangles@secondary@phoebe01@compute', triangles[1])
    b.set_value("syncpar@primary@component", 1.0)
    b.set_value("syncpar@secondary@component", 1.0)


    return b


def run_observation(binary, phases):
    binary.add_dataset('lc', times=phases, dataset='lc01')
    binary.run_compute()


def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        f(*args, **kw)
        te = time.time()
        return te - ts
    return timed


def main():
    test_runs = 10
    result = {
        "phoebe2": {
            "circular": dict(),
            "eccentric": dict()
        }
    }

    for eccentricity in [0, 0.2]:
        orbit = "circular" if eccentricity == 0 else "eccentric"
        for alpha in [2, 5, 7, 10]:
            binary = get_binary(eccentricity, ntriangles[str(alpha)])
            for n_phases in [20, 50, 100, 150, 200, 250, 300]:
                phases = np.linspace(-0.6, 0.6, n_phases, endpoint=True)
                runtime = 0
                for _ in range(test_runs):
                    runtime += timeit(run_observation)(binary, phases)
                result["phoebe2"][orbit][str(n_phases)] = runtime / test_runs

    print(json.dumps(result, indent=4))
    # axs, artists = binary['lc@model'].plot(x='phases')


if __name__ == "__main__":
    main()
