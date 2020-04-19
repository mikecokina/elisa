import numpy as np
from matplotlib import pyplot as plt

from elisa.binary_system.system import BinarySystem
from elisa.observer.observer import Observer

PHASES = np.linspace(-0.6, 0.6, endpoint=True)
SYSTEM_BLUEPRINT = {
    "system": {
        "inclination": 90.0,
        "period": 0.5,
        "eccentricity": 0.0,
        "argument_of_periastron": 90.0,
        "gamma": 0.0
    },
    "primary": {
        "mass": 2.0,
        "surface_potential": 2.80,
        "t_eff": 6000.0,
        "gravity_darkening": 0.09,
        "synchronicity": 1.0,
        "albedo": 0.5,
        "metallicity": 0.0,
        "spots": list(),
        "discretization_factor": 2
    },
    "secondary": {
        "mass": 1.0,
        "surface_potential": 2.80,
        "t_eff": 6000,
        "gravity_darkening": 0.09,
        "synchronicity": 1.0,
        "albedo": 0.5,
        "metallicity": 0.0
    }
}

SPOTS_DEFINITION = [
    [
        {
            "longitude": 315.0,
            "latitude": 45.0,
            "angular_radius": 10,
            "temperature_factor": 0.85
        },
        {
            "longitude": 315.0,
            "latitude": 45.0,
            "angular_radius": 5,
            "temperature_factor": 0.7
        }
    ],
    [
        {
            "longitude": 315.0,
            "latitude": 45.0,
            "angular_radius": 15,
            "temperature_factor": 0.85
        },
        {
            "longitude": 315.0,
            "latitude": 45.0,
            "angular_radius": 10,
            "temperature_factor": 0.7
        }
    ],
    [
        {
            "longitude": 315.0,
            "latitude": 45.0,
            "angular_radius": 20,
            "temperature_factor": 0.85
        },
        {
            "longitude": 315.0,
            "latitude": 45.0,
            "angular_radius": 15,
            "temperature_factor": 0.7
        }
    ]
]


def generate_system():
    result = list()
    for _def in SPOTS_DEFINITION:
        _result = list()
        for i in range(1, 3):
            _spots = _def[:i]

            _params = SYSTEM_BLUEPRINT.copy()
            _params["primary"]["spots"] = _spots

            system: BinarySystem = BinarySystem.from_json(data=_params, _verify=False, _kind_of="std")
            observer = Observer(passband="Generic.Bessell.V", system=system)
            # system.plot.surface(edges=True, colormap="temperature", azimuth=315, inclination=35)

            lc = observer.observe.lc(phases=PHASES)

            _result.append(lc)
        result.append(_result)

    for res in result:
        _max = np.array([res[0][1]["Generic.Bessell.V"], res[1][1]["Generic.Bessell.V"]]).max()
        diff = (np.array(res[0][1]["Generic.Bessell.V"]) / _max) - (np.array(res[1][1]["Generic.Bessell.V"]) / _max)

        fig, axs = plt.subplots(2, sharex=True, sharey=False)

        axs[0].plot(PHASES, (np.array(res[0][1]["Generic.Bessell.V"]) / _max), c="blue")
        axs[0].plot(PHASES, (np.array(res[1][1]["Generic.Bessell.V"]) / _max), c="red")
        axs[1].plot(PHASES, diff)

        axs[0].set(ylabel='flux')
        axs[1].set(xlabel='phase')
        axs[1].set(ylabel=r'$\Delta$ $flux_{simple} - flux_{composed}$')

        axs[0].grid()
        axs[1].grid()

        # axs[1].set_position([0.125, 0.379, 0.775, 0.15])
        # plt.legend(loc='upper left')

    plt.show()
    # print()


# system.plot.surface(edges=True, colormap="temperature", azimuth=45, inclination=35)


def main():
    generate_system()


if __name__ == "__main__":
    main()
