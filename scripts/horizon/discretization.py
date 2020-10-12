"""
Horizon progress regards to discretization factor.
"""
import numpy as np

from elisa import BinarySystem
from elisa.analytics.tools import horizon
from elisa.const import FULL_ARC
from matplotlib import pyplot as plt


BINARY_DEFINITION = {
    "system": {
        "argument_of_periastron": 90.0,
        "gamma": 0.0,
        "period": 5.0,
        "eccentricity": 0.0,
        "inclination": 90.0,
        "primary_minimum_time": 0.0,
        "phase_shift": 0.0
    },
    "primary": {
        "mass": 3.0,
        "surface_potential": 4.2,
        "synchronicity": 1.0,
        "t_eff": 6000.0,
        "gravity_darkening": 0.09,
        "albedo": 0.5,
        "metallicity": 0.0,
        "discretization_factor": 5
    },
    "secondary": {
        "mass": 0.5,
        "surface_potential": 5.0,
        "synchronicity": 1.0,
        "t_eff": 5000.0,
        "gravity_darkening": 0.09,
        "albedo": 0.5,
        "metallicity": 0.0
    }
}

DISCRETIZATION_FACTORS = [2, 5, 7, 10, 12]
PHASE = 0.1


def main():
    for df in DISCRETIZATION_FACTORS[1:2]:
        params = BINARY_DEFINITION.copy()
        params["primary"]["discretization_factor"] = df
        binary = BinarySystem.from_json(BINARY_DEFINITION)

        discrete_horizon = horizon.get_discrete_horizon(binary=binary, phase=PHASE, polar=True)
        phi_argsort = np.argsort(discrete_horizon.T[1] % FULL_ARC)
        rs, phis = discrete_horizon[phi_argsort].T[0], discrete_horizon[phi_argsort].T[1] % FULL_ARC
        rs, phis = rs[:-1], phis[:-1]

        plt.scatter(phis % FULL_ARC, rs * 10, s=12, c="k", label="discrete")

        analytic_horizon = horizon.get_analytics_horizon(binary=binary, phase=PHASE, tol=1e-3, polar=True,
                                                         phi_density=200, theta_density=10000)
        phi_argsort = np.argsort(analytic_horizon.T[1] % FULL_ARC)
        rs, phis = analytic_horizon[phi_argsort].T[0], analytic_horizon[phi_argsort].T[1] % FULL_ARC
        rs, phis = rs[:-1], phis[:-1]

        plt.plot(phis % FULL_ARC, rs * 10, c="r", label="analytic")

        axis_font = {'size': '12'}
        plt.xlabel(r"$\theta$", **axis_font)
        plt.ylabel(r"$\varrho$", **axis_font)
        plt.legend()

        params = {'legend.fontsize': 11,
                  'legend.handlelength': 3}
        plt.rcParams.update(params)
        plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=10)
        plt.legend(loc=2)

        ax = plt.subplot()
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(11)

        plt.show()

        plt.show()
        plt.cla()




if __name__ == '__main__':
    main()
