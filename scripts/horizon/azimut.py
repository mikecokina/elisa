"""
Horizon progress regards to discretization factor.
"""
import json
import os

import numpy as np
import pandas as pd
import os.path as op

from elisa import BinarySystem
from elisa.analytics.tools import horizon
from elisa.const import FULL_ARC
from matplotlib import pyplot as plt, ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import Akima1DInterpolator


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

DISCRETIZATION_FACTORS = [3, 5, 7, 10]
PHASE = 0.5 * 6 / 360.0
PHASES = [0.0, 0.25, 0.3, 0.5]


def multiple_main():
    axis_font = {'size': '13'}
    colors = ["red", "blue", "green", "orange"]
    ylims = 0.0012

    data = {}
    data_path = "phase.json"
    if op.isfile(data_path):
        os.remove(data_path)

    # computational
    params = BINARY_DEFINITION.copy()
    params["primary"]["discretization_factor"] = 5
    binary = BinarySystem.from_json(params)

    figure, axes = plt.subplots(4, 1, figsize=(8, 5))
    for idx, phs in enumerate(PHASES):
        data[phs] = {}

        discrete_horizon, vertex_discrete_horizon = \
            horizon.get_discrete_horizon(binary=binary, phase=phs, polar=True)

        # edge based discrete horizon
        phi_argsort = np.argsort(discrete_horizon.T[1] % FULL_ARC)
        rs_d, phis_d = discrete_horizon[phi_argsort].T[0], discrete_horizon[phi_argsort].T[1] % FULL_ARC
        rs_d, phis_d = rs_d[:-1], phis_d[:-1]

        # analytic horizon
        binary = BinarySystem.from_json(params)
        analytic_horizon = horizon.get_analytics_horizon(binary=binary, phase=phs, tol=1e-3, polar=True,
                                                         phi_density=200, theta_density=20000)
        phi_argsort = np.argsort(analytic_horizon.T[1] % FULL_ARC)
        rs, phis = analytic_horizon[phi_argsort].T[0], analytic_horizon[phi_argsort].T[1] % FULL_ARC
        rs, phis = rs[:-1], phis[:-1]
        # drop repeating value
        remove = [True] + [True if phis[i] - phis[i + 1] else False for i in range(0, len(phis) - 1)]
        rs, phis = rs[remove], phis[remove]

        # interpolation
        akima = Akima1DInterpolator(phis, rs)

        # residuai
        residua = (rs_d - akima(phis_d)) / akima(phis_d)

        ax = axes[idx]
        ax.axhline(y=0.0, color='k', linestyle='--', linewidth=1)
        ax.plot(phis_d % FULL_ARC, residua, label=f"phase: {np.round(phs, 2)}" + r"$^\circ$", linewidth=1,
                c=colors[idx])
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.4f}"))

        ax.set_ylim(-ylims, ylims)
        if idx < len(axes) - 1:
            xticks = [""] * len(phis_d)
            ax.set_xticklabels(xticks)
            ax.set_xlabel("", **axis_font)

        data[phs]["x"] = np.array(phis_d).tolist()
        data[phs]["y"] = np.array(residua).tolist()

    with open(data_path, "a+") as f:
        f.write(json.dumps(data, indent=4))

    for idx, _ in enumerate(PHASES):
        ax = axes[idx]
        ax.legend(loc=2)
        ax.set_ylabel("", **axis_font)
    figure.text(0.02, 0.5, r"$(\varrho - \varrho_d) / \varrho$", va='center', rotation='vertical')
    figure.text(0.5, 0.04, r"$\theta$", ha='center')

    params = {'legend.fontsize': 13, 'legend.handlelength': 3}
    plt.rcParams.update(params)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)

    plt.subplots_adjust(wspace=0.0, hspace=0.0, top=0.98, right=0.98)
    plt.show()
    plt.cla()


if __name__ == '__main__':
    multiple_main()
