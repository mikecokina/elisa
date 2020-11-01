"""
Horizon progress regards to discretization factor.
"""
import json
import os

import numpy as np
import os.path as op

from elisa import BinarySystem, Observer
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
        "discretization_factor": 10
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

INCLINATIONS = [90, 70, 50, 10]
PHASE = 0.1


def single_main():
    for i in INCLINATIONS[1:2]:
        # computational
        params = BINARY_DEFINITION.copy()
        params["system"]["inclination"] = 50
        binary = BinarySystem.from_json(params)

        # observer = Observer(["Generic.Bessell.V"], binary)
        # observer.observe.lc(phases=[0.1])

        # discrete horizon
        discrete_horizon, vertex_discrete_horizon = horizon.get_discrete_horizon(binary=binary, phase=PHASE, polar=True)

        # edge based discrete horizon
        phi_argsort = np.argsort(discrete_horizon.T[1] % FULL_ARC)
        rs_d, phis_d = discrete_horizon[phi_argsort].T[0], discrete_horizon[phi_argsort].T[1] % FULL_ARC
        rs_d, phis_d = rs_d[:-1], phis_d[:-1]

        # vertices based discrete horizon
        phi_argsort = np.argsort(vertex_discrete_horizon.T[1] % FULL_ARC)
        rs_vd, phis_vd = vertex_discrete_horizon[phi_argsort].T[0], vertex_discrete_horizon[phi_argsort].T[1] % FULL_ARC
        rs_vd, phis_vd = rs_vd[:-1], phis_vd[:-1]

        # plt.scatter(phis_d, rs_d)
        # plt.show()

        # analytic horizon
        analytic_horizon = horizon.get_analytics_horizon(binary=binary, phase=PHASE, tol=1e-3, polar=True,
                                                         phi_density=200, theta_density=20000)
        phi_argsort = np.argsort(analytic_horizon.T[1] % FULL_ARC)
        rs, phis = analytic_horizon[phi_argsort].T[0], analytic_horizon[phi_argsort].T[1] % FULL_ARC
        rs, phis = rs[:-1], phis[:-1]
        # drop repeating value
        remove = [True] + [True if phis[i] - phis[i+1] else False for i in range(0, len(phis)-1)]
        rs, phis = rs[remove], phis[remove]

        # interpolation
        akima = Akima1DInterpolator(phis, rs)

        # residua
        residua = (rs_d - akima(phis_d)) / akima(phis_d)

        # plot
        figure, ax1 = plt.subplots(1, 1, figsize=(8, 5))

        # discrete horizon plot
        ax1.scatter(phis_vd % FULL_ARC, rs_vd * 10, s=12, c="b")
        ax1.plot(phis_d % FULL_ARC, rs_d * 10, c="b", label="discrete", linewidth=1)

        # analytics horizon plot
        ax1.plot(phis % FULL_ARC, rs * 10, c="r", label="analytic")

        # here comes residua
        divider = make_axes_locatable(ax1)
        ax2 = divider.append_axes("bottom", size=1, pad=0.1)
        ax1.figure.add_axes(ax2)
        ax2.plot(phis_d % FULL_ARC, residua, c="g")
        ax1.legend(loc=2)

        # settings
        axis_font = {'size': '13'}
        ax1.set_ylabel(r"$\varrho$", **axis_font)
        ax2.set_ylabel(r"$(\varrho - \varrho_d) / \varrho$", **axis_font)
        ax2.set_ylim([-0.001, 0.001])

        ax2.set_xlabel(r"$\theta$", **axis_font)
        ax2.axhline(y=0.0, color='k', linestyle='--')
        params = {'legend.fontsize': 13, 'legend.handlelength': 3}
        plt.rcParams.update(params)
        plt.rc('xtick', labelsize=10)
        plt.rc('ytick', labelsize=10)

        for _ax in [ax1, ax2]:
            for label in (_ax.get_xticklabels() + _ax.get_yticklabels()):
                label.set_fontsize(11)

        plt.show()


def multiple_main():
    data = {}
    data_path = "inclination.json"
    if op.isfile(data_path):
        os.remove(data_path)

    # define plot
    figure, ax1 = plt.subplots(1, 1, figsize=(8, 5))

    # run eval
    for i in INCLINATIONS:
        data[i] = {}

        # computational
        params = BINARY_DEFINITION.copy()
        params["system"]["inclination"] = i
        binary = BinarySystem.from_json(params)

        discrete_horizon, vertex_discrete_horizon = horizon.get_discrete_horizon(binary=binary, phase=PHASE, polar=True)

        # edge based discrete horizon
        phi_argsort = np.argsort(discrete_horizon.T[1] % FULL_ARC)
        rs_d, phis_d = discrete_horizon[phi_argsort].T[0], discrete_horizon[phi_argsort].T[1] % FULL_ARC
        rs_d, phis_d = rs_d[:-1], phis_d[:-1]

        # vertices based discrete horizon
        phi_argsort = np.argsort(vertex_discrete_horizon.T[1] % FULL_ARC)
        rs_vd, phis_vd = vertex_discrete_horizon[phi_argsort].T[0], vertex_discrete_horizon[phi_argsort].T[1] % FULL_ARC
        rs_vd, phis_vd = rs_vd[:-1], phis_vd[:-1]

        # analytic horizon
        binary = BinarySystem.from_json(params)
        analytic_horizon = horizon.get_analytics_horizon(binary=binary, phase=PHASE, tol=1e-3, polar=True,
                                                         phi_density=200, theta_density=20000)
        phi_argsort = np.argsort(analytic_horizon.T[1] % FULL_ARC)
        rs, phis = analytic_horizon[phi_argsort].T[0], analytic_horizon[phi_argsort].T[1] % FULL_ARC
        rs, phis = rs[:-1], phis[:-1]
        # drop repeating value
        remove = [True] + [True if phis[i] - phis[i + 1] else False for i in range(0, len(phis) - 1)]
        rs, phis = rs[remove], phis[remove]

        # interpolation
        akima = Akima1DInterpolator(phis, rs)

        # residua
        residua = (rs_d - akima(phis_d)) / akima(phis_d)

        # plot
        ax1.axhline(y=0.0, color='k', linestyle='--', linewidth=1)
        ax1.plot(phis_d % FULL_ARC, residua, label=f"inclination: {i}" + r"$^\circ$", linewidth=1)
        ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.4f}"))

        data[i]["x"] = np.array(phis_d).tolist()
        data[i]["y"] = np.array(residua).tolist()

    with open(data_path, "a+") as f:
        f.write(json.dumps(data, indent=4))

    ax1.legend(loc=2)
    axis_font = {'size': '13'}
    ax1.set_ylabel(r"$(\varrho - \varrho_d) / \varrho$", **axis_font)
    ax1.set_xlabel(r"$\theta$", **axis_font)

    params = {'legend.fontsize': 13, 'legend.handlelength': 3}
    plt.rcParams.update(params)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)

    for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
        label.set_fontsize(11)

    plt.show()
    plt.cla()


if __name__ == '__main__':
    multiple_main()
