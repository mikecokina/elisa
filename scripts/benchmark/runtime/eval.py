import json
import os.path as op
import numpy as np

from matplotlib import pyplot as plt


DATA = op.abspath(op.join(op.dirname(__file__), "data"))


def get_json_content(filename):
    with open(op.join(DATA, filename), "r") as f:
        return json.loads(f.read())


def main():
    phoebe2 = get_json_content("phoebe2.json")
    elisa = get_json_content("elisa.json")
    result = dict(diff=dict())

    # for orbit in ["circular"]:
    for orbit in ["eccentric"]:
        result["diff"][orbit] = dict()
        for discretization in elisa["elisa"][orbit].keys():
            result["diff"][orbit][str(discretization)] = dict()
            for n_phases in elisa["elisa"][orbit][str(discretization)].keys():

                _elisa = elisa["elisa"][orbit][str(discretization)][str(n_phases)]
                _phoebe2 = phoebe2["phoebe2"][orbit][str(discretization)][str(n_phases)]

                result["diff"][orbit][str(discretization)][str(n_phases)] = _phoebe2 / _elisa

    # for orbit in ["circular"]:
    for orbit in ["eccentric"]:
        for discretization in result["diff"][orbit].keys():
            xs, ys = list(), list()
            for n_phases in result["diff"][orbit][str(discretization)].keys():
                xs.append(n_phases)
                ys.append(result["diff"][orbit][str(discretization)][str(n_phases)])

            plt.plot(np.array(xs, dtype=int), ys, label=f"Discretization: {discretization}")
    #
    plt.xlabel('n_phases')
    plt.ylabel(r'runtime ' + r'$[T_{phoebe2}\;/\;T_{elisa}]$')
    plt.legend(loc='upper left')

    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
