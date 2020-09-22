import json
import os.path as op
import tempfile
import numpy as np
import os

from time import time
from elisa.binary_system import system
from elisa.conf import settings
from elisa.observer import observer
from elisa.logger import getLogger
from matplotlib import pyplot as plt
from importlib import reload

from elisa.utils import is_empty

logger = getLogger('benchmark')


STYPE_TO_FILENAME = {
    "detached.ecc.sync": "detached.ecc.sync.json",
    "detached.circ.sync": "detached.circ.sync.json",
    "detached.circ.async": "detached.circ.async.json"
}

DATA = op.join(op.abspath(op.dirname(__file__)), "data")


def get_params(stype):
    filename = STYPE_TO_FILENAME[stype]
    fpath = op.join(DATA, filename)
    with open(fpath, "r") as f:
        return json.loads(f.read())


def store_result(filename, data):
    tempdir = tempfile.gettempdir()
    fpath = op.join(tempdir, filename)
    with open(fpath, "w") as f:
        f.write(json.dumps(data, indent=4))
    return fpath


class BenchMark(object):
    def __init__(self, stype, n_steps=5, n_each=5, n_from=10, n_to=200, multiprocess=True):
        self.stype = stype
        self.n_steps = n_steps
        self.n_from = n_from
        self.n_to = n_to
        self.n_each = n_each
        self.mp_result = {"cores": int(os.cpu_count()), "n_phases": [], "elapsed_time": []}
        self.sc_result = {"cores": 1, "n_phases": [], "elapsed_time": []}

        settings.POINTS_ON_ECC_ORBIT = -1
        settings.MAX_RELATIVE_D_R_POINT = 0.0

        self._multiprocess = True
        setattr(self, "multiprocess", multiprocess)

    @property
    def multiprocess(self):
        return self._multiprocess

    @multiprocess.setter
    def multiprocess(self, value):
        self._multiprocess = value
        settings.NUMBER_OF_PROCESSES = int(os.cpu_count()) if self.multiprocess else 1
        reload(system)
        reload(observer)

    def eval(self, store=False):
        logger.info("starting benchmark evaluation")
        data = get_params(stype=self.stype)
        for n_run in range(self.n_from, self.n_to, self.n_each):
            if n_run % 5 == 0:
                logger.info(f'evaluating run for {n_run} phases')

            inter_run = []
            phases = np.linspace(-0.6, 0.6, n_run)

            # initialize objects
            binary = system.BinarySystem.from_json(data)
            o = observer.Observer(passband='Generic.Bessell.V', system=binary)

            for _ in range(0, self.n_steps):
                start_time = time()
                o.observe.lc(phases=phases)
                inter_run.append(time() - start_time)

            if self.multiprocess:
                self.mp_result["n_phases"].append(n_run)
                self.mp_result["elapsed_time"].append(np.mean(inter_run))
            else:
                self.sc_result["n_phases"].append(n_run)
                self.sc_result["elapsed_time"].append(np.mean(inter_run))

        logger.info("benchmark evaluation finished")

        if store:
            mp_in_name = {True: "multiprocess", False: "singleprocess"}
            filename = f'{mp_in_name[self.multiprocess]}.{self.stype}.json'
            result = self.mp_result if self.multiprocess else self.sc_result
            stored_in = store_result(filename, result)
            logger.info(f"result stored in {stored_in}")

    def plot(self):
        plt.figure(figsize=(8, 6))
        if not is_empty(self.mp_result["n_phases"]):
            plt.plot(self.mp_result["n_phases"], np.round(self.mp_result["elapsed_time"], 2), label=f"multiprocessing")
        if not is_empty(self.sc_result["n_phases"]):
            plt.plot(self.sc_result["n_phases"], np.round(self.sc_result["elapsed_time"], 2), label=f"singlecore")

        plt.legend()
        plt.xlabel('n_phases [-]')
        plt.ylabel('elapsed_time [s]')
        plt.show()


def main():
    bm = BenchMark('detached.circ.sync', n_steps=10, n_from=10, n_to=400)
    bm.eval(store=True)
    bm.multiprocess = False
    bm.eval(store=True)


if __name__ == "__main__":
    main()
