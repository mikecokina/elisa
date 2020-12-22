import json
import os.path as op

import numpy as np

from elisa import units, settings
from elisa.analytics import LCBinaryAnalyticsTask
from elisa.analytics import LCData
from elisa.binary_system import t_layer

passbands = ["Generic.Bessell.B", "Generic.Bessell.V", "Generic.Bessell.R", "Generic.Bessell.I"]
np.random.seed(1)
DATA = op.join(op.abspath(op.dirname(__file__)))

P = 0.37491730
T0 = 52500.2071
t0 = 2452500.2071
do_plot = False


def mag_to_flx(ys):
    m2, f2 = 0.0, 10e10
    return {band: f2 * np.power(10, ((m2 - ys[band]) / 2.5)) for band in passbands}


def get_lc():
    fpath = op.join(DATA, "lc.json")
    with open(fpath, "r") as f:
        return json.loads(f.read())


def phase_filter(phases, threshold=0.05):
    phase_mask = {band: [True] for band in passbands}
    for band in passbands:
        xs = phases[band]
        xs_roll = np.roll(xs, shift=1)
        diff = np.abs(xs - xs_roll)

        cum_sum = 0
        for _diff in diff[1:]:
            cum_sum += _diff
            if cum_sum >= threshold:
                cum_sum = 0
                phase_mask[band].append(True)
            else:
                phase_mask[band].append(False)
    return phase_mask


def main():
    lc = get_lc()

    xs = {band: t_layer.jd_to_phase(t0, P, np.array(lc[band]["jd"])) for band in passbands}
    xs_argsort = {band: np.argsort(xs[band]) for band in passbands}
    xs = {band: xs[band][xs_argsort[band]] for band in passbands}

    ys_mag = {band: np.array(lc[band]["mag"])[xs_argsort[band]] for band in passbands}
    phase_mask = phase_filter(xs, threshold=0.01)

    ys_flx = mag_to_flx(ys_mag)
    ys_max = {band: max(np.array(ys_flx[band])) for band in passbands}
    ys = {band: ys_flx[band] / ys_max[band] for band in passbands}

    yerrs_mag = {band: np.array(np.array(lc[band]["err"]))[xs_argsort[band]] for band in passbands}

    ys_mag_max = {band: ys_mag[band] - yerrs_mag[band] for band in passbands}
    ys_mag_min = {band: ys_mag[band] + yerrs_mag[band] for band in passbands}

    ys_flx_max = mag_to_flx(ys_mag_max)
    ys_flx_min = mag_to_flx(ys_mag_min)

    yerrs = {band: (ys_flx_max[band] - ys_flx_min[band]) / (2. * ys_max[band])
             for band in passbands}

    xs = {band: xs[band][phase_mask[band]] for band in passbands}
    ys = {band: ys[band][phase_mask[band]] for band in passbands}
    yerrs = {band: yerrs[band][phase_mask[band]] for band in passbands}

    data = {passband: LCData(**{
        "x_data": xs[passband],
        "y_data": ys[passband],
        "y_err": yerrs[passband],
        "x_unit": units.dimensionless_unscaled,
        "y_unit": units.dimensionless_unscaled,
        "passband": passband
    }) for passband in lc}

    task = LCBinaryAnalyticsTask(data=data, method='least_squares', expected_morphology="over-contact")
    task.load_result(op.join(settings.HOME, "thesis_trra_synthetic.result.json"))
    task.plot.model()
    exit()
    task = LCBinaryAnalyticsTask(data=data, method='mcmc')
    task.load_result("/home/mike/.elisa/thesis_mcmc_rtmi.result.json")
    # task.load_chain("thesis_mcmc_rtmi", discard=30000)
    task.plot.model()
    task.plot.corner(truths=True)
    # task.plot.traces()
    # task.plot.autocorrelation()
    # task.result_summary()


if __name__ == '__main__':
    main()
