import os
import json
import numpy as np
import os.path as op

from datetime import datetime
from typing import Dict

from .. params import parameters
from ... import settings
from ... logger import getPersistentLogger


logger = getPersistentLogger('analytics.binary_fit.mixins')


class MCMCMixin(object):
    @staticmethod
    def renormalize_flat_chain(flat_chain, labels, normalization):
        """
        Renormalize values in chain if renormalization Dict is supplied.
        """
        return np.array([[parameters.renormalize_value(val, normalization[key][0], normalization[key][1])
                          for key, val in zip(labels, sample)] for sample in flat_chain])

    @staticmethod
    def resolve_mcmc_result(flat_chain, fitable, normalization, percentiles=None):
        """
        Function process flat chain (output from McMcFit._fit.get_chain(flat=True)) and produces dictionary with
        results.

        :param normalization: Dict; normalisation map
        :param flat_chain: emcee.ensemble.EnsembleSampler.get_chain(flat=True);
        :param fitable: Dict; fitable parameters
        :param percentiles: List; [percentile for left side error estimation, percentile of the centre,
                                   percentile for right side error estimation]
        :return: Dict;
        """
        percentiles = [16, 50, 84] if percentiles is None else percentiles
        result = dict()
        for idx, key in enumerate(fitable):
            mcmc_result = np.percentile(flat_chain[:, idx], percentiles)
            vals = parameters.renormalize_value(mcmc_result, normalization[key][0], normalization[key][1])

            # rounding up values to significant digits
            sigma = np.min(np.abs([vals[2] - vals[1], vals[1] - vals[0]]))
            prec = - int(np.log10(sigma)) + 1
            vals = np.round(vals, decimals=prec)

            result[key] = {
                "value": vals[1],
                "confidence_interval": {
                    "min": min(vals),
                    "max": max(vals)},
                "fixed": False,
                "min": normalization[key][0],
                "max": normalization[key][1],
                "unit": fitable[key].to_dict()['unit']
            }

        return result

    @staticmethod
    def save_flat_chain(flat_chain: np.array, fitable: Dict, norm: Dict, fit_id=None):
        """
        Store state of mcmc run.

        :param fit_id: str; fit_id of stored chain
        :param flat_chain: numpy.array; flatted array of parameters values in each mcmc step::

            [[param0_0, param1_0, ..., paramk_0],
            [param0_1, param1_1, ..., paramk_1], ...
            [param0_b, param1_n, ..., paramk_n]]

        :param norm: Dict; normalization dict
        :param fitable: Union[List, numpy.array]; labels of parameters in order of params in `flat_chain`
        """

        home = settings.HOME
        if fit_id is not None:
            if op.isdir(op.dirname(fit_id)):
                fdir = op.dirname(fit_id)
                fname = op.basename(fit_id)
                home = ''
            else:
                fdir = str(fit_id)
                fname = f'{str(fit_id)}.json'
        else:
            now = datetime.now()
            fdir = now.strftime(settings.DATE_MASK)
            fname = f'{now.strftime(settings.DATETIME_MASK)}.json'

        fpath = op.join(home, fdir, fname)
        os.makedirs(op.join(settings.HOME, fdir), exist_ok=True)
        data = {
            "flat_chain": flat_chain.tolist() if isinstance(flat_chain, np.ndarray) else flat_chain,
            "fitable_parameters": list(fitable.keys()),
            "normalization": norm,
            "fitable": {key: val.to_dict() for key, val in fitable.items()}
        }
        with open(fpath, "w") as f:
            f.write(json.dumps(data, indent=4))
        logger.info(f'MCMC chain, variable`s fitable and normalization constants were stored in: {fpath}')
        return fpath

    @staticmethod
    def load_flat_chain(fit_id):
        """
        Restore stored state from mcmc run.

        :param fit_id: str; base fit_id of stored state
        :return: Dict;
        """

        fname = fit_id if str(fit_id).endswith('.json') else f'{fit_id}.json'

        # expected full path
        if op.isfile(fname):
            fpath = fname
        else:
            # expect timestamp default name
            fdir = fit_id[:len(settings.DATE_MASK) + 2]
            fpath = op.join(settings.HOME, fdir, fname)
            if not os.path.isfile(fpath):
                # expected user defined fit_id
                fpath = op.join(settings.HOME, fit_id, fname)

        with open(fpath, "r") as f:
            return json.loads(f.read())

    @staticmethod
    def worker(sampler, p0, nsteps, nsteps_burn_in, progress=False):
        logger.info("running burn-in...")
        p0, _, _ = sampler.run_mcmc(p0, nsteps_burn_in, progress=progress) if nsteps_burn_in > 0 else p0, None, None
        sampler.reset()
        logger.info("running production...")
        _, _, _ = sampler.run_mcmc(p0, nsteps, progress=progress)

    @staticmethod
    def mcmc_nwalkers_vs_ndim_validity_check(nwalkers, ndim):
        """
        Validate mcmc number of walkers and number of vector dimension.
        Has to be satisfied `nwalkers < ndim * 2`.

        :param nwalkers:
        :param ndim:
        :raise: RuntimeError; when condition `nwalkers < ndim * 2` is not satisfied
        """
        if nwalkers < ndim * 2:
            msg = f'Fit cannot be executed with fewer walkers ({nwalkers}) than twice the number of dimensions ({ndim})'
            raise RuntimeError(msg)
