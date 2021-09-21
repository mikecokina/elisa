import os
import json
import numpy as np
import os.path as op
from time import time

from datetime import datetime
from typing import Dict

from .. params import parameters
from ... import settings
from ... logger import getPersistentLogger


logger = getPersistentLogger('analytics.binary_fit.mixins')


class MCMCMixin(object):
    """
    Module for handling of the MCMC chain and the sampler.
    """
    @staticmethod
    def renormalize_flat_chain(flat_chain, all_lables, labels, normalization):
        """
        Re-normalize values in chain stored within (0, 1) interval to their original values.

        :param flat_chain: numpy.array; resulting flatten chain obtained from MCMC sampling
        :param all_lables: List; names of all variable model parameters
        :param labels: List; names variable model parameters desired in the output array
        :param normalization: Dict[str, tuple]; normalization bounds for variable model parameters
        :return: numpy.array; re-normalized flat chain
        """
        retval = []
        for ii, label in enumerate(labels):
            idx = all_lables.index(label)
            retval.append(parameters.renormalize_value(flat_chain[:, idx],
                                                       normalization[label][0], normalization[label][1]))

        retval = np.column_stack(retval)
        return retval

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
        :return: Dict; JSON with variable model parameters in flat format
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
        Store samples of the MCMC run.

        :param fit_id: str; id or location (ending with .json) which identifies fit file (if not specified, current
                            datetime is used)
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
        Lead the result (flat chain) from the MCMC run.

        :param fit_id: str; id or location (ending with .json) which identifies fit file (if not specified, current
                            datetime is used)
        :return: Dict;  flat_chain, variable parameters, normalization bounds, JSON with results
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
    def worker(sampler, p0, nsteps, nsteps_burn_in,
               save=False, fit_id=None, fitable=None, normalization=None, progress=False):
        """
        Multiprocessor worker for MCMC sampling routine.

        :param sampler: emcee.EnsembleSampler;
        :param p0: numpy.array; (n_walkers x n_variables) distribution of normalized parameters in the initial step
        :param nsteps: int; number of MCMC sampling iterations
        :param nsteps_burn_in: int; initial discarded iterations of the MCMC corresponding to the expected
                                    thermalization portion of the chain
        :param save: bool; if true, the MCMC flat chain will be stored
        :param fit_id: str; id or location (ending with .json) which identifies fit file (if not specified, current
                            dateime is used)
        :param fitable: Dict; JSON containing variable model parameters
        :param normalization: Dict[str, tuple]; normalization boundaries
        :param progress: bool; display the progress bar of the sampling
        """
        logger.info("running burn-in...")
        p0, _, _ = sampler.run_mcmc(p0, nsteps_burn_in, progress=progress, store=False) \
                       if nsteps_burn_in > 0 else p0, None, None
        sampler.reset()
        logger.info("running production...")

        if save:
            t_between_dumps = time()
            for _ in sampler.sample(p0, iterations=nsteps, progress=progress):
                if time() - t_between_dumps > settings.MCMC_SAVE_INTERVAL:
                    MCMCMixin.save_flat_chain(sampler.get_chain(flat=True), fitable=fitable, norm=normalization,
                                              fit_id=fit_id)
                    t_between_dumps = time()
        else:
            _, _, _ = sampler.run_mcmc(p0, nsteps, progress=progress)

    @staticmethod
    def mcmc_nwalkers_vs_ndim_validity_check(nwalkers, ndim):
        """
        Validate number of MCMC walkers satisfies the condition `nwalkers < ndim * 2`, where ndim is a number of free
        parameters.

        :param nwalkers: int; The number of walkers in the ensemble. Minimum is 2 * number of free parameters.
        :param ndim: int; number of free variables
        :raise: RuntimeError; when condition `nwalkers < ndim * 2` is not satisfied
        """
        if nwalkers < ndim * 2:
            msg = f'Fit cannot be executed with fewer walkers ({nwalkers}) than twice the number of dimensions ({ndim})'
            raise RuntimeError(msg)
