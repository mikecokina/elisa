from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Any

import numpy as np

from elisa import settings
from elisa.logger import getPersistentLogger
from elisa.utc import UTC

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.types import Float

logger = getPersistentLogger("analytics.binary_fit.mixins")


class MCMCMixin:
    """Module for handling of the MCMC chain and the sampler."""

    @staticmethod
    def renormalize_flat_chain(
        flat_chain: NDArray[Float],
        all_lables: list[str],
        labels: list[str],
        normalization: dict[str, tuple[Float, Float]],
    ) -> NDArray[Float]:
        """Re-normalize values in chain stored within (0, 1) interval to their original values.

        Denormalizes chain values from normalized [0, 1] interval back to their original
        value ranges using the normalization boundaries for each parameter.

        :param flat_chain: Resulting flattened chain obtained from MCMC sampling.
        :type flat_chain: NDArray[Float]
        :param all_lables: Names of all variable model parameters.
        :type all_lables: list[str]
        :param labels: Names of variable model parameters desired in the output array.
        :type labels: list[str]
        :param normalization: Normalization bounds for variable model parameters.
        :type normalization: dict[str, tuple[Float, Float]]
        :return: Re-normalized flat chain.
        :rtype: NDArray[Float]
        """
        from elisa.analytics.params import parameters  # noqa: PLC0415

        retval = []
        for label in labels:
            idx = all_lables.index(label)
            retval.append(
                parameters.renormalize_value(
                    flat_chain[:, idx],
                    normalization[label][0],
                    normalization[label][1],
                ),
            )

        return np.column_stack(retval)

    @staticmethod
    def resolve_mcmc_result(
        flat_chain: NDArray[Float],
        fitable: dict[str, Any],
        normalization: dict[str, tuple[Float, Float]],
        percentiles: list[int] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Process flat chain from MCMC sampling and produce dictionary with results.

        Converts the flattened MCMC chain into a result dictionary with parameter values,
        confidence intervals, and metadata. Calculates percentiles for error estimation
        and rounds values to appropriate significant digits.

        :param flat_chain: Flattened MCMC chain from EnsembleSampler.get_chain(flat=True).
        :type flat_chain: NDArray[Float]
        :param fitable: Fitable parameters dictionary.
        :type fitable: dict[str, Any]
        :param normalization: Normalization map with min/max boundaries.
        :type normalization: dict[str, tuple[Float, Float]]
        :param percentiles: Percentiles for error estimation [left, centre, right].
            Defaults to [16, 50, 84].
        :type percentiles: list[int] | None
        :return: Dictionary with variable model parameters in flat format including
            values, confidence intervals, and metadata.
        :rtype: dict[str, dict[str, Any]]
        """
        from elisa.analytics.params import parameters  # noqa: PLC0415

        percentiles = [16, 50, 84] if percentiles is None else percentiles
        result: dict[str, dict[str, Any]] = {}
        for idx, key in enumerate(fitable):
            mcmc_result = np.percentile(flat_chain[:, idx], percentiles)
            vals = parameters.renormalize_value(
                mcmc_result,
                normalization[key][0],
                normalization[key][1],
            )

            # rounding up values to significant digits
            sigma = np.min(np.abs(np.array([vals[2] - vals[1], vals[1] - vals[0]])))
            prec = -int(np.log10(sigma)) + 1
            vals = np.round(vals, decimals=prec)

            result[key] = {
                "value": float(vals[1]),
                "confidence_interval": {
                    "min": float(min(vals)),
                    "max": float(max(vals)),
                },
                "fixed": False,
                "min": normalization[key][0],
                "max": normalization[key][1],
                "unit": fitable[key].to_dict()["unit"],
            }

        return result

    @staticmethod
    def save_flat_chain(
        flat_chain: NDArray[Float],
        fitable: dict[str, Any],
        norm: dict[str, tuple[Float, Float]],
        fit_id: str | None = None,
    ) -> str:
        """Store samples of the MCMC run to a JSON file.

        Saves the flattened MCMC chain along with parameter labels, normalization
        bounds, and fitable parameter metadata to a JSON file.

        :param flat_chain: Flattened array of parameter values in each MCMC step.
            Shape: (n_samples, n_parameters).
        :type flat_chain: NDArray[Float]
        :param fitable: Dictionary containing fitable parameters with metadata.
        :type fitable: dict[str, Any]
        :param norm: Normalization dictionary with min/max boundaries for each parameter.
        :type norm: dict[str, tuple[Float, Float]]
        :param fit_id: ID or location (ending with .json) identifying the fit file.
            If None, current datetime is used.
        :type fit_id: str | None
        :return: Path to the saved file.
        :rtype: str
        """
        home = Path(settings.HOME)
        if fit_id is not None:
            fit_path = Path(fit_id)
            if fit_path.parent.is_dir():
                fdir = fit_path.parent
                fname = fit_path.name if fit_path.suffix == ".json" else f"{fit_path.name}.json"
                home = Path()
            else:
                fdir = fit_path
                fname = f"{fit_id}.json" if not fit_id.endswith(".json") else fit_id
        else:
            now = datetime.now(tz=UTC)
            fdir = Path(now.strftime(settings.DATE_MASK))
            fname = f"{now.strftime(settings.DATETIME_MASK)}.json"

        fpath = home / fdir / fname
        fpath.parent.mkdir(parents=True, exist_ok=True)

        data: dict[str, Any] = {
            "flat_chain": flat_chain.tolist() if isinstance(flat_chain, np.ndarray) else flat_chain,
            "fitable_parameters": list(fitable.keys()),
            "normalization": norm,
            "fitable": {key: val.to_dict() for key, val in fitable.items()},
        }

        with fpath.open("w") as f:
            f.write(json.dumps(data, indent=4))

        logger.info("MCMC chain, variable's fitable and normalization constants were stored in: %s", fpath)
        return str(fpath)

    @staticmethod
    def load_flat_chain(fit_id: str) -> dict[str, Any]:
        """Load the result (flat chain) from the MCMC run.

        Loads MCMC sampling results from a JSON file, including the flattened chain,
        parameter labels, normalization bounds, and parameter metadata.

        :param fit_id: ID or location (ending with .json) identifying the fit file.
        :type fit_id: str
        :return: Dictionary containing flat_chain, fitable_parameters, normalization,
            and fitable metadata.
        :rtype: dict[str, Any]
        """
        fname = fit_id if str(fit_id).endswith(".json") else f"{fit_id}.json"

        # expected full path
        fpath = Path(fname)
        if fpath.is_file():
            filepath = fpath
        else:
            # expect timestamp default name
            fdir = fit_id[: len(settings.DATE_MASK) + 2]
            filepath = Path(settings.HOME) / fdir / fname
            if not filepath.is_file():
                # expected user defined fit_id
                filepath = Path(settings.HOME) / fit_id / fname

        with filepath.open() as f:
            return json.loads(f.read())

    @staticmethod
    def worker(
        sampler: Any,
        p0: NDArray[Float],
        nsteps: int,
        nsteps_burn_in: int,
        *,
        save: bool = False,
        fit_id: str | None = None,
        fitable: dict[str, Any] | None = None,
        normalization: dict[str, tuple[Float, Float]] | None = None,
        progress: bool = False,
    ) -> None:
        """Multiprocessor worker for MCMC sampling routine.

        Executes MCMC sampling with optional burn-in phase and periodic chain saving.
        Handles both single-process and multi-process scenarios.

        :param sampler: MCMC ensemble sampler from emcee.
        :type sampler: Any
        :param p0: Initial walker distribution of normalized parameters.
            Shape: (n_walkers, n_variables).
        :type p0: NDArray[Float]
        :param nsteps: Number of MCMC sampling iterations.
        :type nsteps: int
        :param nsteps_burn_in: Initial iterations discarded for thermalization.
        :type nsteps_burn_in: int
        :param save: If True, the MCMC flat chain will be stored periodically.
        :type save: bool
        :param fit_id: ID or location (ending with .json) identifying the fit file.
            If None, current datetime is used.
        :type fit_id: str | None
        :param fitable: Dictionary containing fitable parameters with metadata.
        :type fitable: dict[str, Any] | None
        :param normalization: Normalization boundaries dictionary.
        :type normalization: dict[str, tuple[Float, Float]] | None
        :param progress: Display the progress bar of the sampling.
        :type progress: bool
        """
        logger.info("running burn-in...")
        if nsteps_burn_in > 0:
            p0, _, _ = sampler.run_mcmc(p0, nsteps_burn_in, progress=progress, store=False)
        sampler.reset()
        logger.info("running production...")

        if save:
            t_between_dumps = time()
            for _ in sampler.sample(p0, iterations=nsteps, progress=progress):
                if time() - t_between_dumps > settings.MCMC_SAVE_INTERVAL:
                    MCMCMixin.save_flat_chain(
                        sampler.get_chain(flat=True),
                        fitable=fitable,
                        norm=normalization,
                        fit_id=fit_id,
                    )
                    t_between_dumps = time()
        else:
            _, _, _ = sampler.run_mcmc(p0, nsteps, progress=progress)

    @staticmethod
    def mcmc_nwalkers_vs_ndim_validity_check(nwalkers: int, ndim: int) -> None:
        """Validate number of MCMC walkers satisfies `nwalkers >= ndim * 2`.

        Ensures the ensemble sampler has sufficient walkers relative to the number
        of free parameters as required by emcee.

        :param nwalkers: The number of walkers in the ensemble.
        :type nwalkers: int
        :param ndim: Number of free variables (dimensions).
        :type ndim: int
        :raises RuntimeError: When condition `nwalkers < ndim * 2` is not satisfied.
        """
        if nwalkers < ndim * 2:
            error_msg = (
                f"Fit cannot be executed with fewer walkers ({nwalkers}) "
                f"than twice the number of dimensions ({ndim})"
            )
            raise RuntimeError(error_msg)
