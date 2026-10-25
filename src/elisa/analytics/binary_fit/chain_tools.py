"""Chain boundary validation and diagnostics that preserve ensemble time axes."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
from emcee.autocorr import AutocorrError, function_1d, integrated_time

from elisa import settings

if TYPE_CHECKING:
    from emcee import EnsembleSampler
    from numpy.typing import NDArray

FLAT_NDIM = 2
ENSEMBLE_NDIM = 3
CHAIN_ORDER = "step,walker,parameter"


class DiscardUnit(str, Enum):
    """Select complete ensemble steps or individual flat rows for chain discard.

    String values remain compatible with existing chain-loading calls. Omitting
    the unit selects steps for structured chains and rows for legacy flat files.
    """

    STEPS = "steps"
    ROWS = "rows"

    def __str__(self) -> str:
        """Return the serialized discard unit for configuration and logging."""
        return self.value


def get_production_chain(sampler: EnsembleSampler | None) -> NDArray[np.float64]:
    """Retrieve the sampler's production chain, rejecting missing output.

    :param sampler: Completed sampler instance, or ``None`` before sampling.
    :returns: Structured production samples with step, walker and parameter axes.
    :raises ValueError: If the sampler or its production chain is unavailable.
    """
    chain = sampler.get_chain() if sampler is not None else None
    if chain is None:
        msg = "Sampler did not provide a production chain"
        raise ValueError(msg)
    return chain


def validate_flat_samples(data: dict[str, Any]) -> NDArray[np.float64]:
    """Validate samples and column labels at an export or load boundary."""
    flat = np.asarray(data["flat_chain"], dtype=float)
    if flat.ndim != FLAT_NDIM or flat.size == 0 or not np.all(np.isfinite(flat)):
        msg = "Chain samples must be a finite nonempty matrix"
        raise ValueError(msg)
    labels = data["fitable_parameters"]
    if not isinstance(labels, list) or len(labels) != flat.shape[1]:
        msg = "Chain parameter dimension does not match variable labels"
        raise ValueError(msg)
    if any(not isinstance(label, str) for label in labels) or len(set(labels)) != len(labels):
        msg = "Chain variable labels must be unique strings"
        raise ValueError(msg)
    return flat


def restore_temporal_chain(flat: NDArray[np.float64], data: dict[str, Any]) -> NDArray[np.float64] | None:
    """Restore axes only when both shape and explicit ordering metadata are valid."""
    if "chain_shape" not in data and "chain_order" not in data:
        return None
    shape = data.get("chain_shape")
    if not isinstance(shape, (list, tuple)) or len(shape) != ENSEMBLE_NDIM:
        msg = "Invalid temporal chain metadata: expected step, walker, parameter axes"
        raise ValueError(msg)
    if any(type(size) is not int or size <= 0 for size in shape):
        msg = "Invalid temporal chain metadata: dimensions must be positive integers"
        raise ValueError(msg)
    if data.get("chain_order") != CHAIN_ORDER or flat.shape != (shape[0] * shape[1], shape[2]):
        msg = "Invalid temporal chain metadata: shape or ordering does not match samples"
        raise ValueError(msg)
    return flat.reshape(tuple(shape))


def validate_discard(discard: int, discard_unit: DiscardUnit | str | None) -> DiscardUnit | None:
    """Validate discard arguments and normalize an explicit unit before slicing.

    :param discard: Nonnegative integer number of steps or rows to discard.
    :param discard_unit: Explicit unit, or ``None`` for automatic selection by the
        caller. Accepts ``DiscardUnit.STEPS``, ``DiscardUnit.ROWS`` and their string values.
    :type discard_unit: DiscardUnit | str | None
    :returns: Normalized enum member, or ``None`` if no unit was specified.
    """
    if isinstance(discard, (bool, np.bool_)) or not isinstance(discard, (int, np.integer)):
        msg = "discard must be a nonnegative integer"
        raise TypeError(msg)
    if discard < 0:
        msg = "discard must be a nonnegative integer"
        raise ValueError(msg)
    if discard_unit is None:
        return None
    try:
        return DiscardUnit(discard_unit)
    except (TypeError, ValueError) as exc:
        msg = "discard_unit must be DiscardUnit.STEPS or DiscardUnit.ROWS ('steps' or 'rows')"
        raise ValueError(msg) from exc


def prepare_chain_samples(
    data: dict[str, Any],
    discard: int = 0,
    discard_unit: DiscardUnit | str | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Discard steps for structured exports and rows for legacy flat exports.

    Explicit row cuts preserve temporal axes only at complete ensemble boundaries.
    No walker count or ordering is inferred for legacy samples.

    :param data: Flat samples, parameter labels and optional temporal metadata.
    :param discard: Nonnegative integer number of steps or rows to discard.
    :param discard_unit: ``DiscardUnit.STEPS`` requires temporal metadata;
        ``DiscardUnit.ROWS`` discards flat rows. ``None`` selects steps for structured
        chains and rows for legacy files. Strings "steps" and "rows" are accepted.
    :type discard_unit: DiscardUnit | str | None
    :returns: Remaining flat samples and the temporal chain, if preserved.
    """
    unit = validate_discard(discard, discard_unit)
    flat = validate_flat_samples(data)
    chain = restore_temporal_chain(flat, data)
    if unit is None:
        unit = DiscardUnit.STEPS if chain is not None else DiscardUnit.ROWS
    if unit is DiscardUnit.STEPS:
        return discard_steps(chain, discard)
    if discard >= flat.shape[0]:
        msg = "Discard leaves no samples"
        raise ValueError(msg)
    if chain is not None:
        nwalkers = chain.shape[1]
        chain = chain[discard // nwalkers:] if discard % nwalkers == 0 else None
    return flat[discard:], chain


def discard_steps(chain: NDArray[np.float64] | None, discard: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Slice complete ensemble iterations before flattening."""
    if chain is None:
        msg = "Step discard requires temporal chain metadata; use discard_unit=DiscardUnit.ROWS for legacy files"
        raise ValueError(msg)
    if discard >= chain.shape[0]:
        msg = "Discard leaves no samples"
        raise ValueError(msg)
    chain = chain[discard:]
    return chain.reshape(-1, chain.shape[-1]), chain


def ensemble_autocorrelation(chain: Any) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return walker-averaged ACFs and reliable integrated times, in step units.

    Missing temporal axes raise ValueError. Short chains and parameters with any
    stuck walker return NaN times. No effective sample size is inferred from flat
    samples. Diagnostics run only on request, outside posterior evaluations.
    """
    values = np.asarray(chain, dtype=float)
    if values.ndim != ENSEMBLE_NDIM or values.size == 0 or not np.all(np.isfinite(values)):
        msg = "Temporal autocorrelation requires a finite nonempty (steps, walkers, parameters) chain"
        raise ValueError(msg)
    functions = np.full((values.shape[0], values.shape[2]), np.nan)
    times = np.full(values.shape[2], np.nan)
    for idx in range(values.shape[2]):
        parameter = values[:, :, idx]
        if np.any(np.ptp(parameter, axis=0) == 0):
            continue
        functions[:, idx], times[idx] = parameter_autocorrelation(parameter)
    return functions, times


def parameter_autocorrelation(parameter: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
    """Use emcee's ensemble estimator and reject its insufficient-length estimates."""
    functions = np.mean([function_1d(walker) for walker in parameter.T], axis=0)
    try:
        # A 3D input keeps emcee 3.x's step/walker/parameter interpretation explicit.
        tau = float(
            integrated_time(
                parameter[:, :, None],
                tol=settings.MCMC_AUTOCORR_TOL,
            )[0],
        )
    except AutocorrError:
        return functions, np.nan
    return functions, tau if np.isfinite(tau) and tau > 0 else np.nan
