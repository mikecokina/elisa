from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from elisa.analytics.binary_fit.mixins import MCMCMixin
from elisa.analytics.binary_fit.shared import AbstractFit
from elisa.analytics.params import parameters
from elisa.analytics.params.parameters import BinaryInitialParameters, ParameterMeta

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from elisa.analytics.binary_fit.lc_fit import LCFitMCMC
    from elisa.analytics.binary_fit.rv_fit import RVFitMCMC

# Constants for boundary and error handling
BOUNDARY_LENGTH = 2
MIN_ERROR_THRESHOLD = 1e-15
LOG_ZERO_REPLACEMENT = 1e6


def filter_chain(
    mcmc_fit_cls: LCFitMCMC | RVFitMCMC,
    **boundaries: tuple[float, float],
) -> NDArray:
    """Filter MCMC chain down to given parameter intervals.

    This function is useful for filtering bimodal distributions of the MCMC chain.
    Parameters can be specified using flat format (using ``@`` notation for components).

    :param mcmc_fit_cls: MCMC fitting class instance (e.g., LCFitMCMC, RVFitMCMC).
    :type mcmc_fit_cls: LCFitMCMC | RVFitMCMC
    :param boundaries: Dictionary of parameter boundaries in flat format.
        Example: ``{'primary@te_ff': (5000, 6000), ...}``.
    :type boundaries: tuple[float, float]
    :returns: Filtered flat chain.
    :rtype: NDArray
    :raises TypeError: If boundary is not tuple, list, or ndarray.
    :raises TypeError: If boundary length is not 2.
    :raises NameError: If boundary key is not a valid model parameter.
    :raises ValueError: If boundaries yield an empty array.
    """
    for key, boundary in boundaries.items():
        if not isinstance(boundary, (tuple, list, np.ndarray)):
            error_msg = f"`{key}` boundary is not tuple or list."
            raise TypeError(error_msg)
        if len(boundary) != BOUNDARY_LENGTH:
            error_msg = f"`{key}` has incorrect length of {len(boundary)}."
            raise TypeError(error_msg)
        if key not in mcmc_fit_cls.variable_labels:
            error_msg = f"{key} is not valid model parameter."
            raise NameError(error_msg)

        column_idx = mcmc_fit_cls.variable_labels.index(key)
        column = mcmc_fit_cls.flat_chain[:, column_idx]

        condition_mask = np.logical_and(
            column
            > parameters.normalize_value(
                boundary[0],
                *mcmc_fit_cls.normalization[key],
            ),
            column
            < parameters.normalize_value(
                boundary[1],
                *mcmc_fit_cls.normalization[key],
            ),
        )
        if np.sum(condition_mask) == 0:
            error_msg = f"Boundaries for {key} yielded an empty array."
            raise ValueError(error_msg)
        mcmc_fit_cls.flat_chain = mcmc_fit_cls.flat_chain[condition_mask, :]

    fitted_params = {key: mcmc_fit_cls.flat_result[key] for key in mcmc_fit_cls.variable_labels}
    update_solution(mcmc_fit_cls, fitted_params, percentiles=None)

    return mcmc_fit_cls.flat_chain


def load_chain(
    mcmc_fit_cls: LCFitMCMC | RVFitMCMC,
    fit_id: str,
    discard: int = 0,
    percentiles: list[float] | None = None,
) -> tuple[NDArray, list[str], dict[str, tuple[float, float]]]:
    """Load MCMC chain along with auxiliary data from JSON file.

    Loads MCMC chain and related metadata from JSON file created after MCMC run.
    Automatically handles discarding of burn-in steps.

    :param mcmc_fit_cls: MCMC fitting class instance (e.g., LCFitMCMC, RVFitMCMC).
    :type mcmc_fit_cls: LCFitMCMC | RVFitMCMC
    :param fit_id: Chain identifier or filename containing the chain.
    :type fit_id: str
    :param discard: Number of steps to discard from the chain as burn-in (default: 0).
    :type discard: int
    :param percentiles: Percentile intervals used to generate confidence intervals,
        provided as [lower, centre, upper].
    :type percentiles: list[float] | None
    :returns: Tuple containing flattened MCMC chain, labels of variables in flat_chain
        columns, and dictionary of boundaries for reconstructing real values from
        normalized flat_chain array.
    :rtype: tuple[NDArray, list[str], dict[str, tuple[float, float]]]
    """
    data = MCMCMixin.load_flat_chain(fit_id=fit_id)

    mcmc_fit_cls.flat_chain = np.array(data["flat_chain"])[discard:, :]
    mcmc_fit_cls.variable_labels = data["fitable_parameters"]
    mcmc_fit_cls.normalization = data["normalization"]

    update_solution(mcmc_fit_cls, data["fitable"], percentiles)

    return (
        mcmc_fit_cls.flat_chain,
        mcmc_fit_cls.variable_labels,
        mcmc_fit_cls.normalization,
    )


def update_solution(
    mcmc_fit_cls: LCFitMCMC | RVFitMCMC,
    fitted_params: dict[str, Any],
    percentiles: list[float] | None,
) -> None:
    """Update fitting solutions based on MCMC chain distribution.

    Resolves MCMC results from the chain and applies constraint evaluations
    to produce final fitting results.

    :param mcmc_fit_cls: MCMC fitting class instance (e.g., LCFitMCMC, RVFitMCMC).
    :type mcmc_fit_cls: LCFitMCMC | RVFitMCMC
    :param fitted_params: Dictionary containing only variable part of flat_result.
    :type fitted_params: dict[str, Any]
    :param percentiles: Percentiles used for evaluation of confidence intervals.
    :type percentiles: list[float] | None
    :returns: None.
    :rtype: None
    :raises ValueError: If result is None and attempted to update solution.
    """
    fitable = {key: ParameterMeta(**val) for key, val in fitted_params.items()}

    # reproducing results from chain
    flat_result_update = MCMCMixin.resolve_mcmc_result(
        mcmc_fit_cls.flat_chain,
        fitable,
        mcmc_fit_cls.normalization,
        percentiles=percentiles,
    )

    if mcmc_fit_cls.result is not None:
        mcmc_fit_cls.flat_result.update(flat_result_update)

        # evaluating constraints
        fit_params = parameters.serialize_result(mcmc_fit_cls.flat_result)
        constrained = BinaryInitialParameters(**fit_params).get_constrained()
        mcmc_fit_cls.flat_result = AbstractFit.eval_constrained_results(
            mcmc_fit_cls.flat_result,
            constrained,
        )

        mcmc_fit_cls.result = parameters.serialize_result(mcmc_fit_cls.flat_result)

    else:
        error_msg = (
            "Load fit parameters before loading the chain. For example, call load_results() or similar method first."
        )
        raise ValueError(error_msg)


def write_ln(
    write_fn: Callable[[str], None],
    designation: str,
    value: float | str,
    bot: float | str,
    top: float | str,
    unit: str,
    status: str,
    line_sep: str,
    precision: int = 8,
) -> None:
    """Write formatted parameter line for output.

    Helper function that formats and writes a single parameter line,
    handling precision rounding for numeric values.

    :param write_fn: Function used to write into console or file.
    :type write_fn: Callable[[str], None]
    :param designation: Display name of the parameter.
    :type designation: str
    :param value: Numeric value or string representation of the parameter.
    :type value: float | str
    :param bot: Bottom boundary or error indicator.
    :type bot: float | str
    :param top: Top boundary or error indicator.
    :type top: float | str
    :param unit: Unit of measurement for the parameter.
    :type unit: str
    :param status: Status of the parameter (Fixed, Variable, Derived, etc.).
    :type status: str
    :param line_sep: Symbols to finish the line.
    :type line_sep: str
    :param precision: Number of significant figures for rounding (default: 8).
    :type precision: int
    :returns: None.
    :rtype: None
    """
    val = round(value, precision) if not isinstance(value, str) else value
    write_fn(
        f"{designation:<35} {val:>20}{bot:>20}{top:>20}{unit:>20}    {status:<50}{line_sep}",
    )


def write_param_ln(
    fit_params: dict[str, Any],
    param_id: str,
    designation: str,
    write_fn: Callable[[str], None],
    line_sep: str,
    precision: int = 8,
) -> None:
    """Write parameter line for fitted parameters with confidence intervals.

    Auxiliary function for fit_summary functions that produces formatted output
    for a single parameter including confidence intervals and status information.

    :param fit_params: Dictionary containing fitted parameter information.
    :type fit_params: dict[str, Any]
    :param param_id: Name of the parameter in fit_params.
    :type param_id: str
    :param designation: Display name of the parameter.
    :type designation: str
    :param write_fn: Function used to write into console or file.
    :type write_fn: Callable[[str], None]
    :param line_sep: Symbols to finish the line.
    :type line_sep: str
    :param precision: Number of significant figures for rounding (default: 8).
    :type precision: int
    :returns: None.
    :rtype: None
    """
    if "confidence_interval" in fit_params[param_id]:
        bot = fit_params[param_id]["value"] - fit_params[param_id]["confidence_interval"]["min"]
        top = fit_params[param_id]["confidence_interval"]["max"] - fit_params[param_id]["value"]

        aux = np.abs([bot, top])
        aux[aux == 0] = LOG_ZERO_REPLACEMENT
        sig_figures = -int(np.log10(np.min(aux)) // 1) + 1

        bot = round(bot, sig_figures)
        top = round(top, sig_figures)
    else:
        bot, top = "-", "-"
        sig_figures = precision

    status = "Not recognized"
    if "fixed" in fit_params[param_id]:
        status = "Fixed" if fit_params[param_id]["fixed"] else "Variable"
    elif "constraint" in fit_params[param_id]:
        status = fit_params[param_id]["constraint"]
    elif param_id == "r_squared":
        status = "Derived"

    unit = str(fit_params[param_id]["unit"]) if "unit" in fit_params[param_id] else "-"
    args = (
        write_fn,
        designation,
        round(fit_params[param_id]["value"], sig_figures),
        bot,
        top,
        unit,
        status,
        line_sep,
    )
    write_ln(*args, precision=sig_figures)


def write_propagated_ln(
    values: NDArray,
    fit_params: dict[str, Any],
    param_id: str,
    designation: str,
    write_fn: Callable[[str], None],
    line_sep: str,
    unit: str,
) -> None:
    """Write parameter line for propagated error parameters.

    Auxiliary function for fit_summary functions that produces formatted output
    for parameters with propagated errors. Omits lines if parameter does not exist
    in the given fitting mode.

    :param values: Array containing [value, lower_error, upper_error].
    :type values: NDArray
    :param fit_params: Dictionary containing fitted parameter information.
    :type fit_params: dict[str, Any]
    :param param_id: Name of the parameter in fit_params.
    :type param_id: str
    :param designation: Display name of the parameter.
    :type designation: str
    :param write_fn: Function used to write into console or file.
    :type write_fn: Callable[[str], None]
    :param line_sep: Symbols to finish the line.
    :type line_sep: str
    :param unit: Unit of measurement for the parameter.
    :type unit: str
    :returns: None.
    :rtype: None
    """
    # if parameter does not exist in given fitting mode, the line in summary is omitted
    if np.isnan(values).any():
        return

    aux = np.abs([values[1], values[2]])
    aux[aux <= MIN_ERROR_THRESHOLD] = MIN_ERROR_THRESHOLD
    sig_figures = -int(np.log10(np.min(aux)) // 1) + 1

    values = np.round(values, sig_figures)

    if param_id not in fit_params:
        status = "Derived"
    elif "fixed" in fit_params[param_id]:
        status = "Fixed" if fit_params[param_id]["fixed"] else "Variable"
    elif "constraint" in fit_params[param_id]:
        status = fit_params[param_id]["constraint"]
    elif param_id == "r_squared":
        status = "Derived"
    else:
        status = "Unknown"

    write_ln(
        write_fn,
        designation,
        values[0],
        values[1],
        values[2],
        unit,
        status,
        line_sep,
    )
