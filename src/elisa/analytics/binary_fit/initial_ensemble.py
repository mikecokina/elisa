"""One-time physical LC walker preflight, independent of posterior failure policy."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

from elisa import settings
from elisa import units as u
from elisa.analytics.binary_fit.failure_policy import SYNTHESIS_FAILURES
from elisa.analytics.models.lc import prepare_binary
from elisa.analytics.params import parameters
from elisa.base.error import (
    ElisaLimbDarkeningError,
    ElisaSettingsError,
    MorphologyError,
    raise_elisa_monte_carlo_error,
    raise_elisa_numeric_error,
    raise_elisa_prediction_alignment_error,
)
from elisa.logger import getPersistentLogger
from elisa.observer.observer import Observer

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.analytics.binary_fit.mcmc import LightCurveFit

logger = getPersistentLogger(__name__)


def _validate_candidate(fit: LightCurveFit, observer: Observer, candidate: NDArray) -> None:
    """Construct the real binary and synthesize one unnormalized point in every band."""
    # Use the likelihood's parameter preparation to include fixed values and
    # evaluated constraints, then apply the fitted atmosphere and LD overrides.
    _, kwargs = fit._prepare_likelihood_parameters(candidate)  # noqa: SLF001
    kwargs = parameters.extend_json_with_atm_params(
        kwargs,
        atmosphere_model=fit.atmosphere_model,
        limb_darkening_coefficients=fit.limb_darkening_coefficients,
    )
    binary = prepare_binary(discretization=fit.discretization, **kwargs)
    # Construction checks physical scales, Roche morphology and component radii.
    if fit.MORPHOLOGY is not None and binary.morphology != fit.MORPHOLOGY:
        msg = f"Expected {fit.MORPHOLOGY} morphology, got {binary.morphology}"
        raise MorphologyError(msg)
    observer._system = binary  # noqa: SLF001
    observer.system_cls = type(binary)
    # Orbit phases are relative to primary conjunction. Observer subtracts the
    # photometric shift before orbit evaluation; undo that here. Circular orbits
    # have no physical periastron, so use conjunction as a deterministic reference.
    phase = binary.orbit.periastron_phase if binary.eccentricity != 0.0 else 0.0
    # Explicit physical units prevent Observer from reusing dimensionless
    # normalization, which would hide invalid fluxes in a single-point curve.
    _, fluxes = observer.observe.lc(
        phases=np.array([phase + binary.phase_shift]),
        normalize=False,
        flux_unit=u.W / u.m**2,
    )
    if fluxes.keys() != fit.y_data.keys():
        raise_elisa_prediction_alignment_error("Initial LC synthesis and fitted passbands differ")
    for band, flux in fluxes.items():
        values = np.asarray(flux)
        if values.shape != (1,):
            raise_elisa_prediction_alignment_error(f"Initial single-point LC shape differs for {band}")
        if not np.all(np.isfinite(values) & (values > 0.0)):
            raise_elisa_numeric_error(f"Initial LC flux must be finite and positive for {band}")


def _propagate_source_failure(exc: Exception) -> None:
    """Do not regenerate for source failures wrapped by legacy synthesis handlers."""
    # A domain exception may wrap a missing file or a programming error.
    # Drawing different parameters cannot repair those failures.
    cause = exc.__cause__
    while cause is not None:
        if isinstance(cause, (OSError, ElisaSettingsError, TypeError, AttributeError, NameError, AssertionError)):
            raise exc
        cause = cause.__cause__
    # This legacy handler translates even source triangulation/configuration bugs.
    if isinstance(exc, ElisaLimbDarkeningError) and str(exc).startswith("Cannot construct limb-darkening interpolator"):
        raise exc


def initialize_lc_ensemble(
    fit: LightCurveFit,
    ensemble: NDArray,
    *,
    supplied: bool,
    rng: np.random.Generator | None = None,
    max_attempts: int | None = None,
) -> NDArray:
    """Validate supplied coordinates or replace rejected automatic uniform draws.

    Each walker has its own attempt limit, including its original draw (or x0).
    The default limit comes from MCMC_INITIAL_WALKER_MAX_ATTEMPTS.
    The RNG only supplies replacement draws;
    original draws retain generate_initial_states' distribution and ordering.
    Explicit ensembles are never repaired. No likelihood is evaluated.
    """
    rng = np.random.default_rng() if rng is None else rng
    nwalkers, ndim = ensemble.shape
    limit = settings.MCMC_INITIAL_WALKER_MAX_ATTEMPTS if max_attempts is None else max_attempts
    # Load the fitted passbands once; this Observer owns all mutable result state.
    observer = Observer(passband=list(fit.y_data))
    result = ensemble.copy()
    attempted = accepted = 0
    invalid = []
    reasons: Counter[str] = Counter()
    last_failure = ""
    exhausted_walker: tuple[int, int, dict[str, int]] | None = None
    for index, original in enumerate(ensemble):
        # Try the existing candidate first, preserving x0 when it is valid.
        # Reset the budget for each walker; supplied coordinates get one check.
        candidate = original
        walker_reasons: Counter[str] = Counter()
        for walker_attempt in range(1, (1 if supplied else limit) + 1):
            attempted += 1
            try:
                _validate_candidate(fit, observer, candidate)
            except SYNTHESIS_FAILURES as exc:
                # Share the sampling policy's domain classification, but never
                # apply its emergency PENALIZE_ALL_EXCEPTIONS override here.
                _propagate_source_failure(exc)
                reasons[type(exc).__name__] += 1
                walker_reasons[type(exc).__name__] += 1
                last_failure = str(exc)[:512]
                if supplied:
                    # Collect all invalid supplied indices without repairing them.
                    invalid.append(index)
                    break
                if walker_attempt < limit:
                    # Replace the entire parameter combination with a uniform
                    # draw in normalized prior coordinates.
                    candidate = rng.uniform(0.0, 1.0, ndim)
            else:
                result[index] = candidate
                accepted += 1
                break
        if not supplied and accepted != index + 1:
            # This walker exhausted its own budget; do not start later walkers.
            exhausted_walker = index, limit, dict(walker_reasons)
            break
    # Keep aggregate diagnostics and the last rejection without per-draw tracebacks.
    summary = (
        f"Initial LC ensemble: attempted={attempted}, accepted={accepted}, rejected={attempted - accepted}; "
        f"requested={nwalkers}; failures={dict(reasons)}; last failure={last_failure}"
    )
    fit._initialization_diagnostics = {  # noqa: SLF001
        "attempted": attempted,
        "accepted": accepted,
        "rejected": attempted - accepted,
        "failures": dict(reasons),
    }
    logger.info("Initial ensemble: %s/%s valid, %s rejected", accepted, nwalkers, attempted - accepted)
    if invalid:
        raise_elisa_monte_carlo_error(f"Invalid supplied initial walkers (zero-based): {invalid}. {summary}")
    if exhausted_walker is not None:
        failed_index, attempts, failures = exhausted_walker
        raise_elisa_monte_carlo_error(
            f"Initial walker {failed_index} (zero-based) retry limit exhausted after {attempts} attempts; "
            f"walker failures={failures}. {summary}",
        )
    return result
