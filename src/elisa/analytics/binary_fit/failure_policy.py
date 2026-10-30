"""Fitting failure policy and bounded MCMC candidate diagnostics.

LSQ converts only SYNTHESIS_FAILURES into penalty residuals. Other errors
propagate out of its callback. MCMC records ordinary exceptions during prior
or likelihood evaluation and rejects the candidate with -inf. Setup errors
and process-control exceptions (such as KeyboardInterrupt) still propagate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from reprlib import Repr
from traceback import format_exception
from typing import TYPE_CHECKING

from elisa import settings
from elisa.base.error import (
    ElisaFloatingPointError,
    ElisaParametersError,
    ElisaSynthesisError,
    MaxIterationError,
)
from elisa.logger import getPersistentLogger

# Candidate models validate their own parameters. Those domain failures
# reject LSQ proposals just like synthesis failures; setup remains outside callbacks.
# Numerical markers retain their legacy inheritance and remain explicit here.
SYNTHESIS_FAILURES = (
    ElisaParametersError,
    ElisaSynthesisError,
    MaxIterationError,
    ElisaFloatingPointError,
)


if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.analytics.binary_fit.mcmc import MCMCFit
    from elisa.types import Float


logger = getPersistentLogger(__name__)
_MAX_FAILURE_TYPES = 16
_MAX_TRACEBACK_LENGTH = 4096
_diagnostic_repr = Repr()
_diagnostic_repr.maxother = 512
_diagnostic_repr.maxstring = 512
_diagnostic_repr.maxtuple = 16


@dataclass
class CandidateFailureDiagnostics:
    """Bounded, pickleable diagnostics local to a fit instance or worker.

    Counts retain up to sixteen exception types plus an overflow bucket.
    Only the latest failure is retained, as text without exception/traceback
    objects that would keep model arrays alive. Worker copies are not merged
    into the parent fit; their first failures are also reported through logging.
    """

    total: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    last: dict[str, str] = field(default_factory=dict)


def record_candidate_failure(
    fit: MCMCFit,
    xn: NDArray[Float],
    exc: Exception,
    *,
    stage: str = "likelihood",
) -> None:
    """Record an MCMC rejection and warn once per retained exception category.

    Warning suppression does not disable in-memory diagnostics. Normalized
    candidate values and their parameter order use bounded representations;
    traceback text includes the original exception chain when available.
    """
    # noinspection PyProtectedMember
    diagnostics = fit._candidate_failure_diagnostics  # noqa: SLF001
    error_type = f"{type(exc).__module__}.{type(exc).__qualname__}"[:256]
    category = error_type
    if category not in diagnostics.counts and len(diagnostics.counts) >= _MAX_FAILURE_TYPES:
        category = "other"
    diagnostics.total += 1
    diagnostics.counts[category] = diagnostics.counts.get(category, 0) + 1
    diagnostics.last = {
        "stage": stage,
        "exception_type": error_type,
        "message": str(exc)[:512],
        "normalized_candidate": _diagnostic_repr.repr(xn),
        "parameter_order": _diagnostic_repr.repr(tuple(getattr(fit, "fitable", {}))),
        "traceback": "".join(format_exception(type(exc), exc, exc.__traceback__))[-_MAX_TRACEBACK_LENGTH:],
    }
    if diagnostics.counts[category] == 1 and not settings.SUPPRESS_WARNINGS:
        logger.warning(
            "MCMC candidate rejected during %s; parameters=%s; normalized=%s. "
            "Further failures in category %s are counted without repeated warnings.\n%s",
            stage,
            diagnostics.last["parameter_order"],
            diagnostics.last["normalized_candidate"],
            category,
            diagnostics.last["traceback"],
        )
