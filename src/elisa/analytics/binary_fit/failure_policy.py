"""Expected proposal failures shared by LC/RV least squares and MCMC.

Only the types in SYNTHESIS_FAILURES and their subclasses are converted to
proposal rejections at fitting evaluation boundaries: finite penalty residuals
in LSQ, or -inf in MCMC. These include physical/domain exclusions, numerical
synthesis failures and unavailable atmosphere/limb-darkening tables; rejection
does not necessarily mean that the proposed system is physically impossible.

Other exceptions reaching these boundaries propagate out of the optimizer or
sampler and abort the fit; they must not be silently converted to penalties.
In particular, origin-only value/runtime markers and native statistical
arithmetic errors are not rejection categories. Existing local recovery inside
synthesis is separate, as are MCMC prior rejections. Setup errors propagate
before these evaluation handlers, even when their type appears in this tuple.
"""
from elisa.base.error import (
    ElisaFloatingPointError,
    ElisaSynthesisError,
    MaxIterationError,
)

# Numerical markers retain their legacy inheritance and remain explicit here.
SYNTHESIS_FAILURES = (
    ElisaSynthesisError,
    MaxIterationError,
    ElisaFloatingPointError,
)
