"""Project-specific exceptions used across ELISa.

This module defines the base and domain-specific exception types used by
ELISa. Keep these lightweight so they can be raised and caught by the
rest of the codebase.
"""
from __future__ import annotations

from jsonschema import ValidationError


class ElisaValueError(ValueError):
    """Value error explicitly raised by ELISA, with no proposal-rejection semantics.

    This origin marker covers configuration, numerical and internal contract
    failures. It is deliberately separate from PhysicalStateError and ElisaError.
    """


class ElisaRuntimeError(RuntimeError):
    """Runtime failure explicitly raised by ELISA; not a physical-state signal."""


class PredictionAlignmentError(ValueError):
    """Synthetic prediction keys or shapes do not align with observations.

    Structural integration errors must propagate through fitting rather than
    becoming physical-state rejections or least-squares penalties.
    """


class YouHaveNoIdeaError(ValidationError):
    """Raised when a JSON schema validation fails in an unexpected way."""


class ElisaError(Exception):
    """Base class for all ELISa-specific exceptions.

    Catch this when you want to handle all errors produced by ELISa.
    """


class MaxIterationError(ElisaError):
    """Raised when an iterative solver exceeds the maximum allowed iterations."""


class ElisaMaxIterationError(MaxIterationError):
    """Convergence failure explicitly identified by ELISA."""


class ElisaFloatingPointError(FloatingPointError):
    """Floating-point failure explicitly identified by ELISA."""


class InitialParamsError(ElisaError):
    """Raised when provided initial parameters are invalid or inconsistent."""


class AtmosphereError(ElisaError):
    """Raised for errors related to atmosphere model handling."""


class TemperatureError(ElisaError):
    """Raised when temperature-related validation or conversions fail."""


class LimbDarkeningError(ElisaError):
    """Raised for limb-darkening related issues (parsing / validation)."""


class MetallicityError(ElisaError):
    """Raised when metallicity-related validation fails."""


class GravityError(ElisaError):
    """Raised when gravity-related validation fails."""


class PhysicalStateError(ElisaError):
    """An explicitly identified inadmissible physical configuration."""


class MorphologyError(PhysicalStateError):
    """Raised when morphology computations fail or produce invalid results."""


class SpotError(ElisaError):
    """Raised for spot-related errors (definition / serialization)."""


class SolutionBubbleError(ElisaError):
    """Exception carrying a proposed solution object.

    This exception is used to signal that the solver reached a solution
    bubble (a local solution) and carries the candidate solution via the
    ``solution`` attribute. The exception message is accepted either as
    the first positional argument or as the ``message`` keyword argument.

    :param args: Positional arguments forwarded to :class:`Exception`.
    :param solution: Optional solution object attached to the exception.
    :param kwargs: Additional keyword arguments; ``message`` is treated as
        the exception message if provided.
    """

    def __init__(self, *args, solution: object | None = None, **kwargs) -> None:
        # Preserve backwards compatible behaviour: message comes either
        # from args[0] or from kwargs['message'] if present.
        message = args[0] if args else kwargs.get("message")

        # Attach solution object for consumers of this exception
        self.solution = solution if solution is not None else kwargs.get("solution")

        # Always pass a single message variable to the base Exception
        super().__init__(message)
