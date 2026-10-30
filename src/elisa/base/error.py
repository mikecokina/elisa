"""Project-specific exceptions used across ELISa.

This module defines the base and domain-specific exception types used by
ELISa. Keep these lightweight so they can be raised and caught by the
rest of the codebase.
"""
from __future__ import annotations

from typing import NoReturn


class ElisaValueError(ValueError):
    """Value error explicitly raised by ELISA, with no proposal-rejection semantics.

    This origin marker covers configuration, numerical and internal contract
    failures. It is deliberately separate from PhysicalStateError and ElisaError.
    """


class ElisaRuntimeError(RuntimeError):
    """Runtime failure explicitly raised by ELISA; not a physical-state signal."""


class ElisaNotImplementedError(NotImplementedError):
    """An ELISA feature or abstract operation has no implementation."""


class PredictionAlignmentError(ValueError):
    """Synthetic prediction keys or shapes do not align with observations.

    Direct prediction calls expose these errors. Fitting callbacks reject the
    candidate and record bounded diagnostics for integration failures.
    """


class ElisaError(Exception):
    """Base class for all ELISa-specific exceptions.

    Catch this when you want to handle all errors produced by ELISa.
    """

# ###

class ElisaInputKwargsError(ElisaError):
    """Unknown or missing required keyword arguments."""


class ElisaInvalidKwargsError(ElisaInputKwargsError):
    """Keyword argument names are not accepted by the target object."""


class ElisaMissingKwargsError(ElisaInputKwargsError):
    """Required keyword arguments were not supplied."""


class ElisaAnalyticsTaskError(ElisaError):
    """Invalid configuration of an analytics task."""


class ElisaAnalyticsDataError(ElisaError):
    """Invalid observational data or dataset conversion metadata."""


class ElisaInitialParametersError(ElisaError):
    """Invalid initial fitting parameters, bounds, units or constraints."""


class ElisaParametersError(ElisaError):
    """Invalid parameters supplied to a physical model or its components."""


class ElisaGraphicsError(ElisaError):
    """Invalid configuration of a graphics / plotting task."""


class ElisaMonteCarloError(ElisaError):
    """Invalid configuration of a Monte Carlo helpers."""
# ###


class ElisaSynthesisError(ElisaError):
    """Expected failure while constructing a physical synthetic model."""


class ElisaValidationError(ElisaSynthesisError):
    """Invalid parameter value detected during ELISA parameter validation."""


class PulsationModeError(ElisaSynthesisError):
    """An invalid pulsation mode or unusable pulsation perturbation."""


class MaxIterationError(ElisaError):
    """Raised when an iterative solver exceeds the maximum allowed iterations."""


class ElisaMaxIterationError(MaxIterationError):
    """Convergence failure explicitly identified by ELISA."""


class ElisaFloatingPointError(FloatingPointError):
    """Floating-point failure explicitly identified by ELISA."""


class AtmosphereError(ElisaSynthesisError):
    """Raised for errors related to atmosphere model handling."""


class TemperatureError(ElisaSynthesisError):
    """Raised when temperature-related validation or conversions fail."""


class LimbDarkeningError(ElisaSynthesisError):
    """Raised for limb-darkening related issues (parsing / validation)."""


class LimbDarkeningSourceError(LimbDarkeningError):
    """Raised when a required limb-darkening source table is unavailable."""


class PhysicalStateError(ElisaSynthesisError):
    """An explicitly identified inadmissible physical configuration."""


class MorphologyError(PhysicalStateError):
    """Raised when morphology computations fail or produce invalid results."""


class SpotError(ElisaSynthesisError):
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
        # Preserve backwards compatible behavior: message comes either
        # from args[0] or from kwargs['message'] if present.
        message = args[0] if args else kwargs.get("message")

        # Attach solution object for consumers of this exception
        self.solution = solution if solution is not None else kwargs.get("solution")

        # Always pass a single message variable to the base Exception
        super().__init__(message)


def _raise_error(
    error_type: type[Exception],
    message: str,
    exc: Exception | None = None,
) -> NoReturn:
    """Raise an exception, preserving implicit context unless a cause is supplied."""
    if exc is not None:
        raise error_type(message) from exc
    raise error_type(message)


def raise_value_error(message: str, exc: Exception | None = None) -> NoReturn:
    """Raise a ValueError with a given message.

    This function is a convenience wrapper to raise a ValueError with a
    specific message. It can be used to standardize error handling across
    the codebase.

    :param message: The error message to include in the ValueError.
    :type message: str
    :param exc: The original exception that caused this error, if any.
    :type exc: Exception | None
    :raises ValueError: Always raises a ValueError with the provided message.
    """
    _raise_error(ValueError, message, exc)


def raise_attribute_error(message: str, exc: Exception | None = None) -> NoReturn:
    """Raise an AttributeError with a given message.

    This function is a convenience wrapper to raise an AttributeError with a
    specific message. It can be used to standardize error handling across
    the codebase.

    :param message: The error message to include in the AttributeError.
    :type message: str
    :param exc: The original exception that caused this error, if any.
    :type exc: Exception | None
    :raises AttributeError: Always raises an AttributeError with the provided message.
    """
    _raise_error(AttributeError, message, exc)


def raise_elisa_not_implemented_error(message: str, exc: Exception | None = None) -> NoReturn:
    """Raise ElisaNotImplementedError, preserving an optional original cause.

    :param message: Description of the operation that lacks an implementation.
    :param exc: Original exception, if any.
    :raises ElisaNotImplementedError: Always raised with the provided message.
    """
    _raise_error(ElisaNotImplementedError, message, exc)


def raise_elisa_graphics_error(message: str, exc: Exception | None = None) -> NoReturn:
    """Raise an ElisaGraphicsError with a given message.

    This function is a convenience wrapper to raise an ElisaGraphicsError with a
    specific message. It can be used to standardize error handling across
    the codebase.

    :param message: The error message to include in the ElisaGraphicsError.
    :type message: str
    :param exc: The original exception that caused this error, if any.
    :type exc: Exception | None
    :raises ElisaGraphicsError: Always raises an ElisaGraphicsError with the provided message.
    """
    _raise_error(ElisaGraphicsError, message, exc)


def raise_elisa_monte_carlo_error(message: str, exc: Exception | None = None) -> NoReturn:
    """Raise an ElisaMonteCarloError with a given message.

    This function is a convenience wrapper to raise an ElisaMonteCarloError with a
    specific message. It can be used to standardize error handling across
    the codebase.

    :param message: The error message to include in the ElisaMonteCarloError.
    :type message: str
    :param exc: The original exception that caused this error, if any.
    :type exc: Exception | None
    :raises ElisaMonteCarloError: Always raises an ElisaMonteCarloError with the provided message.
    """
    _raise_error(ElisaMonteCarloError, message, exc)


def raise_elisa_invalid_kwargs_error(message: str, exc: Exception | None = None) -> NoReturn:
    """Raise an ElisaInvalidKwargsError with a given message.

    This function is a convenience wrapper to raise an ElisaInvalidKwargsError with a
    specific message. It can be used to standardize error handling across
    the codebase.

    :param message: The error message to include in the ElisaInvalidKwargsError.
    :type message: str
    :param exc: The original exception that caused this error, if any.
    :type exc: Exception | None
    :raises ElisaInvalidKwargsError: Always raises an ElisaInvalidKwargsError with the provided message.
    """
    _raise_error(ElisaInvalidKwargsError, message, exc)


def raise_elisa_missing_kwargs_error(message: str, exc: Exception | None = None) -> NoReturn:
    """Raise an ElisaMissingKwargsError with a given message.

    This function is a convenience wrapper to raise an ElisaMissingKwargsError with a
    specific message. It can be used to standardize error handling across
    the codebase.

    :param message: The error message to include in the ElisaMissingKwargsError.
    :type message: str
    :param exc: The original exception that caused this error, if any.
    :type exc: Exception | None
    :raises ElisaMissingKwargsError: Always raises an ElisaMissingKwargsError with the provided message.
    """
    _raise_error(ElisaMissingKwargsError, message, exc)


def raise_elisa_analytics_task_error(message: str, exc: Exception | None = None) -> NoReturn:
    """Raise an ElisaAnalyticsTaskError with a given message.

    This function is a convenience wrapper to raise an ElisaAnalyticsTaskError with a
    specific message. It can be used to standardize error handling across
    the codebase.

    :param message: The error message to include in the ElisaAnalyticsTaskError.
    :type message: str
    :param exc: The original exception that caused this error, if any.
    :type exc: Exception | None
    :raises ElisaAnalyticsTaskError: Always raises an ElisaAnalyticsTaskError with the provided message.
    """
    _raise_error(ElisaAnalyticsTaskError, message, exc)


def raise_elisa_analytics_data_error(message: str, exc: Exception | None = None) -> NoReturn:
    """Raise an ElisaAnalyticsDataError with a given message.

    This function is a convenience wrapper to raise an ElisaAnalyticsDataError with a
    specific message. It can be used to standardize error handling across
    the codebase.

    :param message: The error message to include in the ElisaAnalyticsDataError.
    :type message: str
    :param exc: The original exception that caused this error, if any.
    :type exc: Exception | None
    :raises ElisaAnalyticsDataError: Always raises an ElisaAnalyticsDataError with the provided message.
    """
    _raise_error(ElisaAnalyticsDataError, message, exc)


def raise_elisa_initial_parameters_error(message: str, exc: Exception | None = None) -> NoReturn:
    """Raise an ElisaInitialParametersError with a given message.

    This function is a convenience wrapper to raise an ElisaInitialParametersError with a
    specific message. It can be used to standardize error handling across
    the codebase.

    :param message: The error message to include in the ElisaInitialParametersError.
    :type message: str
    :param exc: The original exception that caused this error, if any.
    :type exc: Exception | None
    :raises ElisaInitialParametersError: Always raises an ElisaInitialParametersError with the provided message.
    """
    _raise_error(ElisaInitialParametersError, message, exc)


def raise_elisa_parameters_error(message: str, exc: Exception | None = None) -> NoReturn:
    """Raise an ElisaParametersError with a given message.

    This function is a convenience wrapper to raise an ElisaParametersError with a
    specific message. It can be used to standardize error handling across
    the codebase.

    :param message: The error message to include in the ElisaParametersError.
    :type message: str
    :param exc: The original exception that caused this error, if any.
    :type exc: Exception | None
    :raises ElisaParametersError: Always raises an ElisaParametersError with the provided message.
    """
    _raise_error(ElisaParametersError, message, exc)


def raise_elisa_atmosphere_error(message: str, exc: Exception | None = None) -> NoReturn:
    """Raise an AtmosphereError with a given message.

    This function is a convenience wrapper to raise an AtmosphereError with a
    specific message. It can be used to standardize error handling across
    the codebase.

    :param message: The error message to include in the AtmosphereError.
    :type message: str
    :param exc: The original exception that caused this error, if any.
    :type exc: Exception | None
    :raises AtmosphereError: Always raises an AtmosphereError with the provided message.
    """
    _raise_error(AtmosphereError, message, exc)
