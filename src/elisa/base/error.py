"""Project-specific exceptions used across ELISa.

This module defines the base and domain-specific exception types used by
ELISa. Keep these lightweight so they can be raised and caught by the
rest of the codebase.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    from collections.abc import Callable


class ElisaUnexpectedError(RuntimeError):
    """An internal state or execution-order invariant guaranteed by ELISA was violated."""


class ElisaNotImplementedError(NotImplementedError):
    """An ELISA feature or abstract operation has no implementation."""


class ElisaPredictionAlignmentError(ValueError):
    """Synthetic prediction keys or shapes do not align with observations.

    Direct prediction calls expose these structural mismatches. Candidate
    penalization depends on the fitting policy and PENALIZE_ALL_EXCEPTIONS setting.
    """


class ElisaError(Exception):
    """Base class for ELISa exceptions that share this hierarchy.

    Some ELISa exceptions intentionally inherit directly from builtin exceptions
    and are not caught by catching ElisaError.
    """


class ElisaOperationInputError(ElisaError):
    """Invalid argument values or combinations for a public model operation.

    This concerns invoking an operation on an existing model, rather than
    defining physical model parameters or rejecting a synthetic candidate.
    """


class ElisaPassbandError(ElisaError):
    """Invalid fixed passband definition or unavailable passband source.

    This is an input/source failure, never a reason to reject a model candidate.
    """


class ElisaSettingsError(ElisaError):
    """Invalid ELISA configuration or failure while preparing its supporting assets."""

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


class PulsationModeError(ElisaError):
    """An invalid pulsation mode or unusable pulsation perturbation."""


class ElisaMaxIterationError(ElisaError):
    """Convergence failure explicitly identified by ELISA."""


class ElisaNumericError(FloatingPointError):
    """Recognized numerical failure while evaluating an ELISA model or candidate."""


class ElisaNumericSolverError(ElisaError):
    """Low-level solver failure; its caller assigns domain and fitting semantics."""


class ElisaLookupError(ElisaError):
    """Low-level numeric lookup failure; the caller must assign domain semantics.

    This marker is not a candidate rejection category. Atmosphere, LD or other
    callers must translate it before it reaches application/fitting boundaries.
    """


class ElisaAtmosphereError(ElisaError):
    """Raised for errors related to atmosphere model handling."""


class ElisaTemperatureError(ElisaError):
    """An inadmissible temperature or unusable temperature model calculation."""


class ElisaLimbDarkeningError(ElisaError):
    """Failure to obtain usable limb-darkening coefficients for a model."""


class MorphologyError(ElisaError):
    """Raised when morphology computations fail or produce invalid results."""


class ElisaSpotError(ElisaError):
    """An unusable spot geometry or failed spot model calculation."""


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


def raise_elisa_unexpected_error(message: str, exc: Exception | None = None) -> NoReturn:
    """Raise an internal contract failure, preserving an optional original cause.

    :param message: Description of the violated internal invariant.
    :param exc: Original exception, if any.
    :raises ElisaUnexpectedError: Always raised with the provided message.
    """
    _raise_error(ElisaUnexpectedError, message, exc)


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


def raise_elisa_operation_input_error(message: str, exc: Exception | None = None) -> NoReturn:
    """Raise an invalid-operation-input error with an optional original cause.

    :param message: Description of invalid operation arguments.
    :param exc: Original exception, if any.
    :raises ElisaOperationInputError: Always raised with the provided message.
    """
    _raise_error(ElisaOperationInputError, message, exc)


def raise_elisa_numeric_error(message: str, exc: Exception | None = None) -> NoReturn:
    """Raise a recognized numerical failure, preserving an optional original cause.

    :param message: Description of the nonfinite or unusable numerical result.
    :param exc: Original exception, if any.
    :raises ElisaNumericError: Always raised with the provided message.
    """
    _raise_error(ElisaNumericError, message, exc)


def raise_elisa_numeric_solver_error(message: str, exc: Exception | None = None) -> NoReturn:
    """Wrap a low-level solver failure while preserving its original cause."""
    _raise_error(ElisaNumericSolverError, message, exc)


def raise_elisa_temperature_error(message: str, exc: Exception | None = None) -> NoReturn:
    """Reject an unusable temperature model, preserving an optional cause."""
    _raise_error(ElisaTemperatureError, message, exc)


def raise_elisa_spot_error(message: str, exc: Exception | None = None) -> NoReturn:
    """Reject an unusable spot model while preserving an optional cause."""
    _raise_error(ElisaSpotError, message, exc)


def raise_elisa_max_iteration_error(message: str, exc: Exception | None = None) -> NoReturn:
    """Raise a recognized solver convergence failure with an optional cause.

    :param message: Description of the failed iteration.
    :param exc: Original exception, if any.
    :raises ElisaMaxIterationError: Always raised with the provided message.
    """
    _raise_error(ElisaMaxIterationError, message, exc)


def raise_elisa_prediction_alignment_error(message: str, exc: Exception | None = None) -> NoReturn:
    """Raise ElisaPredictionAlignmentError, preserving an optional original cause.

    :param message: Description of incompatible dataset keys, shapes or coordinates.
    :param exc: Original exception, if any.
    :raises ElisaPredictionAlignmentError: Always raised with the provided message.
    """
    _raise_error(ElisaPredictionAlignmentError, message, exc)


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


def raise_elisa_passband_error(message: str, exc: Exception | None = None) -> NoReturn:
    """Raise a passband input/source error, preserving an optional cause.

    :param message: Description of the invalid passband or unavailable source.
    :param exc: Original exception, if any.
    :raises ElisaPassbandError: Always raised with the provided message.
    """
    _raise_error(ElisaPassbandError, message, exc)


def raise_elisa_settings_error(message: str, exc: Exception | None = None) -> NoReturn:
    """Raise a settings or asset-setup failure, preserving an optional original cause.

    :param message: Description of the configuration or asset-setup failure.
    :param exc: Original exception, if any.
    :raises ElisaSettingsError: Always raised with the provided message.
    """
    _raise_error(ElisaSettingsError, message, exc)


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


def raise_elisa_pulsation_mode_error(message: str, exc: Exception | None = None) -> NoReturn:
    """Reject an unusable pulsation mode, preserving an optional cause."""
    _raise_error(PulsationModeError, message, exc)


def raise_elisa_morphology_error(message: str, exc: Exception | None = None) -> NoReturn:
    """Reject a physically inadmissible morphology, preserving an optional cause."""
    _raise_error(MorphologyError, message, exc)


def raise_elisa_lookup_error(message: str, exc: Exception | None = None) -> NoReturn:
    """Raise an intermediate lookup failure while preserving its native cause."""
    _raise_error(ElisaLookupError, message, exc)


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
    _raise_error(ElisaAtmosphereError, message, exc)


def raise_elisa_limb_darkening_error(message: str, exc: Exception | None = None) -> NoReturn:
    """Raise a limb-darkening failure, preserving its original cause when supplied."""
    _raise_error(ElisaLimbDarkeningError, message, exc)


def _register_numba_error_helpers() -> None:
    """Register numerical raises for compiled kernels when Numba is already needed.

    Compiled code supplies only a message; Python callers retain cause chaining
    through the ordinary helper. Importing the exception module alone stays light.
    """
    from numba import types  # noqa: PLC0415
    from numba.extending import overload  # noqa: PLC0415

    @overload(raise_elisa_numeric_error)
    def _numeric_error_overload(
        message: object, exc: object = None,  # noqa: ARG001 - matching helper signature
    ) -> Callable[..., NoReturn] | None:
        if exc is not None and not isinstance(exc, types.NoneType):
            return None

        def implementation(message: object, exc: object = None) -> NoReturn:  # noqa: ARG001 - matching helper signature
            raise ElisaNumericError(message)

        return implementation
