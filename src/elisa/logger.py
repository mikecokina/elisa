"""Logging utilities for ELISa.

This module provides a unified logging interface with support for suppressing loggers
when needed. It wraps Python's standard logging module with additional flexibility.

The module supports two types of loggers:
- Standard loggers: Return Python's logging.Logger instances
- Suppressed loggers: Return a dummy Logger that silently ignores all messages

Logger suppression can be controlled globally via settings.SUPPRESS_LOGGER.
"""

from __future__ import annotations

import logging
from logging import Logger as StandardLogger
from typing import TYPE_CHECKING

from elisa import settings

if TYPE_CHECKING:
    from typing import Any

settings.set_up_logging()


# noinspection PyPep8Naming
def getLogger(name: str, suppress: bool = False) -> StandardLogger | Logger:  # noqa: N802,FBT001,FBT002
    """Get a logger instance with optional suppression.

    Returns either a standard Python logger or a suppressed logger that silently
    ignores all logging calls. The suppression can be controlled per-logger or
    globally via settings.SUPPRESS_LOGGER.

    :param name: str; logger name (typically module name, e.g., 'elisa.binary_system')
    :param suppress: bool; if True, return a suppressed logger; can be overridden
                     by settings.SUPPRESS_LOGGER (default: False)

    :return: logging.Logger | Logger; standard logger if not suppressed,
             dummy Logger instance if suppressed
    """
    if settings.SUPPRESS_LOGGER is not None:
        suppress = settings.SUPPRESS_LOGGER

    return logging.getLogger(name=name) if not suppress else Logger(name)


# noinspection PyPep8Naming
def getPersistentLogger(name: str) -> StandardLogger:  # noqa: N802
    """Get a persistent logger that cannot be suppressed.

    Always returns a standard Python logger regardless of suppression settings.
    Use this for critical logging that should never be suppressed.

    :param name: str; logger name (typically module name, e.g., 'elisa.binary_system')

    :return: logging.Logger; standard logger instance
    """
    return logging.getLogger(name=name)


class Logger:
    """Dummy logger that silently ignores all logging calls.

    This class provides a no-op logger implementation that mimics the interface
    of Python's standard logging.Logger but performs no actual logging.
    Used when logging suppression is enabled via getLogger(suppress=True).
    """

    def __init__(self, name: str, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        """Initialize a dummy logger instance.

        :param name: str; logger name (stored but not used for logging)
        :param args: Any; positional arguments (ignored for compatibility)
        :param kwargs: Any; keyword arguments (ignored for compatibility)
        """
        self.name = name

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log an info level message (no-op).

        :param msg: str; message to log (ignored)
        :param args: Any; positional arguments for message formatting (ignored)
        :param kwargs: Any; keyword arguments (ignored)

        :return: None
        """
        ...  # noqa: PIE790

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log an error level message (no-op).

        :param msg: str; message to log (ignored)
        :param args: Any; positional arguments for message formatting (ignored)
        :param kwargs: Any; keyword arguments (ignored)

        :return: None
        """
        ...  # noqa: PIE790

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a debug level message (no-op).

        :param msg: str; message to log (ignored)
        :param args: Any; positional arguments for message formatting (ignored)
        :param kwargs: Any; keyword arguments (ignored)

        :return: None
        """
        ...  # noqa: PIE790

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a warning level message (no-op).

        :param msg: str; message to log (ignored)
        :param args: Any; positional arguments for message formatting (ignored)
        :param kwargs: Any; keyword arguments (ignored)

        :return: None
        """
        ...  # noqa: PIE790

    def warn(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a warning level message using deprecated name (no-op).

        This is an alias for warning() for backward compatibility.

        :param msg: str; message to log (ignored)
        :param args: Any; positional arguments for message formatting (ignored)
        :param kwargs: Any; keyword arguments (ignored)

        :return: None
        """
        ...  # noqa: PIE790
