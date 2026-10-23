"""Utility for managing logging configuration during fitting operations.

This module provides context managers to temporarily change the logging
configuration during fitting (to use the 'fit' schema) and restore it
afterward.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from elisa.conf.logconfig import LoggingProfile
from elisa.conf.settings import Settings

if TYPE_CHECKING:
    from collections.abc import Generator


@contextlib.contextmanager
def fit_logging() -> Generator[None, Any, None]:
    """Context manager to temporarily enable fit logging configuration.

    Sets the logging configuration to 'fit' to suppress noisy model logs
    while retaining fitting output, then restores the prior effective policy
    on exit, including nested contexts and exceptions.

    Example::

        with fit_logging():
            result = task.fit(x0=params)

    :yields: None
    :rtype: contextlib.AbstractContextManager[None]
    """
    settings = Settings()
    with settings.logging_context(LoggingProfile.FIT.value):
        yield
