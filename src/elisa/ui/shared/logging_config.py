"""Utility for managing logging configuration during fitting operations.

This module provides context managers to temporarily change the logging
configuration during fitting (to use the 'fit' schema) and restore it
afterward.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from elisa.conf.settings import Settings

if TYPE_CHECKING:
    from collections.abc import Generator


@contextlib.contextmanager
def fit_logging() -> Generator[None, Any, None]:
    """Context manager to temporarily enable fit logging configuration.

    Sets the logging configuration to 'fit' (which is more verbose for
    analytics operations) upon entry, and restores the original
    configuration upon exit.

    Example::

        with fit_logging():
            result = task.fit(x0=params)

    :yields: None
    :rtype: contextlib.AbstractContextManager[None]
    """
    settings = Settings()
    original_log_config = settings.LOG_CONFIG

    try:
        settings.configure(LOG_CONFIG="fit")
        yield
    finally:
        # Restore original logging configuration
        settings.configure(LOG_CONFIG=original_log_config)

