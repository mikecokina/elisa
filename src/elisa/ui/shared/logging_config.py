"""Utility for managing logging configuration during fitting operations.

This module provides context managers to temporarily change the logging
configuration during fitting (to use the 'fit' schema) and restore it
afterward.
"""

from __future__ import annotations

import contextlib
from threading import RLock
from typing import TYPE_CHECKING, Any

from elisa.conf.logconfig import LoggingProfile
from elisa.conf.settings import Settings

if TYPE_CHECKING:
    from collections.abc import Generator


# Fit callbacks may overlap across tabs/sessions. Protect only policy transitions,
# retaining the fit profile until the last active callback exits.
_fit_log_lock = RLock()
_active_fits = 0
_previous_profile: str | None = None


@contextlib.contextmanager
def fit_logging() -> Generator[None, Any, None]:
    """Context manager to temporarily enable fit logging configuration.

    Sets the logging configuration to 'fit' to suppress noisy model logs
    while retaining fitting output, then restores the prior effective policy
    after the last active fit exits, including overlapping callbacks and errors.

    Example::

        with fit_logging():
            result = task.fit(x0=params)

    :yields: None
    :rtype: contextlib.AbstractContextManager[None]
    """
    global _active_fits, _previous_profile  # noqa: PLW0603
    settings = Settings()
    with _fit_log_lock:
        if _active_fits == 0:
            previous = settings.LOG_CONFIG
            settings.configure_logging(LoggingProfile.FIT.value)
            _previous_profile = previous
        _active_fits += 1
    try:
        yield
    finally:
        with _fit_log_lock:
            _active_fits -= 1
            if _active_fits == 0:
                previous = _previous_profile
                if previous is None:
                    msg = "Previous logging profile is missing when the last fit exits."
                    raise RuntimeError(msg)
                _previous_profile = None
                settings.configure_logging(previous)
