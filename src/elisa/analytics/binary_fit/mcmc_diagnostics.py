"""Best-effort MCMC diagnostics with explicitly shared per-run counters.

The parent activates a run before creating its pool and deactivates it after the
pool closes. Worker initializers activate that context and mutate only the locked counters
and an idempotent failure flag. Shared objects never live on the pickled fitter.
Only the parent reports snapshots or writes files; independent run contexts do
not share counters. Ordinary instrumentation failures disable diagnostics while
leaving synthesis, posterior evaluation and sampler exceptions untouched.
"""
from __future__ import annotations

# Ordinary instrumentation errors must not escape into MCMC.
# ruff: noqa: BLE001
import json
import os
import tempfile
from contextlib import suppress
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    # noinspection PyProtectedMember
    from multiprocessing.context import BaseContext

from elisa import settings
from elisa.analytics.binary_fit.failure_policy import SYNTHESIS_FAILURES
from elisa.logger import getPersistentLogger
from elisa.observer.mp_manager import get_mp_context

logger = getPersistentLogger(__name__)
_active: ContextVar[RunDiagnostics | None] = ContextVar("mcmc_diagnostics", default=None)
_NAMES = (*(cls.__name__ for cls in SYNTHESIS_FAILURES), "other")


def count_failure(exc: Exception) -> bool:
    """Count one penalized evaluation in a worker or the serial parent.

    Return True for an active (even disabled) run, suppressing legacy per-failure
    payloads. Return False outside sampling. Counter/context failures are nonfatal;
    unavailable context is treated as instrumentation failure, not a model error.
    """
    try:
        run = _active.get()
        if run is None:
            return False
        if run.enabled():
            index = next((i for i, cls in enumerate(SYNTHESIS_FAILURES) if isinstance(exc, cls)), len(_NAMES) - 1)
            with run.counters.get_lock():
                run.counters[index] += 1
    except Exception as failure:
        diagnostics_failed(failure)
    return True


class RunDiagnostics:
    """Parent snapshots and one synchronized shared array per sampling run."""

    def __init__(self, context: BaseContext | None, initialization: dict, chain_path: str | Path | None = None) -> None:
        """Allocate parent-owned run state; None requests the supported worker context.

        Context lookup and shared-memory allocation are optional diagnostics work:
        ordinary failures leave a disabled run that sampling can still enter.
        ``chain_path`` is an already resolved destination; no fit ID is resolved here.
        """
        self.disabled = False
        self.warned = False
        self.owner_pid = None
        self.failed = None
        self.token = None
        try:
            self.owner_pid = os.getpid()
            context = get_mp_context() if context is None else context
            self.failed = context.RawValue("b", 0)
            self.counters = context.Array("q", len(_NAMES))
            self.path = Path(chain_path).with_suffix(".diagnostics.json") if chain_path else None
            self.data: dict[str, Any] = {"initialization": initialization}
            for phase in ("burn_in", "production"):
                self.data[phase] = {"completed_steps": 0, "total_failures": 0, "failures": {}, "history": []}
            self.baseline = [0] * len(_NAMES)
            self.previous = [0] * len(_NAMES)
        except Exception as failure:
            self.disable(failure)

    def __getstate__(self) -> dict[str, Any]:
        """Transfer shared worker state without parent tokens or output bookkeeping.

        Pool initialization permits passing the synchronized array and failure flag
        to spawned workers. These objects must never be sent as ordinary pool tasks.
        """
        return {
            "disabled": self.disabled,
            "warned": False,
            "owner_pid": self.owner_pid,
            "failed": self.failed,
            "counters": getattr(self, "counters", None),
            "token": None,
            "path": None,
        }

    def activate_worker(self) -> None:
        """Attach shared counters in a worker without taking ownership of output."""
        try:
            _active.set(self)
        except Exception as failure:
            self.disable(failure)

    def __enter__(self) -> Self:
        """Activate this parent run, returning even if diagnostics is disabled."""
        try:
            self.token = _active.set(self)
        except Exception as failure:
            self.disable(failure)
        return self

    def __exit__(self, *_exc: object) -> None:
        """Flush in the parent and restore context without masking sampler errors.

        Ordinary write/reset failures disable diagnostics; control-flow exceptions
        remain uncaught. A failed reset gets a best-effort context clear.
        """
        try:
            self.write()
        except Exception as failure:
            self.disable(failure)
        finally:
            try:
                if self.token is not None:
                    _active.reset(self.token)
            except Exception as failure:
                self.disable(failure)
                with suppress(Exception):
                    _active.set(None)

    def disable(self, failure: Exception) -> None:
        """Mark local/shared diagnostics unusable; warn at most once in the parent.

        Workers set the shared flag rather than logging. Even failure of the flag
        or warning operation is contained so rejection penalties remain unchanged.
        """
        self.disabled = True
        try:
            if self.failed is not None:
                self.failed.value = True
            if os.getpid() == self.owner_pid and not self.warned:
                self.warned = True
                logger.warning("MCMC diagnostics disabled (%s)", type(failure).__name__)
        except Exception:  # noqa: S110
            pass  # Diagnostics, including its warning, must never affect sampling.

    def enabled(self) -> bool:
        """Return usability, propagating a worker's failure flag to the parent warning."""
        try:
            if self.failed is not None and self.failed.value and not self.disabled:
                self.disable(RuntimeError("worker diagnostics failed"))
        except Exception as failure:
            self.disable(failure)
        return not self.disabled

    def begin_production(self) -> None:
        """Snapshot the parent's burn-in baseline without resetting shared counters.

        Later production counts exclude burn-in. Read failures disable diagnostics.
        """
        if not self.enabled():
            return
        try:
            with self.counters.get_lock():
                self.baseline = self.counters[:]
            self.previous = [0] * len(_NAMES)
        except Exception as failure:
            self.disable(failure)

    def report(self, phase: str, step: int, total: int) -> None:
        """Read counters and record/log a parent snapshot at an interval or phase end.

        Totals are cumulative within the phase; history contains interval deltas.
        Snapshot, logging and persistence errors disable this optional run only.
        """
        if not self.enabled():
            return
        try:
            if step % settings.MCMC_DIAGNOSTICS_INTERVAL_STEPS and step != total:
                return
            with self.counters.get_lock():
                counts = [v - b for v, b in zip(self.counters[:], self.baseline, strict=True)]
            interval = [v - p for v, p in zip(counts, self.previous, strict=True)]
            section = self.data[phase]
            section.update(
                completed_steps=step, total_steps=total, total_failures=sum(counts), failures=_failures(counts),
            )
            section["history"].append(
                {"step": step, "rejected_in_interval": sum(interval), "failures": _failures(interval)},
            )
            self.previous = counts
            label = "MCMC step" if phase == "production" else "MCMC burn-in step"
            logger.info("%s %s/%s: %s invalid-model proposals", label, step, total, sum(counts))
            self.write()
        except Exception as failure:
            self.disable(failure)

    def write(self) -> None:
        """Atomically write the parent's snapshot beside the chain, when saving.

        Temporary files are cleaned up where possible. Serialization, filesystem
        and cleanup failures disable diagnostics without replacing an older file
        until a complete new snapshot is ready, or escaping into sampling.
        """
        if not self.enabled():
            return
        try:
            if self.path is None:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", dir=self.path.parent, delete=False) as stream:
                    temporary = stream.name
                    json.dump(self.data, stream, indent=2)
                Path(temporary).replace(self.path)
            finally:
                if temporary and Path(temporary).exists():
                    Path(temporary).unlink()
        except Exception as failure:
            self.disable(failure)


def _failures(values: list[int]) -> dict[str, int]:
    """Build the parent's sparse JSON category counts from a counter snapshot."""
    return {name: value for name, value in zip(_NAMES, values, strict=True) if value}


def diagnostics_failed(failure: Exception) -> None:
    """Disable active diagnostics after a callback failure, in parent or worker.

    Context lookup itself may be broken. In that case reporting is unavailable,
    so suppress ordinary failures here rather than masking the model's penalty.
    """
    with suppress(Exception):
        run = _active.get()
        if run is not None:
            run.disable(failure)

