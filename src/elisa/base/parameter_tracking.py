"""Remember which constructor parameters came from the caller.

For example, ``star.albedo = 0.7`` makes albedo an explicit input. An albedo
calculated by BinarySystem must remain automatic, so initialization temporarily
disables recording assignments on the system and its children.

The decorators wrap existing methods; the methods still perform all validation,
unit conversion and physical calculations themselves:

* @suspend_tracking_during_initialization protects constructors and helpers that assign internal values.
  A successful external constructor call replaces the set of explicit inputs.
* @preserve_explicit_inputs_during_reinitialization protects init() and tells the nested constructor to retain
  that set instead of replacing it with the reconstruction payload.

Two separate counters serve different purposes: _tracking_depth disables
assignment recording, while _reinitializing distinguishes init() -> __init__()
from a caller invoking __init__() directly. Both are restored even on failure.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from inspect import signature
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


class ParameterTracking:
    """Track successful writes to constructor fields, never derived state."""

    ALL_KWARGS: tuple[str, ...] = ()
    MANDATORY_KWARGS: tuple[str, ...] = ()
    _explicit_parameters: set[str]
    name: str

    def __setattr__(self, name: str, value: Any) -> None:
        # Run the normal assignment/property setter first. If it rejects the
        # value, no input is recorded and its existing validation error escapes.
        super().__setattr__(name, value)
        if name in self.__dict__.get("_constructor_parameters", self.ALL_KWARGS) and not self.__dict__.get(
            "_tracking_depth", 0,
        ):
            explicit = self.__dict__.get("_explicit_parameters")
            # This set is created only after the first successful constructor.
            # Record names, not values: init() reads the current attribute later.
            if explicit is not None:
                explicit.add(name)

    def _explicit_kwargs(self, values: dict[str, Any]) -> dict[str, Any]:
        """Filter an existing unit-safe serialization to reconstruction inputs."""
        # Keep required fields even if originally missing: their invalid default
        # must reach validation rather than silently disappearing from the input.
        result = {
            key: value
            for key, value in values.items()
            if key in self._explicit_parameters or key in self.MANDATORY_KWARGS
        }
        if "name" in self._explicit_parameters:
            result["name"] = self.name
        return result


@contextmanager
def suspended(*objects) -> Iterator[None]:
    """Disable assignment recording for the duration of the context block.

    Include existing children because system setup writes defaults into Stars,
    Spots and PulsationModes. Each nested block increments the depth: leaving a
    child's init() must not enable recording while its parent's init() continues.
    """
    visited = set()
    states = []

    def enter(model: Any) -> None:
        # A child can be reachable through several roots. Visit it once per
        # block; ignore ordinary scalar arguments passed alongside model objects.
        if not isinstance(model, ParameterTracking) or id(model) in visited:
            return
        visited.add(id(model))
        current_depth = model.__dict__.get("_tracking_depth", 0)
        states.append((model, current_depth))
        # Bypass the model's __setattr__ for this private bookkeeping.
        object.__setattr__(model, "_tracking_depth", current_depth + 1)
        for name in ("_components", "_spots", "_pulsations"):
            children = model.__dict__.get(name, {})
            for child in children.values() if isinstance(children, dict) else children:
                enter(child)

    try:
        for root in objects:
            enter(root)
        yield  # Execute the caller's with-block while recording is disabled.
    finally:
        # Unwind children before parents, including when the block raises.
        # Restore the saved depth, not zero: an outer block may still be active.
        for suspended_model, previous_depth in reversed(states):
            object.__setattr__(suspended_model, "_tracking_depth", previous_depth)


def suspend_tracking_during_initialization(method: Callable) -> Callable:
    """Wrap a constructor or an internal helper that assigns constructor fields."""
    sig = signature(method)

    @wraps(method)  # Preserve the wrapped method's name, docstring and signature.
    def wrapped(self: ParameterTracking, *args, **kwargs) -> Any:
        # Map positional and keyword arguments to their signature names. This
        # includes named inputs such as name/primary/star, outside **kwargs.
        bound = sig.bind(self, *args, **kwargs)
        # Incoming children must also be protected: a fresh system has not yet
        # attached them to self._components when its constructor starts.
        with suspended(self, *args, *kwargs.values()):
            result = method(self, *args, **kwargs)
            # Reached only on success. Helpers never replace provenance, and a
            # constructor invoked by init() retains the existing explicit set.
            if method.__name__ == "__init__" and not self.__dict__.get("_reinitializing", 0):
                supplied = set(bound.arguments.get("kwargs", {}))
                supplied.update(set(bound.arguments) - {"self", "kwargs"})
                object.__setattr__(self, "_explicit_parameters", supplied)
                names = set(self.ALL_KWARGS) | (set(sig.parameters) - {"self", "kwargs"})
                object.__setattr__(self, "_constructor_parameters", names)
            return result

    return wrapped


def preserve_explicit_inputs_during_reinitialization(method: Callable) -> Callable:
    """Wrap init() so its internal __init__() preserves explicit input names.

    For BinarySystem.init(), suspension covers both child rebuilds AND the later
    parent calculations. Thus, a freshly calculated stellar albedo stays automatic
    after the child's own init() has returned.
    """

    @wraps(method)
    def wrapped(self: ParameterTracking, *args, **kwargs) -> Any:
        depth = self.__dict__.get("_reinitializing", 0)
        with suspended(self):
            # The constructor decorator checks this marker. It must survive
            # until the complete init() finishes, including nested constructors.
            object.__setattr__(self, "_reinitializing", depth + 1)
            try:
                return method(self, *args, **kwargs)
            finally:
                object.__setattr__(self, "_reinitializing", depth)

    return wrapped


def parameter_is_explicit(instance: ParameterTracking, name: str) -> bool:
    """Query provenance at the few parent/child initialization boundaries."""
    return name in instance.__dict__["_explicit_parameters"]
