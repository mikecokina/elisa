"""Named configuration, fitting and computation choices for ELISA."""

from enum import Enum


class CurveType(Enum):
    """Select the observable routed by binary curve synthesis."""

    LC = "lc"
    RV = "rv"


class MeanAnomalySolver(str, Enum):
    """Select the secant algorithm used for orbital mean anomalies."""

    VECTORIZED = "vectorized"
    SCALAR = "scalar"

    def __str__(self) -> str:
        """Return the configuration value for INI serialization and logging."""
        return self.value


class LoggingProfile(str, Enum):
    """Built-in logging policy selectors; custom paths remain plain strings."""

    DEFAULT = "default"
    FIT = "fit"


class SamplingMethod(str, Enum):
    """Select how LC synthesis support is prepared during fitter setup.

    UNIFORM spaces support points evenly in orbital phase. ADAPTIVE spaces them
    along the initial synthetic curve to resolve eclipses. Manual phase arrays
    are supplied directly instead of using an enum member. String values retain
    compatibility with existing fitting arguments and JSON configurations.
    """

    UNIFORM = "uniform"
    ADAPTIVE = "adaptive"

    def __str__(self) -> str:
        """Return the serialized sampling value for configuration and logging."""
        return self.value


class DiscardUnit(str, Enum):
    """Select complete ensemble steps or individual flat rows for chain discard.

    String values remain compatible with existing chain-loading calls. Omitting
    the unit selects steps for structured chains and rows for legacy flat files.
    """

    STEPS = "steps"
    ROWS = "rows"

    def __str__(self) -> str:
        """Return the serialized discard unit for configuration and logging."""
        return self.value


class IntersectionBackend(str, Enum):
    """Select the polygon/triangle intersection-area implementation."""

    PYPEX = "pypex"
    REFERENCE = "reference"
    SUTHERLAND_HODGMAN = "sutherland_hodgman"

    def __str__(self) -> str:
        """Return the backend value for configuration and logging."""
        return self.value
