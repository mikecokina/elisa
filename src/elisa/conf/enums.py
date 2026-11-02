"""Named configuration, fitting and computation choices for ELISA."""

from enum import Enum, StrEnum


class BinaryComponent(StrEnum):
    """Select a binary component while accepting existing string selectors."""

    PRIMARY = "primary"
    SECONDARY = "secondary"


class BinaryMorphology(StrEnum):
    """Name the binary surface geometries supported by synthesis."""

    DETACHED = "detached"
    OVER_CONTACT = "over-contact"
    SEMI_DETACHED = "semi-detached"
    DOUBLE_CONTACT = "double-contact"


class CurveType(Enum):
    """Select the observable routed by binary curve synthesis."""

    LC = "lc"
    RV = "rv"


class RVMethod(StrEnum):
    """Select RV computation while accepting existing string configuration values."""

    KINEMATIC = "kinematic"
    RADIOMETRIC = "radiometric"


class MeanAnomalySolver(StrEnum):
    """Select the secant algorithm used for orbital mean anomalies."""

    VECTORIZED = "vectorized"
    SCALAR = "scalar"


class LoggingProfile(StrEnum):
    """Built-in logging policy selectors; custom paths remain plain strings."""

    DEFAULT = "default"
    FIT = "fit"


class SamplingMethod(StrEnum):
    """Select how LC synthesis support is prepared during fitter setup.

    UNIFORM spaces support points evenly in orbital phase. ADAPTIVE spaces them
    along the initial synthetic curve to resolve eclipses. Manual phase arrays
    are supplied directly instead of using an enum member. String values retain
    compatibility with existing fitting arguments and JSON configurations.
    """

    UNIFORM = "uniform"
    ADAPTIVE = "adaptive"


class GeometryUpdateTarget(StrEnum):
    """Select whether internal geometry reuse tracks the object or its spots."""

    OBJECT = "object"
    SPOT = "spot"


class OrbitalSortKey(StrEnum):
    """Select the orbital-supplement column used for sorting."""

    INDEX = "index"
    DISTANCE = "distance"
    RADIUS = "radius"


class NormalizationKind(StrEnum):
    """Select light-curve normalization while accepting existing string inputs."""

    AVERAGE = "average"
    GLOBAL_AVERAGE = "global_average"
    MAXIMUM = "maximum"
    GLOBAL_MAXIMUM = "global_maximum"
    MINIMUM = "minimum"


class DiscardUnit(StrEnum):
    """Select complete ensemble steps or individual flat rows for chain discard.

    String values remain compatible with existing chain-loading calls. Omitting
    the unit selects steps for structured chains and rows for legacy flat files.
    """

    STEPS = "steps"
    ROWS = "rows"


class IntersectionBackend(StrEnum):
    """Select the polygon/triangle intersection-area implementation."""

    PYPEX = "pypex"
    REFERENCE = "reference"
    SUTHERLAND_HODGMAN = "sutherland_hodgman"
