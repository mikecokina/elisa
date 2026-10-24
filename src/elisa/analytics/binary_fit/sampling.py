"""Named phase-sampling strategies shared by LC fitting methods."""
from enum import Enum


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
