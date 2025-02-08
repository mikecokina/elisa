import numpy as np
from packaging import version

INT = int
FLOAT = float
BOOL = bool
COMPLEX = complex

if version.parse(np.__version__) < version.parse("1.20.0"):
    # noinspection PyUnresolvedReferences
    INT = np.int
    # noinspection PyUnresolvedReferences
    FLOAT = np.float
    # noinspection PyUnresolvedReferences
    BOOL = np.bool
    # noinspection PyUnresolvedReferences
    COMPLEX = np.complex
