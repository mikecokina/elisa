from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Float = float | np.float32 | np.float64 | np.floating
Int = int | np.int32 | np.int64 | np.integer
Number = Float | Int

Points3DList = NDArray[NDArray[Float]]
Points2DList = NDArray[NDArray[Float]]


class HasMeshData(Protocol):
    points: NDArray[Float]
    faces: NDArray[np.integer]
    indices: NDArray[np.integer]
