from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Float = float | np.float32 | np.float64
Int = int | np.int32 | np.int64
Number = Float | Int

Points3DList = NDArray[NDArray[Float, Float, Float]]
Points2DList = NDArray[NDArray[Float, Float]]


class HasMeshData(Protocol):
    points: NDArray[Float]
    faces: NDArray[np.integer]
    indices: NDArray[np.integer]
