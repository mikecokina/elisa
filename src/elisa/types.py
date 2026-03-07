from __future__ import annotations

from typing import Protocol, TypeAlias

import numpy as np
from astropy.units import Unit
from numpy.typing import NDArray

Float: TypeAlias = float | np.float32 | np.float64 | np.floating
Int: TypeAlias = int | np.int32 | np.int64 | np.integer
Number: TypeAlias = Float | Int
NumpyBool: TypeAlias = bool | np.bool_

Points3DList: TypeAlias = NDArray
Points2DList: TypeAlias = NDArray
UnitType: TypeAlias = Unit

class HasMeshData(Protocol):
    points: NDArray[Float]
    faces: NDArray[np.integer]
    indices: NDArray[np.integer]
