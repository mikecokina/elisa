from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, TypedDict

import numpy as np
from astropy.units import Quantity, Unit

if TYPE_CHECKING:
    from numpy.typing import NDArray

Float: TypeAlias = float | np.float32 | np.float64 | np.floating
Int: TypeAlias = int | np.int32 | np.int64 | np.integer
Number: TypeAlias = Float | Int
NumpyBool: TypeAlias = bool | np.bool_

UnitType: TypeAlias = Unit

ComponentName: TypeAlias = Literal["primary", "secondary"]
ComponentSelection: TypeAlias = Literal["primary", "secondary", "all", "both"]


AstropyQuantity: TypeAlias = Quantity
AstropyUnit: TypeAlias = Unit

class HasMeshData(Protocol):
    points: NDArray[Float]
    faces: NDArray[np.integer]
    indices: NDArray[np.integer]


class ZeroPointType(TypedDict):
    system: str
    unit: UnitType
    fluxes: dict[str, Float]
    reference_magnitudes: dict[str, Float]


# Define a minimal Protocol for 3D Axes exposing only the methods used
# in this module. Defining it at runtime (it's lightweight) helps IDEs
# and static checkers recognise the 3D axis methods used here.
class Axes3DProtocol(Protocol):
    def get_xlim3d(self) -> tuple[float, float]: ...

    def get_ylim3d(self) -> tuple[float, float]: ...

    def get_zlim3d(self) -> tuple[float, float]: ...

    def set_xlim3d(self, limits: tuple[float, float]) -> None: ...

    def set_ylim3d(self, limits: tuple[float, float]) -> None: ...

    def set_zlim3d(self, limits: tuple[float, float]) -> None: ...
