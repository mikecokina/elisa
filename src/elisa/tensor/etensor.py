from __future__ import annotations

from typing import TYPE_CHECKING

from elisa import settings

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from elisa.types import Float

if settings.CUDA:
    # noinspection PyUnresolvedReferences
    import cupy as cp


    class CupyTensor(cp.ndarray):
        """GPU-backed tensor based on CuPy ndarray.

        This class wraps :class:`cupy.ndarray` to provide a consistent tensor
        interface across CPU and GPU implementations.
        """

        def __new__(
                cls,
                input_array: ArrayLike,
                dtype: str = "float32",
        ) -> cp.ndarray:
            """Create a new GPU tensor from input data.

            :param input_array: Input array-like data.
            :param dtype: Data type of the resulting tensor.
            :returns: GPU tensor view of the input data.
            """
            return cp.asarray(input_array, dtype=dtype).view(cls)


    Tensor = CupyTensor

else:
    import numpy as np


    class NumpyTensor(np.ndarray):
        """CPU-backed tensor based on NumPy ndarray.

        This class wraps :class:`numpy.ndarray` to provide a consistent tensor
        interface across CPU and GPU implementations.
        """

        def __new__(
                cls,
                input_array: ArrayLike,
                dtype: str = "float32",
        ) -> NDArray:
            """Create a new CPU tensor from input data.

            :param input_array: Input array-like data.
            :param dtype: Data type of the resulting tensor.
            :returns: CPU tensor view of the input data.
            """
            return np.asarray(input_array, dtype=dtype).view(cls)

        def get(self) -> NDArray[Float]:
            """Return a NumPy representation of the tensor.

            :returns: NumPy array copy of the tensor data.
            """
            return np.asarray(self)


    Tensor = NumpyTensor

__all__ = ("Tensor",)
