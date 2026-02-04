from .. import settings

if settings.CUDA:
    import cupy as cp

    class CupyTensor(cp.ndarray):
        def __new__(cls, input_array, dtype='float32'):
            obj = cp.asarray(input_array, dtype=dtype).view()
            return obj

    Tensor = CupyTensor

else:
    import numpy as np

    class NumpyTensor(np.ndarray):
        def __new__(cls, input_array, dtype='float32'):
            obj = np.asarray(input_array, dtype=dtype).view(cls)
            return obj

        def get(self):
            return np.asarray(self)

    Tensor = NumpyTensor

__all__ = 'Tensor',
