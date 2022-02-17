import numpy as np
from .. import settings

if settings.CUDA:
    import cupy as cp


class Tensor(object):
    def __init__(self, value, dtype='float32'):
        self.value = value
        self._dtype = dtype
        self._is_cuda = False

        if settings.CUDA:
            self._is_cuda = True

            if not isinstance(self.value, cp.ndarray):
                self.value = cp.array(self.value, dtype=self._dtype)

    def __sub__(self, other):
        return Tensor(self.value - other.value)

    def __truediv__(self, other):
        return Tensor(self.value / other.value)

    def __add__(self, other):
        return Tensor(self.value + other.value)

    def __mul__(self, other):
        return Tensor(self.value * other.value)

    def __copy__(self):
        return Tensor(self.value.copy())

    def copy(self):
        return self.__copy__()

    def to_ndarray(self):
        if self._is_cuda:
            return np.array(self.value.get())
        return self.value

    @property
    def ndim(self):
        return self.value.ndim

    def __getitem__(self, item):
        return Tensor(self.value[item])

    @property
    def T(self):
        return Tensor(self.value.T)
