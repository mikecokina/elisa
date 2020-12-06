import numpy as np
from .. import settings

if settings.CUDA:
    import torch


class Tensor(object):
    def __init__(self, value):
        self.value = value
        self._is_pytorch = False

        if settings.CUDA:
            self._is_pytorch = True

            if not isinstance(self.value, torch.Tensor):
                self.value = torch.Tensor(self.value)
                self.value = self.to_cuda()

    def __sub__(self, other):
        return Tensor(self.value - other.value)

    def __truediv__(self, other):
        return Tensor(self.value / other.value)

    def __add__(self, other):
        return Tensor(self.value + other.value)

    def __mul__(self, other):
        return Tensor(self.value * other.value)

    def to_cpu(self):
        if self._is_pytorch:
            self.value = self.value.to("cpu")
        return self.value

    def to_cuda(self):
        if self._is_pytorch:
            self.value = self.value.to("cuda")
        return self.value

    def to_pytorch(self):
        pass

    def to_ndarray(self):
        if self._is_pytorch:
            self.to_cpu()
        return np.array(self.value)

    @property
    def T(self):
        return Tensor(self.value.T)
