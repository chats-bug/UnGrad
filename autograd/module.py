import inspect
from typing import Iterator

from autograd.parameter import Parameter
from autograd.tensor import Tensor


class Module:
    def parameters(self) -> Iterator[Parameter]:
        for _, value in inspect.getmembers(self):
            if isinstance(value, Parameter):
                yield value
            elif isinstance(value, Module):
                yield from value.parameters()

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()
