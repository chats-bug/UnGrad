from typing import NamedTuple, Callable, Optional
import numpy as np


Arrayable = float | list | np.ndarray


def ensure_array(arrayable: Arrayable) -> np.ndarray:
	if isinstance(arrayable, np.ndarray):
		return arrayable
	return np.array(arrayable)


class Dependency(NamedTuple):
	tensor: 'Tensor'
	grad_fn: Callable[[np.ndarray], np.ndarray]


class Tensor:
	def __init__(
			self,
			data: Arrayable,
			requires_grad: bool = False,
			depends_on: list[Dependency] = [],
	) -> None:
		self.data = ensure_array(data)
		self.requires_grad = requires_grad
		self.depends_on = depends_on
		self.shape = self.data.shape
		self.grad: Tensor | None = None

		if self.requires_grad:
			self.zero_grad()

	def __repr__(self) -> str:
		return f"Tensor({self.data}), requires_grad=({self.requires_grad})"
	
	def zero_grad(self) -> None:
		self.grad = Tensor(np.zeros_like(self.data))
	
	def sum(self) -> 'Tensor':
		return tensor_sum(self)
	
	def backward(self, grad: 'Tensor' = None) -> None:
		assert self.requires_grad, "called backward on non-requires-grad tensor"

		if grad is None:
			if self.shape == ():
				grad = Tensor(1)
			else:
				raise RuntimeError("grad must be a non-0-tensor")
		
		self.grad.data += grad.data # type: ignore

		for dependency in self.depends_on:
			backward_grad = dependency.grad_fn(grad.data)
			dependency.tensor.backward(Tensor(backward_grad))


def tensor_sum(t: Tensor) -> Tensor:
	"""
	Takes a tensor and returns a 0-tensor
	that's the sum of all of its elements
	"""
	data = t.data.sum()
	requires_grad = t.requires_grad
	depends_on = []

	if requires_grad:
		def grad_fn(grad: np.ndarray) -> np.ndarray:
			"""
			grad is necessarily a 0-tensor, so each element
			contributes that much
			"""
			return grad * np.ones_like(t.data)
		
		depends_on = [Dependency(t, grad_fn)]
		
	return Tensor(data, requires_grad, depends_on)


def add(t1: Tensor, t2: Tensor) -> Tensor:
	data = t1.data + t2.data
	requires_grad = t1.requires_grad or t2.requires_grad
	depends_on: list[Dependency] = []

	if t1.requires_grad:
		def grad_fn1(grad: np.ndarray) -> np.ndarray:
			# Handle broadcasting here
			ndims_added = grad.ndim - t1.data.ndim
			for _ in range(ndims_added):
				grad = grad.sum(axis=0)

			# Sum accross broadcasted but not added dims
			for i, dim in enumerate(t1.shape):
				if dim == 1:
					grad = grad.sum(axis=i, keepdims=True)

			return grad
		
		depends_on.append(Dependency(t1, grad_fn1))

	if t2.requires_grad:
		def grad_fn2(grad: np.ndarray) -> np.ndarray:
			ndims_added = grad.ndim - t2.data.ndim
			for _ in range(ndims_added):
				grad = grad.sum(axis=0)
			
			# Sum accross broadcasted but not added dims
			for i, dim in enumerate(t2.shape):
				if dim == 1:
					grad = grad.sum(axis=i, keepdims=True)
			
			return grad
		
		depends_on.append(Dependency(t2, grad_fn2))

	return Tensor(data, requires_grad, depends_on) 


def mul(t1: Tensor, t2: Tensor) -> Tensor:
	data = t1.data * t2.data
	requires_grad = t1.requires_grad or t2.requires_grad
	depends_on: list[Dependency] = []

	if t1.requires_grad:
		def grad_fn1(grad: np.ndarray) -> np.ndarray:
			grad = grad * t2.data

			# Handle broadcasting here
			ndims_added = grad.ndim - t1.data.ndim
			for _ in range(ndims_added):
				grad = grad.sum(axis=0)

			# Sum accross broadcasted but not added dims
			for i, dim in enumerate(t1.shape):
				if dim == 1:
					grad = grad.sum(axis=i, keepdims=True)

			return grad
		
		depends_on.append(Dependency(t1, grad_fn1))

	if t2.requires_grad:
		def grad_fn2(grad: np.ndarray) -> np.ndarray:
			grad = grad * t1.data

			# Handle broadcasting here
			ndims_added = grad.ndim - t2.data.ndim
			for _ in range(ndims_added):
				grad = grad.sum(axis=0)
			
			# Sum accross broadcasted but not added dims
			for i, dim in enumerate(t2.shape):
				if dim == 1:
					grad = grad.sum(axis=i, keepdims=True)
			
			return grad
		
		depends_on.append(Dependency(t2, grad_fn2))

	return Tensor(data, requires_grad, depends_on)


def neg(t: Tensor) -> Tensor:
	data = -t.data
	requires_grad = t.requires_grad
	depends_on = []
	if requires_grad:
		depends_on = [Dependency(t, lambda x: -x)]
	
	return Tensor(data, requires_grad, depends_on)


def sub(t1: Tensor, t2: Tensor) -> Tensor:
	return add(t1, neg(t2))