from typing import NamedTuple, Callable, Optional, Union
import numpy as np


Arrayable = Union[float, list, np.ndarray]

def ensure_array(arrayable: Arrayable) -> np.ndarray:
	if isinstance(arrayable, np.ndarray):
		return arrayable
	return np.array(arrayable)


Tensorable = Union['Tensor', float, np.ndarray]

def ensure_tensor(tensorable: Tensorable) -> 'Tensor':
	if isinstance(tensorable, Tensor):
		return tensorable
	return Tensor(tensorable)


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
		self._data = ensure_array(data)
		self.requires_grad = requires_grad
		self.depends_on = depends_on
		self.shape = self._data.shape
		self.grad: Optional[Tensor] = None

		if self.requires_grad:
			self.zero_grad()

	@property
	def data(self) -> np.ndarray:
		return self._data
	
	@data.setter
	def data(self, new_data: Arrayable) -> None:
		self._data = ensure_array(new_data)
		# If we change the data, we need to invalidate the gradient
		self.grad = None

	def __repr__(self) -> str:
		return f"Tensor({self._data}), requires_grad=({self.requires_grad})"
	
	def __add__(self, other: 'Tensor') -> 'Tensor':
		return add(self, ensure_tensor(other))
	
	def __radd__(self, other: 'Tensor') -> 'Tensor':
		return add(self, ensure_tensor(other))
	
	def __iadd__(self, other: 'Tensor') -> 'Tensor':
		"""
		In-place add
		t += other
		"""
		self.data += ensure_tensor(other).data
		# In-place operations don't create new tensors
		# so we don't need to create a new tensor here
		return self
	
	def __mul__(self, other: 'Tensor') -> 'Tensor':
		return mul(self, ensure_tensor(other))
	
	def __rmul__(self, other: 'Tensor') -> 'Tensor':
		return mul(self, ensure_tensor(other))
	
	def __imul__(self, other: 'Tensor') -> 'Tensor':
		self.data *= ensure_tensor(other).data
		return self
	
	def __neg__(self) -> 'Tensor':
		return neg(self)
	
	def __sub__(self, other: 'Tensor') -> 'Tensor':
		return sub(self, ensure_tensor(other))
	
	def __rsub__(self, other: 'Tensor') -> 'Tensor':
		return sub(self, ensure_tensor(other))
	
	def __isub__(self, other: 'Tensor') -> 'Tensor':
		self.data = self.data - ensure_tensor(other).data
		return self
	
	def __matmul__(self, other: 'Tensor') -> 'Tensor':
		return matmul(self, other)
	
	def __getitem__(self, idxs) -> 'Tensor':
		return slice(self, idxs)
	
	def zero_grad(self) -> None:
		self.grad = Tensor(np.zeros_like(self.data, dtype=np.float64))
	
	def sum(self) -> 'Tensor':
		return tensor_sum(self)
	
	def backward(self, grad: Optional['Tensor'] = None) -> None:
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


def matmul(t1: Tensor, t2: Tensor) -> Tensor:
	"""
	If t3 = t1 @ t2, then the grads are as follows:
		grad_1 = grad_3 @ t2.T
		grad_2 = t1.T @ grad_3
	"""
	data = t1.data @ t2.data
	requires_grad = t1.requires_grad or t2.requires_grad
	depends_on: list[Dependency] = []

	if t1.requires_grad:
		def grad_fn1(grad: np.ndarray) -> np.ndarray:
			return grad @ t2.data.T
		
		depends_on.append(Dependency(t1, grad_fn1))

	if t2.requires_grad:
		def grad_fn2(grad: np.ndarray) -> np.ndarray:
			return t1.data.T @ grad
		
		depends_on.append(Dependency(t2, grad_fn2))

	return Tensor(data, requires_grad, depends_on)


def slice(t: Tensor, idx: slice) -> Tensor:
	data = t.data[idx]
	requires_grad = t.requires_grad
	depends_on: list[Dependency] = []

	if requires_grad:
		def grad_fn(grad: np.ndarray) -> np.ndarray:
			"""
			grad is necessarily a 0-tensor, so each element
			contributes that much
			"""
			b_grad = np.zeros_like(data)
			b_grad[idx] = grad
			return b_grad
		
		depends_on = [Dependency(t, grad_fn)]
	
	return Tensor(data, requires_grad, depends_on)