"""
Optimizers are used to adjust the parameters of a neural network.
Popular optimizers include SGD, Adam, and RMSProp.
"""
import numpy as np

from autograd.module import Module
from autograd.tensor import Tensor



class Optimizer:
	def step(self, module: Module):
		raise NotImplementedError


class SGD(Optimizer):
	def __init__(self, lr=0.001) -> None:
		self.lr = lr

	def step(self, module: Module):
		for param in module.parameters():
			param -= param.grad * self.lr



# ERROR: DO NOT USE THIS UNTIL FIXED
class Adam(Optimizer):
	def __init__(self, lr=0.001) -> None:
		self.m = None
		self.v = None
		self.t = 0
		self.lr = lr

	def step(self, module: Module):
		# TODO: Fix the implementation
		# There is some problem with the shape broadcasting
		for param in module.parameters():
			if self.m is None:
				self.m = Tensor(np.zeros_like(param.data))
				self.v = Tensor(np.zeros_like(param.data))

			self.t += 1
			self.m = self.m * 0.9 + param.grad * (1 - 0.9)
			self.v = self.v * 0.999 + param.grad * param.grad * (1 - 0.999)

			m_hat = self.m / (1 - 0.9 ** self.t)
			v_hat = self.v / (1 - 0.999 ** self.t)

			param -= self.lr * m_hat / (v_hat**0.5 + 1e-8)

