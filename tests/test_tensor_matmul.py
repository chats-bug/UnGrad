import unittest
import numpy as np

from autograd.tensor import Tensor, mul


class TestTensorMatmul(unittest.TestCase):
	def test_simple_matmul(self):
		t1 = Tensor(np.random.randint(0, 10, size=(3, 2)), requires_grad=True)
		t2 = Tensor(np.random.randint(0, 10, size=(2, 4)), requires_grad=True)
		w = np.random.randint(0, 10, size=(3, 4))

		t3 = t1 @ t2
		t3.backward(Tensor(w))

		assert t3.data.tolist() == (t1.data @ t2.data).tolist()

		assert t1.grad.data.tolist() == (w @ t2.data.T).tolist()
		assert t2.grad.data.tolist() == (t1.data.T @ w).tolist()