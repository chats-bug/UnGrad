"""
Use the `Tensor` library to minimize the function f(x) = x^2 + 2x + 1
"""
import numpy as np

from autograd.tensor import Tensor, tensor_sum, mul, add


x = Tensor([-2, -3, 0, 10, -12, 6, 1, 9], requires_grad=True)

# We want to minimize the function f(x) = x^2 + 2x + 1
# We can do this by using gradient descent
for i in range(100):
	y = tensor_sum(
		add(
			add(
				mul(x, x), 
				mul(x, Tensor(2))
				), 
			Tensor(1)
			)
		)
	print(f"EPOCH {i}: y = {y.data.sum(axis=0):.3f}")

	y.backward()

	delta = x.grad.data * 0.025
	x = Tensor(x.data - delta.data, requires_grad=True)

	# ! Important to zero out the gradient
	x.grad.data = 0