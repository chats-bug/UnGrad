"""
Use the `Tensor` library to do cool things
"""
import numpy as np
from autograd.tensor import Tensor


def minimize_function():
	x = Tensor(10*np.random.rand(30), requires_grad=True)

	# We want to minimize the function f(x) = x^2 + 2x + 1
	# We can do this by using gradient descent
	for i in range(100):
		y = (x*x + 2*x + 1).sum()
		y.backward()

		delta = x.grad.data * 0.025
		x -= delta

		print(f"EPOCH {i}: y = {y.data:.3f}")

		# ! Important to zero out the gradient
		x.zero_grad()


def learn_simple_function():
	n, d = 1000, 3
	X = Tensor(np.random.randn(n, d))
	coef = Tensor(np.random.randn(d))
	y = X @ coef + np.random.randint(0, 10) + np.random.randn(n)/100

	w = Tensor(np.random.randn(X.shape[1]), requires_grad=True)
	b = Tensor(np.random.randn(), requires_grad=True)

	num_epochs = 100
	batch_size = 32
	lr = 5e-5
	ga_steps = 10 # gradient accumulation steps
	
	num_steps = X.shape[0] // batch_size
	print(f"Number of steps: {num_steps}")
	print(f"Number of epochs: {num_epochs}")
	print(f"Number of gradient accumulation steps: {ga_steps}")

	for epoch in range(num_epochs):
		w.zero_grad()
		b.zero_grad()
		
		for steps in range(num_steps):
			inputs = X[steps*batch_size:(steps+1)*batch_size]
			y_preds = inputs @ w + b
			err = y_preds - y[steps*batch_size:(steps+1)*batch_size]
			loss = (err * err).sum()

			loss.backward()
			
			if (steps + 1) % ga_steps == 0 or steps == num_steps - 1:
				w -= w.grad * lr
				w.zero_grad()
				b -= b.grad * lr
				b.zero_grad()
				print(f"Epoch {epoch}: loss = {loss.data:.3f} -> Grad Accumulation Step {steps + 1}")
			else:
				print(f"Epoch {epoch}: loss = {loss.data:.3f}")

	print(f"\n--- Training Completed ---\n")
	print(f"Number of steps: {num_steps}")
	print(f"Number of epochs: {num_epochs}")
	print(f"Number of gradient accumulation steps: {ga_steps}")

	# Final loss
	y_preds = X @ w + b
	err = y_preds - y
	loss = (err * err).sum()
	print(f"Final loss: {loss.data:.3f}")


if __name__ == "__main__":
	learn_simple_function()
		
		

	