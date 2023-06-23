import unittest
import numpy as np

from autograd.tensor import Tensor, mul


class TestTensorMul(unittest.TestCase):
    def test_simple_mul(self):
        t1 = Tensor(np.random.randint(0, 10, size=3), requires_grad=True)
        t2 = Tensor(np.random.randint(0, 10, size=3), requires_grad=True)
        w = np.random.randint(0, 10, size=3).tolist()

        t3 = t1 * t2
        t3.backward(Tensor(w))

        assert t3.data.tolist() == (t1.data * t2.data).tolist()

        assert t1.grad.data.tolist() == (t2.data * w).tolist()
        assert t2.grad.data.tolist() == (t1.data * w).tolist()

    def test_broadcast_mul(self):
        t1 = Tensor(np.random.randint(0, 10, size=(2, 3)), requires_grad=True)
        t2 = Tensor(np.random.randint(0, 10, size=3), requires_grad=True)
        w = np.random.randint(0, 10, size=(2, 3)).tolist()

        t3 = t1 * t2
        t3.backward(Tensor(w))

        assert t3.data.tolist() == (t1.data * t2.data).tolist()

        assert t1.grad.data.tolist() == (t2.data * w).tolist()
        assert t2.grad.data.tolist() == (t1.data * w).sum(axis=0).tolist()

    def test_broadcast_mul2(self):
        # Shapes (2, 3) and (1, 3)

        t1 = Tensor(np.random.randint(0, 10, size=(2, 3)), requires_grad=True)
        t2 = Tensor(np.random.randint(0, 10, size=(1, 3)), requires_grad=True)
        w = np.random.randint(0, 10, size=(2, 3)).tolist()

        t3 = t1 * t2
        t3.backward(Tensor(w))

        assert t3.data.tolist() == (t1.data * t2.data).tolist()

        assert t1.grad.data.tolist() == (t2.data * w).tolist()
        assert (
            t2.grad.data.tolist() == (t1.data * w).sum(axis=0, keepdims=True).tolist()
        )
