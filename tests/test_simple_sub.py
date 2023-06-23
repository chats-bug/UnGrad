import unittest
import numpy as np

from autograd.tensor import Tensor, sub


class TestTensorSub(unittest.TestCase):
    def test_simple_sub(self):
        t1 = Tensor(np.random.randint(0, 10, size=3), requires_grad=True)
        t2 = Tensor(np.random.randint(0, 10, size=3), requires_grad=True)
        w = np.random.randint(0, 10, size=3)

        t3 = t1 - t2
        t3.backward(Tensor(w))

        assert t3.data.tolist() == (t1.data - t2.data).tolist()

        assert t1.grad.data.tolist() == w.tolist()
        assert t2.grad.data.tolist() == (-w).tolist()

    def test_broadcast_sub(self):
        # It is obvious that t1.shape == t2.shape to sub the tensors
        # But we are also allowed to "broadcast" a shape
        # i.e. (1, 1, 1, 1, 10, 5) + (10, 5) is allowed
        #
        # Also t1.shape = (10, 5) and t2.shape = (5,)
        # then expand t2.shape => (1, 5)
        # And sub t2 to every row of t1

        t1 = Tensor(np.random.randint(0, 10, size=(2, 3)), requires_grad=True)
        t2 = Tensor(np.random.randint(0, 10, size=3), requires_grad=True)
        w = np.random.randint(0, 10, size=(2, 3))

        t3 = t1 - t2
        t3.backward(Tensor(w))

        assert t3.data.tolist() == (t1.data - t2.data).tolist()

        assert t1.grad.data.tolist() == w.tolist()
        assert t2.grad.data.tolist() == (-w.sum(axis=0)).tolist()

    def test_broadcast_sub2(self):
        # Shapes (2, 3) and (1, 3)

        t1 = Tensor(np.random.randint(0, 10, size=(2, 3)), requires_grad=True)
        t2 = Tensor(np.random.randint(0, 10, size=(1, 3)), requires_grad=True)
        w = np.random.randint(0, 10, size=(2, 3))

        t3 = t1 - t2
        t3.backward(Tensor(w))

        assert t3.data.tolist() == (t1.data - t2.data).tolist()

        assert t1.grad.data.tolist() == w.tolist()
        assert t2.grad.data.tolist() == (-w.sum(axis=0, keepdims=True)).tolist()
