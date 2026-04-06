import numpy as np


class Linear:

    def __init__(self, in_features: int, out_features: int):
        scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(out_features, in_features) * scale
        self.b = np.zeros(out_features)

        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

        self._x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x.copy()
        return self.W @ x + self.b

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        self.grad_W += np.outer(grad_output, self._x)
        self.grad_b += grad_output
        return self.W.T @ grad_output

    def zero_grad(self):
        self.grad_W[:] = 0.0
        self.grad_b[:] = 0.0

    def __repr__(self):
        return f"Linear({self.W.shape[1]} → {self.W.shape[0]})"


class ReLU:

    def __init__(self):
        self._mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0
        return x * self._mask

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * self._mask

    def __repr__(self):
        return "ReLU()"


class MSELoss:

    def __init__(self):
        self._diff = None
        self._n = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self._diff = y_pred - y_true
        self._n = len(y_pred)
        return float(np.mean(self._diff ** 2))

    def backward(self) -> np.ndarray:
        return 2.0 * self._diff / self._n


if __name__ == "__main__":
    np.random.seed(42)
    print("=== Smoke-test слоёв ===\n")

    x = np.array([1.0, -2.0, 3.0, 0.5])

    lin = Linear(4, 3)
    y = lin.forward(x)
    print(f"Linear forward:   {y}")
    g = lin.backward(np.ones(3))
    print(f"Linear grad_W:\n{lin.grad_W}")
    print(f"Linear grad back: {g}\n")

    relu = ReLU()
    y = relu.forward(np.array([-1.0, 0.5, -0.3, 2.0]))
    print(f"ReLU forward:  {y}")
    g = relu.backward(np.ones(4))
    print(f"ReLU backward: {g}\n")

    loss_fn = MSELoss()
    pred = np.array([1.0, 2.0, 3.0])
    true = np.array([1.5, 1.5, 3.5])
    L = loss_fn.forward(pred, true)
    print(f"MSELoss: {L:.4f}")
    print(f"MSELoss grad: {loss_fn.backward()}")
