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


class MLP:
    def __init__(self, layer_sizes: list[int]):
        self.layers = []
        n = len(layer_sizes)
        for i in range(n - 1):
            self.layers.append(Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < n - 2:
                self.layers.append(ReLU())

        self.loss_fn = MSELoss()
        self._last_output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        self._last_output = out
        return out

    def loss(self, y_true: np.ndarray) -> float:
        return self.loss_fn.forward(self._last_output, y_true)

    def backward(self):
        grad = self.loss_fn.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def zero_grad(self):
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.zero_grad()

    def update(self,
               lr: float = 1e-3,
               batch_size: int = 1,
               l2_lambda: float = 0.0,
               grad_clip: float | None = None):

        for layer in self.layers:
            if not isinstance(layer, Linear):
                continue

            gW = layer.grad_W / batch_size
            gb = layer.grad_b / batch_size

            if grad_clip is not None:
                norm = np.linalg.norm(gW)
                if norm > grad_clip:
                    gW = gW * grad_clip / norm

            layer.W -= lr * (gW + l2_lambda * layer.W)
            layer.b -= lr * gb

        self.zero_grad()

    def __repr__(self):
        body = " → ".join(str(l) for l in self.layers)
        return f"MLP({body})"

    def param_count(self) -> int:
        total = 0
        for l in self.layers:
            if isinstance(l, Linear):
                total += l.W.size + l.b.size
        return total

if __name__ == "__main__":
    np.random.seed(0)
    print("=== Smoke-test MLP ===\n")

    mlp = MLP([4, 16, 8, 4])
    print(mlp)
    print(f"Параметров: {mlp.param_count()}\n")

    x = np.random.randn(4)
    y_true = np.array([1.0, 6.0, 2.5, 3.0])

    # one training step
    y_pred = mlp.forward(x)
    L = mlp.loss(y_true)
    mlp.backward()
    mlp.update(lr=0.01)

    print(f"Input:  {x}")
    print(f"Pred:   {y_pred}")
    print(f"Target: {y_true}")
    print(f"Loss:   {L:.4f}")

    losses = []
    for _ in range(200):
        y_pred = mlp.forward(x)
        L = mlp.loss(y_true)
        mlp.backward()
        mlp.update(lr=0.01)
        losses.append(L)

    print(f"\nLoss после 200 итераций: {losses[-1]:.6f}")
    print(f"Уменьшился: {losses[0] > losses[-1]}")
