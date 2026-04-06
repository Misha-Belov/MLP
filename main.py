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
        return f"Linear({self.W.shape[1]} -> {self.W.shape[0]})"


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
        body = " -> ".join(str(l) for l in self.layers)
        return f"MLP({body})"

    def param_count(self) -> int:
        total = 0
        for l in self.layers:
            if isinstance(l, Linear):
                total += l.W.size + l.b.size
        return total


def target_fn(x: np.ndarray) -> np.ndarray:
    """f(x) = (x1^2, 3*x2, 5*x4 - x3, 3)"""
    x1, x2, x3, x4 = x
    return np.array([x1 ** 2, 3 * x2, 5 * x4 - x3, 3.0])


def make_dataset(n_samples: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n_samples, 4))
    Y = np.stack([target_fn(x) for x in X])
    return X, Y


def train_test_split(X, Y, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_ratio)
    return X[idx[n_test:]], Y[idx[n_test:]], X[idx[:n_test]], Y[idx[:n_test]]


def make_batches(X, Y, batch_size: int, rng):
    idx = rng.permutation(len(X))
    for start in range(0, len(X), batch_size):
        bi = idx[start:start + batch_size]
        yield X[bi], Y[bi]


def train(mlp: MLP,
          X_train, Y_train,
          X_val, Y_val,
          num_epochs: int = 50,
          batch_size: int = 32,
          lr: float = 1e-3,
          seed: int = 0) -> dict:

    rng = np.random.default_rng(seed)
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        n_batches = 0

        for X_batch, Y_batch in make_batches(X_train, Y_train, batch_size, rng):
            batch_loss = 0.0

            for x, y in zip(X_batch, Y_batch):
                mlp.forward(x)
                batch_loss += mlp.loss(y)
                mlp.backward()

            mlp.update(lr=lr, batch_size=len(X_batch))

            epoch_loss += batch_loss / len(X_batch)
            n_batches += 1

        val_loss = 0.0
        for x, y in zip(X_val, Y_val):
            y_pred = mlp.forward(x)
            val_loss += mlp.loss_fn.forward(y_pred, y)
        val_loss /= len(X_val)
        train_loss = epoch_loss / n_batches

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{num_epochs}  "
                  f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

    return history

def evaluate(mlp: MLP, X, Y):
    preds = np.stack([mlp.forward(x) for x in X])
    mse = np.mean((preds - Y) ** 2)
    mae = np.mean(np.abs(preds - Y))
    print(f"\nEvaluation:  MSE={mse:.5f}  MAE={mae:.5f}")

    for i in range(5):
        print(f"  pred={preds[i].round(3)}  true={Y[i].round(3)}")
    return mse, mae


if __name__ == "__main__":
    np.random.seed(7)

    print("Basic MLP training\n")

    X, Y = make_dataset(n_samples=3000)
    X_train, Y_train, X_val, Y_val = train_test_split(X, Y)

    print(f"Train: {len(X_train)}  Val: {len(Y_val)}")
    print(f"y-range: min={Y.min():.2f}  max={Y.max():.2f}\n")

    mlp = MLP([4, 64, 32, 16, 4])
    print(mlp, f"  ({mlp.param_count()} params)\n")

    history = train(mlp, X_train, Y_train, X_val, Y_val,
                    num_epochs=60, batch_size=32, lr=5e-3)

    evaluate(mlp, X_val, Y_val)

