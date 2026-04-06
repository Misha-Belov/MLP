
import numpy as np
import copy



class Linear:

    def __init__(self, in_f, out_f):
        scale = np.sqrt(2.0 / in_f)
        self.W = np.random.randn(out_f, in_f) * scale
        self.b = np.zeros(out_f)
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)
        self._x = None

    def forward(self, x):
        self._x = x.copy()
        return self.W @ x + self.b

    def backward(self, g):
        self.grad_W += np.outer(g, self._x)
        self.grad_b += g
        return self.W.T @ g

    def zero_grad(self):
        self.grad_W[:] = 0.0
        self.grad_b[:] = 0.0

    def __repr__(self):
        return f"Linear({self.W.shape[1]} -> {self.W.shape[0]})"


class ReLU:
    def __init__(self):
        self._mask = None

    def forward(self, x):
        self._mask = x > 0
        return x * self._mask

    def backward(self, g):
        return g * self._mask

    def __repr__(self):
        return "ReLU"


class MSELoss:
    def __init__(self):
        self._diff = None
        self._n = None

    def forward(self, y_pred, y_true):
        self._diff = y_pred - y_true
        self._n = len(y_pred)
        return float(np.mean(self._diff ** 2))

    def backward(self):
        return 2.0 * self._diff / self._n


class MLP:
    def __init__(self, sizes):
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                self.layers.append(ReLU())
        self.loss_fn = MSELoss()
        self._out = None

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        self._out = out
        return out

    def loss(self, y):
        return self.loss_fn.forward(self._out, y)

    def backward(self):
        g = self.loss_fn.backward()
        for layer in reversed(self.layers):
            g = layer.backward(g)

    def zero_grad(self):
        for l in self.layers:
            if isinstance(l, Linear):
                l.zero_grad()

    def _linear_layers(self):
        return [l for l in self.layers if isinstance(l, Linear)]

    def param_count(self):
        return sum(l.W.size + l.b.size for l in self._linear_layers())

    def __repr__(self):
        return "MLP(" + " -> ".join(str(l) for l in self.layers) + ")"


class StandardScaler:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def inverse_transform(self, X):
        return X * self.std_ + self.mean_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def global_grad_norm(mlp):
    sq = sum(
        np.sum(l.grad_W ** 2) + np.sum(l.grad_b ** 2)
        for l in mlp._linear_layers()
    )
    return float(np.sqrt(sq))


def clip_grads(mlp, max_norm):
    norm = global_grad_norm(mlp)
    if norm > max_norm:
        scale = max_norm / (norm + 1e-8)
        for l in mlp._linear_layers():
            l.grad_W *= scale
            l.grad_b *= scale
    return norm


def sgd_step(mlp, lr, l2=0.0):
    for l in mlp._linear_layers():
        l.W -= lr * (l.grad_W + l2 * l.W)
        l.b -= lr * l.grad_b
    mlp.zero_grad()


def save_weights(mlp):
    return [(l.W.copy(), l.b.copy()) for l in mlp._linear_layers()]


def load_weights(mlp, weights):
    for (W, b), l in zip(weights, mlp._linear_layers()):
        l.W[:] = W
        l.b[:] = b


def target_fn(x):
    """f(x1, x2, x3, x4) = (x1^2, 3*x2, 5*x4 - x3, 3)^T"""
    x1, x2, x3, x4 = x
    return np.array([x1 ** 2, 3.0 * x2, 5.0 * x4 - x3, 3.0])


def make_dataset(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, (n, 4))
    Y = np.stack([target_fn(xi) for xi in X])
    return X, Y


def split(X, Y, test_ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_ratio)
    return X[idx[n_test:]], Y[idx[n_test:]], X[idx[:n_test]], Y[idx[:n_test]]


def train(mlp, X_tr, Y_tr, X_val, Y_val,
          num_epochs=200, batch_size=64, lr=8e-3,
          lr_decay=0.98, l2=5e-5, grad_clip=5.0,
          patience=25, seed=0):

    rng = np.random.default_rng(seed)
    history = {"train": [], "val": [], "lr": [], "gnorm": []}

    best_val = np.inf
    best_w = None
    no_imp = 0
    cur_lr = lr

    for epoch in range(1, num_epochs + 1):
        idx = rng.permutation(len(X_tr))
        X_s, Y_s = X_tr[idx], Y_tr[idx]

        ep_loss, ep_norm, n_b = 0.0, 0.0, 0

        for s in range(0, len(X_tr), batch_size):
            Xb = X_s[s:s + batch_size]
            Yb = Y_s[s:s + batch_size]
            bs = len(Xb)

            batch_loss = 0.0
            for x, y in zip(Xb, Yb):
                mlp.forward(x)
                batch_loss += mlp.loss(y)
                mlp.backward()

            for l in mlp._linear_layers():
                l.grad_W /= bs
                l.grad_b /= bs

            norm = clip_grads(mlp, grad_clip)
            sgd_step(mlp, cur_lr, l2)

            ep_loss += batch_loss / bs
            ep_norm += norm
            n_b += 1

        train_loss = ep_loss / n_b
        avg_norm = ep_norm / n_b

        val_loss = float(np.mean([
            mlp.loss_fn.forward(mlp.forward(x), y)
            for x, y in zip(X_val, Y_val)
        ]))

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        history["lr"].append(cur_lr)
        history["gnorm"].append(avg_norm)

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_w = save_weights(mlp)
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"Early stopping at epoch {epoch},  best val loss: {best_val:.5f}")
                break

        cur_lr *= lr_decay

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}  train={train_loss:.5f}  "
                  f"val={val_loss:.5f}  lr={cur_lr:.2e}  |grad|={avg_norm:.3f}")

    if best_w:
        load_weights(mlp, best_w)
        print(f"Best weights restored (val={best_val:.5f})")

    return history



def main():
    np.random.seed(42)

    X_raw, Y_raw = make_dataset(n=6000, seed=42)
    X_tr_r, Y_tr_r, X_v_r, Y_v_r = split(X_raw, Y_raw)

    print(f"Dataset: {len(X_raw)} samples  (train={len(X_tr_r)}, val={len(X_v_r)})")

    sx = StandardScaler().fit(X_tr_r)
    sy = StandardScaler().fit(Y_tr_r)

    X_tr = sx.transform(X_tr_r)
    Y_tr = sy.transform(Y_tr_r)
    X_v  = sx.transform(X_v_r)
    Y_v  = sy.transform(Y_v_r)

    mlp = MLP([4, 128, 64, 32, 4])
    print(f"Architecture: {mlp}")
    print(f"Parameters:   {mlp.param_count():,}\n")

    history = train(
        mlp, X_tr, Y_tr, X_v, Y_v,
        num_epochs=300,
        batch_size=64,
        lr=8e-3,
        lr_decay=0.98,
        l2=5e-5,
        grad_clip=5.0,
        patience=30,
    )

    preds_n = np.stack([mlp.forward(x) for x in X_v])
    preds   = sy.inverse_transform(preds_n)

    mse = float(np.mean((preds - Y_v_r) ** 2))
    mae = float(np.mean(np.abs(preds - Y_v_r)))
    r2  = 1 - np.sum((preds - Y_v_r) ** 2) / (
              np.sum((Y_v_r - Y_v_r.mean(axis=0)) ** 2) + 1e-9)

    print(f"\nValidation results:")
    print(f"  MSE = {mse:.5f}")
    print(f"  MAE = {mae:.5f}")
    print(f"  R2  = {r2:.5f}")

    print("\nSample predictions:")
    print(f"  {'Predicted':45s}  True")
    for i in range(8):
        print(f"  {str(preds[i].round(3)):45s}  {Y_v_r[i].round(3)}")


if __name__ == "__main__":
    main()
