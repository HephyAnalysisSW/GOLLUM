import numpy as np
from types import SimpleNamespace

def generate_overlapping_gaussians(n_per_class=100, d=3, separation=1.0, rng=None, weight_mode="uniform"):
    rng = np.random.default_rng(rng)
    mu0 = np.zeros(d)
    mu1 = np.zeros(d)
    mu1[0] = separation
    mu1[1] = 0.5 * separation
    cov = np.eye(d)

    X0 = rng.multivariate_normal(mu0, cov, size=n_per_class)
    X1 = rng.multivariate_normal(mu1, cov, size=n_per_class)
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)])

    if weight_mode == "uniform":
        w = np.ones(2 * n_per_class)
    elif weight_mode == "random":
        w = rng.uniform(0.5, 2.0, size=2 * n_per_class)
    elif weight_mode == "class_imbalance":
        w = np.concatenate([np.ones(n_per_class), np.full(n_per_class, 2.0)])
    else:
        raise ValueError(f"Unknown weight_mode: {weight_mode}")

    return X, y, w

class FakeLoader:
    def __init__(self, X, y, w, batch_size=None):
        self.X = X
        self.y = y
        self.w = w
        self.N = len(y)
        self.batch_size = self.N if batch_size is None else batch_size

    def __iter__(self):
        for start in range(0, self.N, self.batch_size):
            stop = min(start + self.batch_size, self.N)
            indices = np.arange(start, stop)
            yield indices

    def split(self, indices):
        return self.X[indices], self.w[indices], self.y[indices]

def load_training_data(small=False, d=3, rng=None, batch_size=None):
    n_per_class=10**5 if not small else 2*10**3
    X, y, w = generate_overlapping_gaussians(n_per_class=n_per_class, d=d, rng=rng, weight_mode="uniform")
    weight_sums = [
        np.sum(w[y == 0]),  # class 0
        np.sum(w[y == 1]),  # class 1
        0.0,                # placeholder for class 2
        0.0                 # placeholder for class 3
    ]
    loader = FakeLoader(X, y, w, batch_size=None)

    return {
        "loader": loader,
        "weight_sums": weight_sums,
        "X_mean": X.mean(axis=0),
        "X_std": np.sqrt(X.var(axis=0)),
    }
