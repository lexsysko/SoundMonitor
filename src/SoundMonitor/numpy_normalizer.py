import numpy as np

import numpy as np


class NumpyNormalizer:
    def __init__(self, norm: str = "l2"):
        if norm not in ("l1", "l2", "max"):
            raise ValueError("norm must be 'l1', 'l2', or 'max'")
        self.norm = norm

    def fit(self, X: np.ndarray):
        # sklearn Normalizer doesn't compute global params, just stores the rule
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.norm == "l2":
            norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        elif self.norm == "l1":
            norms = np.sum(np.abs(X), axis=1, keepdims=True) + 1e-12
        elif self.norm == "max":
            norms = np.max(np.abs(X), axis=1, keepdims=True) + 1e-12
        return X / norms

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class NumpyStandardScaler:
    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0) if self.with_mean else np.zeros(X.shape[1])
        if self.with_std:
            std = X.std(axis=0)
            std[std == 0] = 1.0  # avoid div-by-zero for constant features
            self.scale_ = std
        else:
            self.scale_ = np.ones(X.shape[1])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fitted yet.")
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.scale_ + self.mean_
