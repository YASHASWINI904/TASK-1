"""Linear regression implemented from scratch with NumPy (no scikit-learn)."""
from __future__ import annotations
import numpy as np

FEATURES = [
    "area_sqft",
    "bedrooms",
    "locality_score",
    "house_age_years",
    "distance_to_facilities_km",
]

class LinearRegressionScratch:
    """Batch-gradient-descent linear regression with standardized features."""
    def __init__(self, learning_rate: float = 0.05, epochs: int = 7000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.mean_ = None
        self.std_ = None
        self.weights_ = None
        self.bias_ = 0.0
        self.loss_history_ = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        Xs = (X - self.mean_) / self.std_
        ym = y.mean()
        ys = y.std()
        if ys == 0:
            ys = 1.0
        yt = (y - ym) / ys
        self.weights_ = np.zeros(Xs.shape[1], dtype=float)
        self.bias_ = 0.0
        for epoch in range(self.epochs):
            pred = Xs @ self.weights_ + self.bias_
            error = pred - yt
            self.weights_ -= self.learning_rate * (Xs.T @ error) / len(Xs)
            self.bias_ -= self.learning_rate * error.mean()
            if epoch % 100 == 0:
                self.loss_history_.append(float(np.mean(error ** 2)))
        self.y_mean_ = float(ym)
        self.y_std_ = float(ys)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Xs = (X - self.mean_) / self.std_
        return (Xs @ self.weights_ + self.bias_) * self.y_std_ + self.y_mean_

    def coefficients_original_scale(self):
        coef = (self.weights_ * self.y_std_) / self.std_
        intercept = self.y_mean_ + self.y_std_ * self.bias_ - float(np.sum(coef * self.mean_))
        return coef, intercept

def train_test_split(X, y, test_size=0.2, random_state=42):
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(X))
    test_count = max(1, int(round(len(X) * test_size)))
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot else 0.0
    return {"mse": mse, "rmse": rmse, "r2": r2}
