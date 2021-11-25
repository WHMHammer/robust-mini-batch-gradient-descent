import numpy as np

from .regularization import Regularization
from .loss import Loss


class MiniBatchGradientDescent:
    def __init__(
        self,
        regularization: Regularization,
        regularization_weight: float,
        loss: Loss,
        learning_rate: float,
        batch_size: int,
        max_iter: int
    ):
        self.regularization: Regularization = regularization
        self.regularization_weight: float = regularization_weight
        self.loss: Loss = loss
        self.learning_rate: float = learning_rate,
        self.batch_size: int = batch_size
        self.max_iter: int = max_iter
        self.rng = np.random.default_rng()

    def fit(self, X: np.ndarray, y: np.ndarray):
        sample_size, dimension = X.shape
        X = np.c_[np.ones(sample_size), X]
        self.w = np.ones(dimension + 1)
        indices = self.rng.choice(sample_size, self.batch_size, False)
        prev_loss, gradient = self.loss(
            X[indices],
            self.w,
            y[indices]
        ) + self.regularization_weight * self.regularization(self.w)
        for _ in range(self.max_iter):
            self.w -= self.learning_rate * gradient
            indices = self.rng.choice(sample_size, self.batch_size, False)
            loss, gradient = self.loss(
                X[indices],
                self.w, y[indices]
            ) + self.regularization_weight * self.regularization(self.w)
            if abs(loss - prev_loss) / prev_loss < 1e-5:
                return
            prev_loss = loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.c_[np.ones(X.shape[0]), X].dot(self.w)
