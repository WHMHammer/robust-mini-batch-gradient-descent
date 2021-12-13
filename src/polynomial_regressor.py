import numpy as np
from statistics import stdev

from .preprocessor import Preprocessor
from .regularization import Regularization
from .loss import Loss
from .mini_batch_gradient_descent import MiniBatchGradientDescent


def power_expand(x: np.ndarray, power: int) -> np.ndarray:
    # | x0 |    | x0 x0^2 ... |
    # | x1 | => | x1 x1^2 ... |
    # | .. |    | .. .... ... |
    X = np.empty((x.shape[0], power))
    X[:, 0] = x
    for i in range(1, power):
        X[:, i] = x ** (i + 1)
    return X


class PolynomialRegressor:
    def __init__(
        self,
        preprocessor: Preprocessor,
        power: int,
        regularization: Regularization,
        regularization_weight: float,
        loss: Loss,
        learning_rate: float,
        batch_size: int,
        max_iter: int
    ):
        self.preprocessor: Preprocessor = preprocessor
        self.power: int = power
        self.model: MiniBatchGradientDescent = MiniBatchGradientDescent(
            regularization,
            regularization_weight,
            loss,
            learning_rate,
            batch_size,
            max_iter
        )

    def fit(self, x: np.ndarray, y: np.ndarray):
        x, y = self.preprocessor(x, y)
        self.model.fit(power_expand(x, self.power), y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(power_expand(x, self.power))


def generate_random_weights(power: int, w_low: float, w_high: float) -> np.ndarray:
    rng = np.random.default_rng()
    return rng.uniform(w_low, w_high, power + 1)


def generate_random_samples(
    w: np.ndarray,
    x_low: float,
    x_high: float,
    noise_level: float,
    sample_size: int
):
    # return x and y, respectively
    rng = np.random.default_rng()
    x = rng.uniform(x_low, x_high, sample_size)
    y = np.c_[np.ones(sample_size), power_expand(x, w.shape[0] - 1)].dot(w)
    y += rng.normal(0, stdev(y) * noise_level, sample_size)
    return x, y

def generate_incomplete_samples(w: np.ndarray,x_low: float,x_high: float,noise_level: float,
                                x_begin:float, x_end:float, sample_size: int):
    rng = np.random.default_rng()
    x1 = rng.uniform(x_low, x_low + (x_high - x_low) * x_begin, sample_size // 2)
    x2 = rng.uniform(x_low + (x_high - x_low) * x_end, x_high, sample_size // 2)
    x = np.append(x1, x2)
    y = np.c_[np.ones(sample_size), power_expand(x, w.shape[0] - 1)].dot(w)
    y += rng.normal(0, stdev(y) * noise_level, sample_size)
    return x, y

def generate_dense_samples(w: np.ndarray, x_low: float, x_high: float, noise_level: float,
                           x_begin:float, x_end:float, sample_size: int):
    rng = np.random.default_rng()
    x1 = rng.uniform(x_low+(x_high-x_low)*x_begin, x_low + (x_high - x_low) * x_end, sample_size//2)
    x2 = rng.uniform(x_low, x_high, sample_size // 2)
    x = np.append(x1, x2)
    y = np.c_[np.ones(sample_size), power_expand(x, w.shape[0] - 1)].dot(w)
    y += rng.normal(0, stdev(y) * noise_level, sample_size)
    return x, y
