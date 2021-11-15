import numpy as np
from math import ceil
from matplotlib import pyplot as plt
from statistics import stdev
from typing import Tuple


def trimmed_squared_loss_and_gradient(X: np.ndarray, w: np.ndarray, y: np.ndarray, epsilon: float = 0) -> Tuple[float, np.ndarray]:
    # return loss and gradient, respectively
    kept_size = ceil(X.shape[0] * (1 - epsilon))
    residuals = X.dot(w) - y
    losses = np.square(residuals)
    kept_indices = np.argsort(losses)[:kept_size]
    loss = sum(losses[kept_indices]) / kept_size
    gradient = X[kept_indices].transpose().dot(
        residuals[kept_indices]) / kept_size * 2
    return loss, gradient


def mini_batch_gradient_descent(X: np.ndarray, y: np.ndarray, epsilon: float = 0, learning_rate: float = 0.1, batch_size: int = 0, max_iter: int = 1000) -> np.ndarray:
    # return the estimated weight vector, with the bias being the 0th element
    sample_size, dimension = X.shape
    rng = np.random.default_rng()
    X = np.c_[np.ones(sample_size), X]
    w = np.full(dimension + 1, 0.5)
    if batch_size == 0:
        prev_loss, gradient = trimmed_squared_loss_and_gradient(
            X, w, y, epsilon)
    else:
        indices = rng.choice(sample_size, batch_size, False)
        prev_loss, gradient = trimmed_squared_loss_and_gradient(
            X[indices], w, y[indices], epsilon)
    for i in range(max_iter):
        w -= gradient * learning_rate
        if batch_size == 0:
            loss, gradient = trimmed_squared_loss_and_gradient(
                X, w, y, epsilon)
        else:
            indices = rng.choice(sample_size, batch_size, False)
            loss, gradient = trimmed_squared_loss_and_gradient(
                X[indices], w, y[indices], epsilon)
        if loss == prev_loss:
            return w
        prev_loss = loss
    return w


def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.c_[np.ones(X.shape[0]), X].dot(w)


def mean_squared_error(y_bar: np.ndarray, y: np.ndarray) -> float:
    return np.sum(np.square(y_bar - y)) / y_bar.shape[0]


def generate_random_weights(dimension: int = 1) -> np.ndarray:
    return np.random.default_rng().random(dimension + 1)


def generate_X_and_y(w: np.ndarray, size: int, noise_level: float = 0) -> Tuple[np.ndarray, np.ndarray]:
    # return X and y, respectively
    rng = np.random.default_rng()
    X = rng.random((size, w.shape[0] - 1))
    y = np.c_[np.ones(size), X].dot(w)
    y += rng.normal(0, stdev(y) * noise_level, size)
    return X, y


def test_no_noise_no_contamination(training_size: int = 100, testing_size: int = 900):
    w = generate_random_weights()
    X_training, y_training = generate_X_and_y(w, training_size, 0)
    X_testing, y_testing = generate_X_and_y(w, testing_size, 0)
    w_bar = mini_batch_gradient_descent(X_training, y_training, batch_size=50)
    y_bar = predict(X_testing, w_bar)
    mse = mean_squared_error(y_bar, y_testing)
    print(f"True weight vector: {w}")
    print(f"Estimated weight vector: {w_bar}")
    plt.figure()
    plt.scatter(X_training[:, 0], y_training, s=2)
    plt.suptitle("noise_level=0, epsilon=0")
    plt.title("Training Set")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(0, w.shape[0])
    plt.savefig("no_noise_no_contamination_training")
    plt.figure()
    plt.scatter(X_testing[:, 0], y_testing, s=2, c="blue", label="Ground Truth")
    plt.scatter(X_testing[:, 0], y_bar, s=2, c="red", label="Estimation")
    plt.suptitle("noise_level=0, epsilon=0")
    plt.title(f"Testing Set, MSE={mse}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(0, w.shape[0])
    plt.legend()
    plt.savefig("no_noise_no_contamination")


def test_no_contamination(training_size: int = 100, testing_size: int = 900, noise_level: float = 0.1):
    w = generate_random_weights()
    X_training, y_training = generate_X_and_y(w, training_size, noise_level)
    X_testing, y_testing = generate_X_and_y(w, testing_size, 0)
    w_bar = mini_batch_gradient_descent(X_training, y_training, batch_size=50)
    y_bar = predict(X_testing, w_bar)
    mse = mean_squared_error(y_bar, y_testing)
    print(f"True weight vector: {w}")
    print(f"Estimated weight vector: {w_bar}")
    plt.figure()
    plt.scatter(X_training[:, 0], y_training, s=2)
    plt.suptitle(f"noise_level={noise_level}, epsilon=0")
    plt.title("Training Set")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(0, w.shape[0])
    plt.savefig("no_contamination_training")
    plt.figure()
    plt.scatter(X_testing[:, 0], y_testing, s=2, c="blue", label="Ground Truth")
    plt.scatter(X_testing[:, 0], y_bar, s=2, c="red", label="Estimation")
    plt.suptitle(f"noise_level={noise_level}, epsilon=0")
    plt.title(f"Testing Set, MSE={mse}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(0, w.shape[0])
    plt.legend()
    plt.savefig("no_contamination")


def test_random_contamination(training_size: int = 100, testing_size: int = 900, noise_level: float = 0.1, epsilon: float = 0.1):
    rng = np.random.default_rng()
    w = generate_random_weights()
    X_training, y_training = generate_X_and_y(w, training_size, noise_level)
    X_contamination = rng.random((int(training_size * epsilon), 1))
    Y_contamination = rng.random(int(training_size * epsilon)) * w.shape[0]
    indices = rng.choice(training_size, int(training_size * epsilon), False)
    X_training[indices] = X_contamination
    y_training[indices] = Y_contamination
    X_testing, y_testing = generate_X_and_y(w, 1000, 0)
    w_bar = mini_batch_gradient_descent(
        X_training, y_training, batch_size=100, epsilon=epsilon)
    y_bar = predict(X_testing, w_bar)
    mse = mean_squared_error(y_bar, y_testing)
    print(f"True weight vector: {w}")
    print(f"Estimated weight vector: {w_bar}")
    plt.figure()
    plt.scatter(X_training[:, 0], y_training, s=2)
    plt.suptitle(f"noise_level={noise_level}, epsilon={epsilon}")
    plt.title("Training Set")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(0, w.shape[0])
    plt.savefig("random_contamination_training")
    plt.figure()
    plt.scatter(X_testing[:, 0], y_testing, s=2, c="blue", label="Ground Truth")
    plt.scatter(X_testing[:, 0], y_bar, s=2, c="red", label="Estimation")
    plt.suptitle(f"noise_level={noise_level}, epsilon={epsilon}")
    plt.title(f"Testing Set, MSE={mse}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(0, w.shape[0])
    plt.legend()
    plt.savefig("random_contamination")


if __name__ == "__main__":
    test_no_noise_no_contamination(1000, 9000)
    test_no_contamination(1000, 9000, 1)
    test_random_contamination(1000, 9000, 1, 0.8)
