import numpy as np
from csv import writer
from math import ceil
from matplotlib import pyplot as plt
from statistics import stdev
from typing import Tuple

DEBUG = False
if DEBUG:
    losses_log_filename = "losses.csv"
    with open(losses_log_filename, "w") as f:
        pass
    batch_index = 0


def trimmed_squared_loss_and_gradient(
    X: np.ndarray,
    w: np.ndarray,
    y: np.ndarray,
    epsilon: float
) -> Tuple[float, np.ndarray]:
    # return loss and gradient, respectively
    kept_size = ceil(X.shape[0] * (1 - epsilon))
    residuals = X.dot(w) - y
    losses = np.square(residuals)
    kept_indices = np.argsort(losses)[:kept_size]
    loss = sum(losses[kept_indices]) / kept_size
    gradient = X[kept_indices].transpose().dot(
        residuals[kept_indices]) / kept_size * 2
    if DEBUG:
        global batch_index
        with open(losses_log_filename, "a") as f:
            csv_writer = writer(f)
            csv_writer.writerows(
                np.c_[np.full(X.shape[0], batch_index), X[:, 1:], losses])
        batch_index += 1
    return loss, gradient


def mini_batch_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    batch_size: int,
    learning_rate: float,
    max_iter: int
) -> np.ndarray:
    # return the estimated weight vector, with the bias being the 0th element
    sample_size, dimension = X.shape
    rng = np.random.default_rng()
    X = np.c_[np.ones(sample_size), X]
    w = np.ones(dimension + 1)
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
        if abs(loss - prev_loss) / prev_loss < 1e-3:
            return w
        prev_loss = loss
    return w


def generate_X_and_y_with_w(w: np.ndarray, x_low: float, x_high: float, size: int, noise_level: float) -> Tuple[np.ndarray, np.ndarray]:
    # return X and y, respectively
    rng = np.random.default_rng()
    X = rng.uniform(x_low, x_high, (size, w.shape[0] - 1))
    y = np.c_[np.ones(size), X].dot(w)
    y += rng.normal(0, stdev(y) * noise_level, size)
    return X, y


def test_model(
    X_training: np.ndarray,
    y_training: np.ndarray,
    contaminated_indices: np.ndarray,
    X_testing: np.ndarray,
    y_testing: np.ndarray,
    epsilon: float,
    batch_size: int,
    learning_rate: float,
    max_iter: int,
    title: str
):
    w_bar = mini_batch_gradient_descent(
        X_training,
        y_training,
        epsilon,
        batch_size,
        learning_rate,
        max_iter
    )
    y_bar_training = np.c_[np.ones(X_training.shape[0]), X_training].dot(w_bar)
    y_bar_testing = np.c_[np.ones(X_testing.shape[0]), X_testing].dot(w_bar)
    testing_mse = np.sum(np.square(y_bar_testing - y_testing)
                         ) / y_bar_testing.shape[0]

    plt.figure()
    plt.suptitle(title)
    plt.title("Training Set")
    if contaminated_indices is None:
        plt.scatter(X_training[:, 0], y_training,
                    s=5, c="blue", label="Raw Data")
    else:
        plt.scatter(np.delete(X_training[:, 0], contaminated_indices), np.delete(
            y_training, contaminated_indices), s=5, c="blue", label="Authentic Data")
        plt.scatter(X_training[contaminated_indices, 0],
                    y_training[contaminated_indices], s=5, c="gray", label="Contamination")
    plt.scatter(X_training[:, 0], y_bar_training,
                s=5, c="red", label="Estimations")
    plt.legend()
    plt.savefig(f"{title} Training")

    plt.figure()
    plt.suptitle(title)
    plt.title(f"Testing Set, MSE={testing_mse}")
    plt.scatter(X_testing[:, 0], y_testing, s=5,
                c="blue", label="Ground Truth")
    plt.scatter(X_testing[:, 0], y_bar_testing,
                s=5, c="red", label="Estimations")
    plt.legend()
    plt.savefig(f"{title} Testing")


def test_no_noise_no_contamination():
    w_low = -10
    w_high = 10
    x_low = -10
    x_high = 10
    noise_level = 0
    training_size = 1000
    testing_size = 1000
    epsilon = 0

    rng = np.random.default_rng()
    w = rng.uniform(w_low, w_high, 2)
    X_training, y_training = generate_X_and_y_with_w(
        w, x_low, x_high, training_size, noise_level)
    X_testing, y_testing = generate_X_and_y_with_w(
        w, x_low, x_high, testing_size, 0)
    test_model(
        X_training,
        y_training,
        None,
        X_testing,
        y_testing,
        epsilon,
        100,
        0.01,
        1000,
        "00 No Noise No Contamination"
    )


def test_no_contamination():
    w_low = -10
    w_high = 10
    x_low = -10
    x_high = 10
    noise_level = 0.5
    training_size = 1000
    testing_size = 1000
    epsilon = 0

    rng = np.random.default_rng()
    w = rng.uniform(w_low, w_high, 2)
    X_training, y_training = generate_X_and_y_with_w(
        w, x_low, x_high, training_size, noise_level)
    X_testing, y_testing = generate_X_and_y_with_w(
        w, x_low, x_high, testing_size, 0)
    test_model(
        X_training,
        y_training,
        None,
        X_testing,
        y_testing,
        epsilon,
        100,
        0.01,
        1000,
        "01 No Contamination"
    )


def test_random_contamination():
    w_low = -10
    w_high = 10
    x_low = -10
    x_high = 10
    noise_level = 0.5
    training_size = 1000
    testing_size = 1000
    epsilon = 0.5

    rng = np.random.default_rng()
    w = rng.uniform(w_low, w_high, 2)
    X_training, y_training = generate_X_and_y_with_w(
        w, x_low, x_high, training_size, noise_level)
    contamination_size = int(training_size * epsilon)
    contaminated_indices = rng.choice(training_size, contamination_size, False)
    X_contamination = rng.uniform(
        X_training.min(), X_training.max(), (contamination_size, 1))
    Y_contamination = rng.uniform(
        y_training.min(), y_training.max(), contamination_size)
    X_training[contaminated_indices] = X_contamination
    y_training[contaminated_indices] = Y_contamination
    X_testing, y_testing = generate_X_and_y_with_w(
        w, x_low, x_high, testing_size, 0)
    test_model(
        X_training,
        y_training,
        contaminated_indices,
        X_testing,
        y_testing,
        epsilon,
        100,
        0.01,
        1000,
        "02 Random Contamination"
    )


if __name__ == "__main__":
    test_no_noise_no_contamination()
    test_no_contamination()
    test_random_contamination()
