import numpy as np
from csv import writer
from math import ceil
from matplotlib import pyplot as plt
from os.path import join
from statistics import stdev
from typing import Tuple

DEBUG = False
if DEBUG:
    losses_log_filename = "losses.csv"
    with open(losses_log_filename, "w") as f:
        pass
    batch_index = 0



def calculate_huber_loss(td_errors, kappa=1.0):
    return torch.where(
        td_errors.abs() <= kappa,
        0.5 * td_errors.pow(2),
        kappa * (td_errors.abs() - 0.5 * kappa))

def l2_regularization(w: np.ndarray, alpha: float) -> Tuple[float, np.ndarray]:
    # returns loss and gradient, respectively
    gradient = w * 2 * alpha * alpha
    gradient[0] = 0
    return np.sum((w * w)[1:]) * alpha * alpha, gradient


def trimmed_squared_loss_and_gradient(
    X: np.ndarray,
    w: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    alpha: float
) -> Tuple[float, np.ndarray]:
    # return loss and gradient, respectively
    kept_size = ceil(X.shape[0] * (1 - epsilon))
    residuals = X.dot(w) - y
    losses = np.square(residuals)
    kept_indices = np.argsort(losses)[:kept_size]
    loss = sum(losses[kept_indices]) / kept_size
    gradient = X[kept_indices].transpose().dot(
        residuals[kept_indices]) / kept_size * 2
    reg_loss, reg_gradient = l2_regularization(w, alpha)
    if DEBUG:
        global batch_index
        with open(losses_log_filename, "a") as f:
            csv_writer = writer(f)
            csv_writer.writerows(
                np.c_[np.full(X.shape[0], batch_index), X[:, 1:], losses])
        batch_index += 1
    return loss + reg_loss, gradient + reg_gradient


def mini_batch_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    batch_size: int,
    learning_rate: float,
    max_iter: int,
    alpha: float
) -> np.ndarray:
    # return the estimated weight vector, with the bias being the 0th element
    sample_size, dimension = X.shape
    rng = np.random.default_rng()
    X = np.c_[np.ones(sample_size), X]
    w = np.ones(dimension + 1)
    if batch_size == 0:
        prev_loss, gradient = trimmed_squared_loss_and_gradient(
            X, w, y, epsilon, alpha)
    else:
        indices = rng.choice(sample_size, batch_size, False)
        prev_loss, gradient = trimmed_squared_loss_and_gradient(
            X[indices], w, y[indices], epsilon, alpha)
    for i in range(max_iter):
        w -= gradient * learning_rate
        if batch_size == 0:
            loss, gradient = trimmed_squared_loss_and_gradient(
                X, w, y, epsilon, alpha)
        else:
            indices = rng.choice(sample_size, batch_size, False)
            loss, gradient = trimmed_squared_loss_and_gradient(
                X[indices], w, y[indices], epsilon, alpha)
        if abs(loss - prev_loss) / prev_loss < 1e-5:
            return w
        prev_loss = loss
    return w


def power_expand(_X: np.ndarray, power: int) -> np.ndarray:
    # | x0 |    | x0 x0^2 ... |
    # | x1 | => | x1 x1^2 ... |
    # | .. |    | .. .... ... |
    X = np.empty((_X.shape[0], power))
    X[:, 0] = _X
    for i in range(1, power):
        X[:, i] = _X ** (i + 1)
    return X


def generate_X_and_y_with_w(w: np.ndarray, x_low: float, x_high: float, size: int, noise_level: float) -> Tuple[np.ndarray, np.ndarray]:
    # return X and y, respectively
    rng = np.random.default_rng()
    dimension = w.shape[0] - 1
    X = power_expand(rng.uniform(x_low, x_high, size), dimension)
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
    alpha: float,
    title: str
):
    w_bar = mini_batch_gradient_descent(
        X_training,
        y_training,
        epsilon,
        batch_size,
        learning_rate,
        max_iter,
        alpha
    )
    y_bar_training = np.c_[np.ones(X_training.shape[0]), X_training].dot(w_bar)
    y_bar_testing = np.c_[np.ones(X_testing.shape[0]), X_testing].dot(w_bar)
    testing_mse = np.sum(np.square(y_bar_testing - y_testing)
                         ) / y_bar_testing.shape[0]

    plt.figure()
    plt.suptitle(f"{title} with trimming")
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
    plt.xlabel("x")
    plt.xlim(X_training.min(), X_training.max())
    plt.ylabel("y")
    plt.ylim(y_training.min(), y_training.max())
    plt.savefig(join("test_result_img", f"{title} Training with trimming"))

    plt.figure()
    plt.suptitle(f"{title} with trimming")
    plt.title(f"Testing Set, MSE={testing_mse}")
    plt.scatter(X_testing[:, 0], y_testing, s=5,
                c="blue", label="Ground Truth")
    plt.scatter(X_testing[:, 0], y_bar_testing,
                s=5, c="red", label="Estimations")
    plt.legend()
    plt.xlabel("x")
    plt.xlim(X_training.min(), X_training.max())
    plt.ylabel("y")
    plt.ylim(y_training.min(), y_training.max())
    plt.savefig(join("test_result_img", f"{title} Testing with trimming"))

    w_bar = mini_batch_gradient_descent(
        X_training,
        y_training,
        0,
        batch_size,
        learning_rate,
        max_iter,
        alpha
    )
    y_bar_training = np.c_[np.ones(X_training.shape[0]), X_training].dot(w_bar)
    y_bar_testing = np.c_[np.ones(X_testing.shape[0]), X_testing].dot(w_bar)
    testing_mse = np.sum(np.square(y_bar_testing - y_testing)
                         ) / y_bar_testing.shape[0]

    plt.figure()
    plt.suptitle(f"{title} without trimming")
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
    plt.xlabel("x")
    plt.xlim(X_training.min(), X_training.max())
    plt.ylabel("y")
    plt.ylim(y_training.min(), y_training.max())
    plt.savefig(join("test_result_img", f"{title} Training without trimming"))

    plt.figure()
    plt.suptitle(f"{title} without trimming")
    plt.title(f"Testing Set, MSE={testing_mse}")
    plt.scatter(X_testing[:, 0], y_testing, s=5,
                c="blue", label="Ground Truth")
    plt.scatter(X_testing[:, 0], y_bar_testing,
                s=5, c="red", label="Estimations")
    plt.legend()
    plt.xlabel("x")
    plt.xlim(X_training.min(), X_training.max())
    plt.ylabel("y")
    plt.ylim(y_training.min(), y_training.max())
    plt.savefig(join("test_result_img", f"{title} Testing without trimming"))


def test_no_noise_no_contamination(
    power: int,
    epsilon: float,
    batch_size: int,
    learning_rate: float,
    max_iter: int,
    alpha: float
):
    w_low = -10
    w_high = 10
    x_low = -1
    x_high = 1
    training_size = 1000
    testing_size = 1000

    rng = np.random.default_rng()
    w = rng.uniform(w_low, w_high, power + 1)
    X_training, y_training = generate_X_and_y_with_w(
        w, x_low, x_high, training_size, 0)
    X_testing, y_testing = generate_X_and_y_with_w(
        w, x_low, x_high, testing_size, 0)
    test_model(
        X_training,
        y_training,
        None,
        X_testing,
        y_testing,
        epsilon,
        batch_size,
        learning_rate,
        max_iter,
        alpha,
        "No Noise No Contamination"
    )


def test_no_contamination(
    power: int,
    noise_level: float,
    epsilon: float,
    batch_size: int,
    learning_rate: float,
    max_iter: int,
    alpha: float
):
    w_low = -10
    w_high = 10
    x_low = -1
    x_high = 1
    training_size = 1000
    testing_size = 1000

    rng = np.random.default_rng()
    w = rng.uniform(w_low, w_high, power + 1)
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
        batch_size,
        learning_rate,
        max_iter,
        alpha,
        "No Contamination"
    )


def test_random_contamination(
    power: int,
    noise_level: float,
    epsilon: float,
    batch_size: int,
    learning_rate: float,
    max_iter: int,
    alpha: float
):
    w_low = -10
    w_high = 10
    x_low = -1
    x_high = 1
    training_size = 1000
    testing_size = 1000

    rng = np.random.default_rng()
    w = rng.uniform(w_low, w_high, power + 1)
    X_training, y_training = generate_X_and_y_with_w(
        w, x_low, x_high, training_size, noise_level)
    contamination_size = int(training_size * epsilon)
    contaminated_indices = rng.choice(training_size, contamination_size, False)
    X_contamination = power_expand(rng.uniform(
        X_training[:, 0].min(), X_training[:, 0].max(), contamination_size), power)
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
        batch_size,
        learning_rate,
        max_iter,
        alpha,
        "Random Contamination"
    )


def test_edge_contamination(
    power: int,
    noise_level: float,
    epsilon: float,
    batch_size: int,
    learning_rate: float,
    max_iter: int,
    alpha: float
):
    w_low = -10
    w_high = 10
    x_low = -1
    x_high = 1
    training_size = 1000
    testing_size = 1000

    rng = np.random.default_rng()
    w = rng.uniform(w_low, w_high, power + 1)
    X_training, y_training = generate_X_and_y_with_w(
        w, x_low, x_high, training_size, noise_level)
    contamination_size = int(training_size * epsilon / 2)
    contaminated_indices = rng.choice(training_size, contamination_size, False)
    X_contamination = power_expand(rng.uniform(X_training[:, 0].max(
    ) - (X_training[:, 0].max() - X_training[:, 0].min())/10, X_training[:, 0].max(), contamination_size), power)
    Y_contamination = rng.uniform(y_training.min() - y_training.max(), y_training.min(
    ) - y_training.max() + (y_training.max() - y_training.min())/10, contamination_size)
    X_training[contaminated_indices] = X_contamination
    y_training[contaminated_indices] = Y_contamination
    contaminated_indices2 = rng.choice(
        training_size, contamination_size, False)
    X_contamination2 = power_expand(rng.uniform(X_training[:, 0].min(), X_training[:, 0].min(
    ) + (X_training[:, 0].max() - X_training[:, 0].min())/10, contamination_size), power)
    Y_contamination2 = rng.uniform(y_training.max(
    ) * 2 - (y_training.max() - y_training.min())/10, y_training.max() * 2, contamination_size)
    X_training[contaminated_indices2] = X_contamination2
    y_training[contaminated_indices2] = Y_contamination2

    contaminated_indices = np.concatenate(
        [contaminated_indices, contaminated_indices2])
    X_testing, y_testing = generate_X_and_y_with_w(
        w, x_low, x_high, testing_size, 0)
    test_model(
        X_training,
        y_training,
        contaminated_indices,
        X_testing,
        y_testing,
        epsilon,
        batch_size,
        learning_rate,
        max_iter,
        alpha,
        "Edge Contamination"
    )


def test_parallel_line_contamination(
    power: int,
    noise_level: float,
    epsilon: float,
    batch_size: int,
    learning_rate: float,
    max_iter: int,
    alpha: float
):
    w_low = -10
    w_high = 10
    x_low = -1
    x_high = 1
    training_size = 1000
    testing_size = 1000

    rng = np.random.default_rng()
    w = rng.uniform(w_low, w_high, power + 1)
    X_training, y_training = generate_X_and_y_with_w(
        w, x_low, x_high, training_size, noise_level)
    contamination_size = int(training_size * epsilon)
    contaminated_indices = rng.choice(training_size, contamination_size, False)
    y_training[contaminated_indices] += (y_training.max() -
                                         y_training.min()) * 2
    X_testing, y_testing = generate_X_and_y_with_w(
        w, x_low, x_high, testing_size, 0)
    test_model(
        X_training,
        y_training,
        contaminated_indices,
        X_testing,
        y_testing,
        epsilon,
        batch_size,
        learning_rate,
        max_iter,
        alpha,
        "Parallel Line Contamination"
    )



def test_begin_contamination(
    power: int,
    noise_level: float,
    epsilon: float,
    batch_size: int,
    learning_rate: float,
    max_iter: int,
    alpha: float
):
    w_low = -10
    w_high = 10
    x_low = -1
    x_high = 1
    training_size = 1000
    testing_size = 1000

    rng = np.random.default_rng()
    w = rng.uniform(w_low, w_high, power + 1)
    X_training, y_training = generate_X_and_y_with_w(
        w, x_low, x_high, training_size, noise_level)
    contamination_size = int(training_size * epsilon)
    indices = np.argsort(X_training[:, 0])
    contaminated_indices = indices[np.arange(contamination_size)]
    
    y_training[contaminated_indices] += (y_training.max() -
                                         y_training.min()) * 2
    X_testing, y_testing = generate_X_and_y_with_w(
        w, x_low, x_high, testing_size, 0)
    test_model(
        X_training,
        y_training,
        contaminated_indices,
        X_testing,
        y_testing,
        epsilon,
        batch_size,
        learning_rate,
        max_iter,
        alpha,
        "Beginning Contamination"
    )


def test_end_contamination(
    power: int,
    noise_level: float,
    epsilon: float,
    batch_size: int,
    learning_rate: float,
    max_iter: int,
    alpha: float
):
    w_low = -10
    w_high = 10
    x_low = -1
    x_high = 1
    training_size = 1000
    testing_size = 1000

    rng = np.random.default_rng()
    w = rng.uniform(w_low, w_high, power + 1)
    X_training, y_training = generate_X_and_y_with_w(
        w, x_low, x_high, training_size, noise_level)
    contamination_size = int(training_size * epsilon)
    indices = np.argsort(X_training[:, 0])
    contaminated_indices = indices[np.arange(training_size - contamination_size,training_size)]
    
    y_training[contaminated_indices] += (y_training.max() -
                                         y_training.min()) * 2
    X_testing, y_testing = generate_X_and_y_with_w(
        w, x_low, x_high, testing_size, 0)
    test_model(
        X_training,
        y_training,
        contaminated_indices,
        X_testing,
        y_testing,
        epsilon,
        batch_size,
        learning_rate,
        max_iter,
        alpha,
        "Endding Contamination"
    )


def test_mid_rand_contamination(
    power: int,
    noise_level: float,
    epsilon: float,
    batch_size: int,
    learning_rate: float,
    max_iter: int,
    alpha: float
):
    w_low = -10
    w_high = 10
    x_low = -1
    x_high = 1
    training_size = 1000
    testing_size = 1000

    rng = np.random.default_rng()
    w = rng.uniform(w_low, w_high, power + 1)
    X_training, y_training = generate_X_and_y_with_w(
        w, x_low, x_high, training_size, noise_level)
    contamination_size = int(training_size * epsilon)
    indices = np.argsort(X_training[:, 0])
    num = np.random.randint(low = 2, high = 10, size=1)[0]
    print(num)
    contaminated_indices = indices[np.arange(int(training_size/num),int(training_size/num) + contamination_size)]
    
    y_training[contaminated_indices] += (y_training.max() -
                                         y_training.min()) * 2
    X_testing, y_testing = generate_X_and_y_with_w(
        w, x_low, x_high, testing_size, 0)
    test_model(
        X_training,
        y_training,
        contaminated_indices,
        X_testing,
        y_testing,
        epsilon,
        batch_size,
        learning_rate,
        max_iter,
        alpha,
        "Middle Random Contamination"
    )

def test_mid_contamination(
    power: int,
    noise_level: float,
    epsilon: float,
    batch_size: int,
    learning_rate: float,
    max_iter: int,
    alpha: float
):
    w_low = -10
    w_high = 10
    x_low = -1
    x_high = 1
    training_size = 1000
    testing_size = 1000

    rng = np.random.default_rng()
    w = rng.uniform(w_low, w_high, power + 1)
    X_training, y_training = generate_X_and_y_with_w(
        w, x_low, x_high, training_size, noise_level)
    contamination_size = int(training_size * epsilon)
    indices = np.argsort(X_training[:, 0])
    contaminated_indices = indices[np.arange(int(training_size/2 - contamination_size/2) ,int(training_size/2 + contamination_size/2))]
    
    y_training[contaminated_indices] += (y_training.max() -
                                         y_training.min()) * 2
    X_testing, y_testing = generate_X_and_y_with_w(
        w, x_low, x_high, testing_size, 0)
    test_model(
        X_training,
        y_training,
        contaminated_indices,
        X_testing,
        y_testing,
        epsilon,
        batch_size,
        learning_rate,
        max_iter,
        alpha,
        "Middle Contamination"
    )


if __name__ == "__main__":
    test_no_noise_no_contamination(
        5,  # power
        0,  # epsilon
        100,  # batch size
        0.01,  # learning rate
        100000,  # max iter,
        0  # alpha
    )
    test_no_contamination(
        5,  # power
        1,  # noise level
        0,  # epsilon
        100,  # batch size
        0.01,  # learning rate
        100000,  # max iter,
        0  # alpha
    )
    test_random_contamination(
        5,  # power
        1,  # noise level
        0.5,  # epsilon
        100,  # batch size
        0.01,  # learning rate
        100000,  # max iter,
        0  # alpha
    )
    test_edge_contamination(
        5,  # power
        1,  # noise level
        0.5,  # epsilon
        100,  # batch size
        0.01,  # learning rate
        100000,  # max iter,
        0  # alpha
    )
    test_parallel_line_contamination(
        5,  # power
        1,  # noise level
        0.49,  # epsilon
        100,  # batch size
        0.01,  # learning rate
        100000,  # max iter,
        0  # alpha
    )
    test_begin_contamination(
        5,  # power
        1,  # noise level
        0.1,  # epsilon
        100,  # batch size
        0.01,  # learning rate
        100000,  # max iter,
        0  # alpha
    )
    test_end_contamination(
        5,  # power
        1,  # noise level
        0.1,  # epsilon
        100,  # batch size
        0.01,  # learning rate
        100000,  # max iter,
        0  # alpha
    )
    test_mid_rand_contamination(
        5,  # power
        1,  # noise level
        0.1,  # epsilon
        100,  # batch size
        0.01,  # learning rate
        100000,  # max iter,
        0  # alpha
    )
    test_mid_contamination(
        5,  # power
        1,  # noise level
        0.1,  # epsilon
        100,  # batch size
        0.01,  # learning rate
        100000,  # max iter,
        0  # alpha
    )

