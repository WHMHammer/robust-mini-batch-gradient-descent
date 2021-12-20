import matplotlib.pyplot as plt
import numpy as np
from os import makedirs
from os.path import join
from typing import Union
from src.loss import *
from src.polynomial_regressor import *
from src.preprocessor import *
from src.regularization import *


def mean_squared_error(predicted_y: np.ndarray, y: np.ndarray) -> float:
    return np.square(predicted_y - y).sum() / predicted_y.shape[0]


def export_figures(
    x_training: np.ndarray,
    y_training: np.ndarray,
    contamination_indices: Union[np.ndarray, None],
    transformed_x: Union[np.ndarray, None],
    transformed_y: Union[np.ndarray, None],
    predicted_y_training: np.ndarray,
    x_testing: np.ndarray,
    y_testing: np.ndarray,
    predicted_y_testing: np.ndarray,
    test_name: str,
    dirname: str
) -> str:
    try:
        makedirs(join("test_results", dirname))
    except FileExistsError:
        pass
    markdown_str = ""

    filename = join("test_results", dirname, "training.png")
    plt.figure()
    plt.suptitle(test_name)
    plt.title("Training Set")
    plt.grid()
    if contamination_indices is None:
        plt.scatter(x_training, y_training, s=4,
                    c="blue", label="True Samples")
    else:
        plt.scatter(
            np.delete(x_training, contamination_indices),
            np.delete(y_training, contamination_indices),
            s=4,
            c="blue",
            label="Raw Samples"
        )
        plt.scatter(
            x_training[contamination_indices],
            y_training[contamination_indices],
            s=4,
            c="gray",
            label="Contamination"
        )
    if transformed_x is not None:
        plt.scatter(
            transformed_x,
            transformed_y,
            s=16,
            c="limegreen",
            marker="^",
            label="Transformed Samples"
        )
    plt.scatter(x_training, predicted_y_training,
                s=4, c="red", label="Predictions")
    plt.legend()
    plt.xlabel("x")
    plt.xlim(-1, 1)
    plt.ylabel("y")
    plt.ylim(y_training.min(), y_training.max())
    plt.savefig(filename)
    plt.close()
    markdown_str += f" ![]({filename}) |"

    filename = join("test_results", dirname, "testing.png")
    plt.figure()
    plt.suptitle(test_name)
    plt.title(
        f"Testing Set, MSE={mean_squared_error(predicted_y_testing, y_testing)}")
    plt.grid()
    plt.scatter(x_testing, y_testing, s=4, c="blue", label="True Samples")
    plt.scatter(x_testing, predicted_y_testing,
                s=4, c="red", label="Predictions")
    plt.legend()
    plt.xlabel("x")
    plt.xlim(-1, 1)
    plt.ylabel("y")
    plt.ylim(y_training.min(), y_training.max())
    plt.savefig(filename)
    plt.close()
    markdown_str += f" ![]({filename}) |"

    return markdown_str


def test_naive(
    power: int,
    x_training: np.ndarray,
    y_training: np.ndarray,
    contamination_indices: np.ndarray,
    x_testing: np.ndarray,
    y_testing: np.ndarray,
    test_name: str,
    dirname: str
) -> str:
    regressor = PolynomialRegressor(
        power,
        NullRegularization(),
        EpsilonTrimmedSquaredLoss(0),
        0.01,
        100,
        100000
    )
    regressor.fit(x_training, y_training)

    predicted_y_training = regressor.predict(x_training)
    predicted_y_testing = regressor.predict(x_testing)
    return export_figures(
        x_training,
        y_training,
        contamination_indices,
        None,
        None,
        predicted_y_training,
        x_testing,
        y_testing,
        predicted_y_testing,
        f"{test_name} (Naive)",
        join(dirname, "naive")
    )


def test_huber_loss(
    power: int,
    x_training: np.ndarray,
    y_training: np.ndarray,
    contamination_indices: np.ndarray,
    x_testing: np.ndarray,
    y_testing: np.ndarray,
    test_name: str,
    dirname: str
) -> str:
    regressor = PolynomialRegressor(
        power,
        NullRegularization(),
        EpsilonTrimmedHuberLoss(0, 10),
        0.01,
        100,
        100000
    )
    regressor.fit(x_training, y_training)

    predicted_y_training = regressor.predict(x_training)
    predicted_y_testing = regressor.predict(x_testing)
    return export_figures(
        x_training,
        y_training,
        contamination_indices,
        None,
        None,
        predicted_y_training,
        x_testing,
        y_testing,
        predicted_y_testing,
        f"{test_name} (Huber Loss)",
        join(dirname, "huber_loss")
    )


def test_epsilon_trimmed_huber_loss(
    power: int,
    x_training: np.ndarray,
    y_training: np.ndarray,
    contamination_indices: np.ndarray,
    x_testing: np.ndarray,
    y_testing: np.ndarray,
    test_name: str,
    dirname: str
) -> str:
    regressor = PolynomialRegressor(
        power,
        NullRegularization(),
        EpsilonTrimmedHuberLoss(0.49, 10),
        0.01,
        100,
        100000
    )
    regressor.fit(x_training, y_training)

    predicted_y_training = regressor.predict(x_training)
    predicted_y_testing = regressor.predict(x_testing)
    return export_figures(
        x_training,
        y_training,
        contamination_indices,
        None,
        None,
        predicted_y_training,
        x_testing,
        y_testing,
        predicted_y_testing,
        f"{test_name} (ε-trimmed Huber Loss)",
        join(dirname, "epsilon_trimmed_huber_loss")
    )


def test_mean_kernel_preprocessor(
    power: int,
    x_training: np.ndarray,
    y_training: np.ndarray,
    contamination_indices: np.ndarray,
    x_testing: np.ndarray,
    y_testing: np.ndarray,
    test_name: str,
    dirname: str
) -> str:
    preprocessor = MeanKernelPreprocessor(
        (0.2, 2),
        (0.02, 0.2),
        0.01
    )
    regressor = PolynomialRegressor(
        power,
        NullRegularization(),
        EpsilonTrimmedSquaredLoss(0),
        0.01,
        100,
        100000
    )
    transformed_x, transformed_y = preprocessor(x_training, y_training)
    regressor.fit(transformed_x, transformed_y)

    predicted_y_training = regressor.predict(x_training)
    predicted_y_testing = regressor.predict(x_testing)
    return export_figures(
        x_training,
        y_training,
        contamination_indices,
        transformed_x,
        transformed_y,
        predicted_y_training,
        x_testing,
        y_testing,
        predicted_y_testing,
        f"{test_name} (Mean-kernel Preprocessor)",
        join(dirname, "mean_kernel_preprocessor")
    )


def test_epsilon_trimmed_huber_loss_with_mean_kernel_preprocessor(
    power: int,
    x_training: np.ndarray,
    y_training: np.ndarray,
    contamination_indices: np.ndarray,
    x_testing: np.ndarray,
    y_testing: np.ndarray,
    test_name: str,
    dirname: str
) -> str:
    preprocessor = MeanKernelPreprocessor(
        (0.2, 2),
        (0.02, 0.2),
        0.01
    )
    regressor = PolynomialRegressor(
        power,
        NullRegularization(),
        EpsilonTrimmedHuberLoss(0.49, 10),
        0.01,
        100,
        100000
    )
    transformed_x, transformed_y = preprocessor(x_training, y_training)
    regressor.fit(transformed_x, transformed_y)

    predicted_y_training = regressor.predict(x_training)
    predicted_y_testing = regressor.predict(x_testing)
    return export_figures(
        x_training,
        y_training,
        contamination_indices,
        transformed_x,
        transformed_y,
        predicted_y_training,
        x_testing,
        y_testing,
        predicted_y_testing,
        f"{test_name} (ε-trimmed Huber Loss with Mean-kernel Preprocessor)",
        join(dirname, "epsilon_trimmed_huber_loss_with_mean_kernel_preprocessor")
    )
