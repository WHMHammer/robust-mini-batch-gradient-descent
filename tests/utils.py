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
