import matplotlib.pyplot as plt
import numpy as np

from typing import Tuple, Union

from src.loss import *
from src.polynomial_regressor import *
from src.preprocessor import *
from src.regularization import *


def export_training_figure(
    raw_x: np.ndarray,
    raw_y: np.ndarray,
    contamination_indices: Union[np.ndarray, None],
    transformed_x: Union[np.ndarray, None],
    transformed_y: Union[np.ndarray, None],
    predicted_y: np.ndarray,
    test_name: str,
    filename: str
):
    plt.figure()
    plt.suptitle(test_name)
    plt.title("Training Set")
    plt.grid()
    if contamination_indices is None:
        plt.scatter(raw_x, raw_y, s=4, c="blue", label="True Samples")
    else:
        plt.scatter(
            np.delete(raw_x, contamination_indices),
            np.delete(raw_y, contamination_indices),
            s=4,
            c="blue",
            label="Raw Samples"
        )
        plt.scatter(
            raw_x[contamination_indices],
            raw_y[contamination_indices],
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
    plt.scatter(raw_x, predicted_y, s=4, c="red", label="Predictions")
    plt.legend()
    plt.xlabel("x")
    plt.xlim(-1, 1)
    plt.ylabel("y")
    plt.ylim(raw_y.min(), raw_y.max())
    plt.savefig(filename)
    plt.close()
