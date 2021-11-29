import matplotlib.pyplot as plt
import numpy as np

from typing import Tuple


def export_figure(
    x: np.ndarray,
    y: np.ndarray,
    x_transformed: np.ndarray,
    y_transformed: np.ndarray,
    y_bar: np.ndarray,
    contaminated_indices: np.ndarray,
    suptitle: str,
    title: str,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    filename: str
):
    plt.figure()
    plt.suptitle(suptitle)
    plt.title(title)
    if contaminated_indices is None:
        plt.scatter(x, y, s=4, c="blue", label="True Samples")
    else:
        plt.scatter(
            np.delete(x, contaminated_indices),
            np.delete(y, contaminated_indices),
            s=4,
            c="blue",
            label="True Samples"
        )
        plt.scatter(
            x[contaminated_indices],
            y[contaminated_indices],
            s=4,
            c="gray",
            label="Contamination"
        )
    if x_transformed is not None:
        plt.scatter(
            x_transformed,
            y_transformed,
            s=4,
            c="green",
            marker="^",
            label="Transformed Samples"
        )
    plt.scatter(x, y_bar, s=5, c="red", label="Predictions")
    plt.legend()
    plt.xlabel("x")
    plt.xlim(xlim)
    plt.ylabel("y")
    plt.ylim(ylim)
    plt.savefig(filename)
    plt.close()
