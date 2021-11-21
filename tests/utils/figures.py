import matplotlib.pyplot as plt
import numpy as np


def export_figure(x: np.ndarray,y: np.ndarray, y_bar: np.ndarray, contaminated_indices: np.ndarray,suptitle: str, title: str, filename: str):
    plt.figure()
    if contaminated_indices is None:
        plt.scatter(x, y, s=5, c="blue", label="True Samples")
    else:
        plt.scatter(
            np.delete(x, contaminated_indices),
            np.delete(y, contaminated_indices),
            s=5, c="blue", label="True Samples"
        )
        plt.scatter(
            x[contaminated_indices],
            y[contaminated_indices],
            s=5, color="gray", label="Contamination"
        )
    plt.scatter(x, y_bar, s=5, c="red", label="Predictions")
    plt.suptitle(suptitle)
    plt.title(title)
    plt.legend()
    plt.xlabel("x")
    x_padding = (x.max() - x.min()) / 10
    plt.xlim(x.min() - x_padding, x.max() + x_padding)
    plt.ylabel("y")
    y_padding = (y.max() - y.min()) / 10
    plt.ylim(y.min() - y_padding, y.max() + y_padding)
    plt.savefig(filename)
    plt.close()
