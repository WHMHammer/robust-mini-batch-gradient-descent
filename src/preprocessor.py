import matplotlib.pyplot as plt
import numpy as np
from abc import ABC
from typing import Tuple


class Preprocessor(ABC):
    def __call__(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def export_figure(self, X: np.ndarray, Y: np.ndarray):
        raise NotImplementedError


class NullPreprocessor(Preprocessor):
    def __call__(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return X, Y

    def export_figure(self, X: np.ndarray, Y: np.ndarray):
        pass
