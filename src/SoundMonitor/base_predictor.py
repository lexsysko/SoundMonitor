from abc import ABC, abstractmethod

import numpy as np


class BasePredictor(ABC):
    @abstractmethod
    def predict(self, psd: np.ndarray, *args, **kwargs) -> tuple[int, float]:
        """
        :param psd: np.ndarray of shape (n_samples, n_dimensions)
        :return:
        - state: 1 (ON) or 0 (OFF)
        - score: average likes [0.0...1.0]
        """
        return 0, 0
