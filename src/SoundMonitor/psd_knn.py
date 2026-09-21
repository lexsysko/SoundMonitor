import logging

import numpy as np


logger = logging.getLogger(__name__)


class PSD_KNN:
    def __init__(self, k: int = 5):
        self.k = k
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    @staticmethod
    def normalize(psd: np.ndarray, normalize_log: bool = True) -> np.ndarray:
        if normalize_log:
            psd = 10 * np.log10(psd + 1e-12)
        norm = np.linalg.norm(psd, axis=-1, keepdims=True)
        return psd / norm

    def fit(
        self,
        off_templates: list[np.ndarray],
        on_templates: list[np.ndarray],
        normalize: bool = False,
        normalize_log: bool = False,
    ) -> None:
        """Store labeled templates for states: 0 - OFF, 1 - ON"""
        X = np.vstack([off_templates, on_templates])

        self.X_train = self.normalize(X, normalize_log) if normalize else X

        self.y_train = np.array([0] * len(off_templates) + [1] * len(on_templates))

    def predict(
        self,
        psd: np.ndarray,
        normalize: bool = False,
        normalize_log: bool = True,
    ) -> tuple[int, float]:
        """
        :return:
        - state: 1 (ON) or 0 (OFF)
        - score: average likes (Cosine Similarity)  winner class [0.0...1.0]
        """
        # 1. Cosine similarity for all templates (value -1.0 ... 1.0)
        psd_norm = self.normalize(psd, normalize_log) if normalize else psd
        # logger.debug("psd_norm", psd_norm)
        # logger.debug("self.X_train", self.X_train)
        # logger.debug("X_train shape:", self.X_train.shape)
        # logger.debug("psd_norm shape:", psd_norm.shape)

        try:
            # similarities = np.dot(self.X_train, psd_norm)
            similarities = self.X_train @ psd_norm.ravel()
            # similarities = self.X_train @ psd_norm
        except ValueError as e:
            logger.error(f"Shape mismatch: X_train {self.X_train.shape}, psd {psd_norm.shape}")
            return 0, 0.0  # or some safe fallback

        # print("similarities", similarities)

        # 2. find top KNN
        top_k_indices = np.argsort(similarities)[-self.k :]
        top_k_labels = self.y_train[top_k_indices]
        top_k_sims = similarities[top_k_indices]

        # logger.debug("top_k_indices", top_k_indices)
        print("top_k_labels", top_k_labels)
        print("top_k_sims", top_k_sims)

        # 3. detect Winner
        on_count = np.sum(top_k_labels)
        state = 1 if on_count > (self.k / 2) else 0

        print("on_count", on_count)

        # 4. average likes only for winners
        winning_mask = top_k_labels == state
        winning_score = float(np.mean(top_k_sims[winning_mask]))

        return state, winning_score
