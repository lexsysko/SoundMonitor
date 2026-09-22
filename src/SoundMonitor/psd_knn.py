import logging
import numpy as np

from SoundMonitor.normalizer import normalize_psd, get_concatenated_norm
from SoundMonitor.numpy_normalizer import NumpyNormalizer, NumpyStandardScaler
from SoundMonitor.settings import NORMALIZE_METHOD

logger = logging.getLogger(__name__)


class PSD_KNN:
    def __init__(self, k: int = 5):
        self.k = k
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.global_norm: np.float64 | None = None
        self.MARGIN_POWER = 0.1
        self.MIN_SCORE = 0.3
        # self.scaler = NumpyStandardScaler(with_mean=True, with_std=False)
        self.scaler = NumpyStandardScaler()
        self.normalizer = NumpyNormalizer(norm="l2")

    def normalize(self, psd: np.ndarray) -> np.ndarray:
        psd_norm, _, self.global_norm = normalize_psd(psd, NORMALIZE_METHOD)
        return psd_norm

    @staticmethod
    def enforce_unit_length(x: np.ndarray) -> np.ndarray:
        return x / (np.linalg.norm(x) + 1e-12)

    def fit(
        self,
        off_templates: list[np.ndarray],
        on_templates: list[np.ndarray],
        normalize: bool = True,
    ) -> None:
        """Store labeled templates for states: 0 - OFF, 1 - ON"""
        X = np.vstack([off_templates, on_templates])

        # X_train = self.normalize(X) if normalize else X
        # # self.X_train = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-12)
        # self.X_train = self.normalizer.fit_transform(X_train)
        X_scaled = self.scaler.fit_transform(X)
        self.X_train = self.normalizer.fit_transform(X_scaled)

        self.y_train = np.array([0] * len(off_templates) + [1] * len(on_templates))

    def predict(
        self,
        psd: np.ndarray,
        normalize: bool = True,
    ) -> tuple[int, float]:
        """
        :return:
        - state: 1 (ON) or 0 (OFF)
        - score: average likes (Cosine Similarity)  winner class [0.0...1.0]
        """
        # 1. Cosine similarity for all templates (value -1.0 ... 1.0)
        # psd_norm = psd / self.global_norm if (normalize and self.global_norm is not None) else psd

        psd_scaled = self.scaler.transform(psd.reshape(1, -1))
        psd_norm = self.normalizer.transform(psd_scaled)[0]

        # psd_norm = self.enforce_unit_length(psd_norm)
        print("predict psd_norm shape", psd_norm.shape)
        # print(self.normalizer.params.shape)
        # print(self.normalizer.params)
        # psd_norm = self.normalizer.transform(psd.reshape(1, -1))[0]

        print("predict psd_norm_max", psd_norm.max())
        print("predict psd_norm_min", psd_norm.min())
        print("predict self.X_train_max", self.X_train.max())
        print("predict self.X_train_min", self.X_train.min())

        # logger.debug("psd_norm", psd_norm)
        # logger.debug("self.X_train", self.X_train)
        # logger.debug("X_train shape:", self.X_train.shape)
        # logger.debug("psd_norm shape:", psd_norm.shape)

        # enforce unit length normalization regardless of training method

        try:
            # similarities = np.dot(self.X_train, psd_norm)
            # similarities = self.X_train @ psd_norm.ravel()
            # similarities = self.X_train @ psd_norm
            similarities = self.X_train @ psd_norm
        except ValueError as e:
            logger.error(f"Shape mismatch: X_train {self.X_train.shape}, psd {psd_norm.shape}")
            return 0, 0.0  # or some safe fallback

        print("similarities", similarities)

        # 2. find top KNN
        top_k_indices = np.argsort(similarities)[-self.k :]
        top_k_labels = self.y_train[top_k_indices]
        top_k_sims = similarities[top_k_indices]

        # logger.debug("top_k_indices", top_k_indices)
        print("top_k_labels", top_k_labels)
        print("top_k_sims", top_k_sims)

        # Weighted vote
        weights = top_k_sims / top_k_sims.sum()
        on_weight = weights[top_k_labels == 1].sum()
        off_weight = weights[top_k_labels == 0].sum()

        mean_label_1 = top_k_sims[top_k_labels == 1].mean()
        mean_label_0 = top_k_sims[top_k_labels == 0].mean()

        off_magring = self.MARGIN_POWER * (on_weight + off_weight)

        winning_score = float(max(on_weight, off_weight))
        logger.debug(
            f"{winning_score=} {weights=} {on_weight=} {off_weight=} {off_magring=} {mean_label_1=}  {mean_label_0=}"
        )

        if mean_label_1 < self.MIN_SCORE:
            state = 0
        else:
            state = 1 if on_weight > off_weight - off_magring else 0

        return state, winning_score
