import logging
import numpy as np

from SoundMonitor.predictors.base_predictor import BasePredictor
from SoundMonitor.analize.correlate import compute_aligned_signals_batch
from SoundMonitor.analize.normalizer import normalize_psd
from SoundMonitor.analize.numpy_normalizer import NumpyNormalizer, NumpyStandardScaler
from SoundMonitor.settings import NORMALIZE_METHOD, THRESHOLD_ON_SIM, THRESHOLD_ON_POWER_ALPHA

logger = logging.getLogger(__name__)


def np_stats(x: np.ndarray, label: str = "psd_norm") -> None:
    """
    Print diagnostic stats for a PSD vector.
    Works for both 1D (single PSD) and 2D (batch of PSDs).
    """
    if logger.getEffectiveLevel() != logging.DEBUG:
        return
    x = np.asarray(x)

    if x.ndim == 1:
        print(f"{label} shape {x.shape}")
        print(f"{label} min {x.min():.6f}")
        print(f"{label} max {x.max():.6f}")
        print(f"{label} mean {x.mean():.6f}")
        print(f"{label} norm {np.linalg.norm(x):.6f}")
    elif x.ndim == 2:
        print(f"{label} shape {x.shape}")
        print(f"{label} row min {x.min(axis=1)}")
        print(f"{label} row max {x.max(axis=1)}")
        print(f"{label} row mean {x.mean(axis=1)}")
        print(f"{label} row norm {np.linalg.norm(x, axis=1)}")
    else:
        print(f"{label} has unsupported ndim {x.ndim}")


class FeatureAdder:
    @staticmethod
    def add_features(
        X: np.ndarray, f: np.ndarray | None = None, power_exponent: float = 1, eps: float = 1e-30
    ) -> np.ndarray:
        """
        Compute extra features for each row in X.
        X: (n_samples, n_features)
        f: frequency array of shape (n_features,) or None
        Returns: (n_samples, n_extra_features)
        """
        # If single vector, promote to 2D
        single_vector = False
        if X.ndim == 1:
            X = X.reshape(1, -1)
            single_vector = True

        # Apply exponent scaling
        X_powered = np.power(X, power_exponent) if power_exponent != 1 else X

        # Log sum power per row
        log_sum_power = np.log10(np.sum(X_powered, axis=1) + eps)

        # Log max power per row
        log_max_power = np.log10(np.max(X_powered, axis=1) + eps)

        # Stack into (n_samples, 2)
        features = np.column_stack((log_sum_power, log_max_power))

        # add peak frequency if provided
        # if f is not None:
        #     peak_indices = np.argmax(X, axis=1)
        #     peak_freqs = f[peak_indices]
        #     features = np.column_stack((features, peak_freqs))

        # If single vector, return 1D array instead of (1, n_extra_features)
        if single_vector:
            return features.ravel()

        return features


class PSD_KNN(BasePredictor):
    def __init__(self, k: int = 5, crop_live_freq_range: bool = False):
        self.k = k
        self.X_train: np.ndarray | None = None
        self.X_features: np.ndarray | None = None
        self.global_norm: np.float64 | None = None
        self.MARGIN_POWER = 0.1
        self.MIN_SCORE = THRESHOLD_ON_SIM
        self.scaler = NumpyStandardScaler()
        self.normalizer = NumpyNormalizer(norm="l2")
        self.alpha = THRESHOLD_ON_POWER_ALPHA
        self.freq: np.ndarray | None = None
        self.freq_low: float | None = None
        self.freq_high: float | None = None
        self.mask: np.ndarray | None = None
        self.crop_live_freq_range: bool = crop_live_freq_range
        self.align_signals: bool = True

    def normalize(self, psd: np.ndarray) -> np.ndarray:
        psd_norm, _, self.global_norm = normalize_psd(psd, NORMALIZE_METHOD)
        return psd_norm

    def crop_freq(self, freq: np.ndarray | None = None, low: float | None = None, high: float | None = None) -> None:
        if freq is None or low is None or high is None:
            return
        self.freq_low = low
        self.freq_high = high
        self.mask = (freq >= low) & (freq <= high)
        self.freq = freq[self.mask]  # keep full axis

    def crop_band(self, psd: np.ndarray) -> np.ndarray:
        """
        Apply frequency mask to PSD data.
        Works for both (n_features,) and (n_samples, n_features).
        """
        if self.mask is None:
            return psd

        psd = np.asarray(psd) if isinstance(psd, list) else psd

        # If already cropped (length matches mask sum), return as-is
        if psd.shape[-1] == np.sum(self.mask):
            return psd

        if psd.ndim == 1:
            # Single PSD vector
            return psd[self.mask]
        elif psd.ndim == 2:
            # Batch of PSDs: mask along feature axis
            return psd[:, self.mask]
        else:
            raise ValueError(f"Unsupported PSD shape {psd.shape}")

    @staticmethod
    def enforce_unit_length(x: np.ndarray) -> np.ndarray:
        return x / (np.linalg.norm(x) + 1e-12)

    def fit(
        self,
        on_templates: list[np.ndarray],
        freq: np.ndarray | None = None,
        freq_low: float | None = None,
        freq_high: float | None = None,
    ) -> None:
        """Store labeled templates for states: 1 - ON"""
        try:
            self.crop_freq(freq, freq_low, freq_high)
            X_f = np.vstack(on_templates)  # (n_samples, n_features)
            X = self.crop_band(X_f)
            np_stats(X, "fit X")

            self.X_train = X
            np_stats(self.X_train, "fit self.X_train")

            # normalize rows
            self.X_train_norm = self.normalizer.fit_transform(X)
            np_stats(self.X_train_norm, "fit self.X_train_norm")

            # compute features for each row
            self.X_features = FeatureAdder.add_features(X, f=self.freq)
            np_stats(self.X_features, "fit self.X_features")

        except ValueError as e:
            logger.error(str(e))

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

        try:
            if self.align_signals:
                psd, freq_shift = compute_aligned_signals_batch(self.X_train, psd)
                logger.debug(f"{freq_shift=} Hz")

            np_stats(psd, "psd")

            # compute features for each row
            psd_features = FeatureAdder.add_features(psd, f=self.freq)
            np_stats(psd_features, "psd_features")

            psd_norm = self.normalizer.transform(psd)
            np_stats(psd_norm, "psd_norm")
            np_stats(self.X_train_norm, "self.X_train_norm ")

            if self.align_signals:
                # 2. Row-wise dot product and vector norms
                similarities = np.einsum("ij,ij->i", self.X_train_norm, psd_norm)
            else:
                similarities = self.X_train_norm @ psd_norm
        except Exception as e:
            logger.error(str(e))
            # logger.error(f"Shape mismatch: X_train {self.X_train.shape}, psd {psd_norm.shape}")
            return 0, 0.0  # or some safe fallback

        sim_features = 1.0 / (1.0 + np.linalg.norm(self.X_features - psd_features))

        # Weighted combination
        similarity_weighted = self.alpha * similarities + (1 - self.alpha) * sim_features

        # 2. find top KNN
        top_k_indices = np.argsort(similarity_weighted)[-self.k :]
        top_k_sims = similarity_weighted[top_k_indices]
        top_k_sims_mean = np.mean(top_k_sims)

        if logger.getEffectiveLevel() == logging.DEBUG:
            print("similarities", similarities)
            print("psd_features", psd_features)
            print("sim_features power", sim_features)
            print("similarity_weighted", similarity_weighted)
            print("top_k_sims", top_k_sims)

        if top_k_sims_mean < self.MIN_SCORE:
            state = 0
        else:
            state = 1

        return state, top_k_sims_mean
