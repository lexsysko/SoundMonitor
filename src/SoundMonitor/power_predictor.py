import logging
import numpy as np

from SoundMonitor.analizer import compute_band_power_db
from SoundMonitor.base_predictor import BasePredictor
from SoundMonitor.settings import THRESHOLD_ON_POWER

logger = logging.getLogger(__name__)


class PowerPredictor(BasePredictor):
    def __init__(self, threshold_power: float | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold_power = threshold_power or THRESHOLD_ON_POWER

    def predict(self, psd: np.ndarray, *args, **kwargs) -> tuple[int, float]:
        """
        :param psd: np.ndarray of shape (n_samples, n_dimensions)
        :return:
        - state: 1 (ON) or 0 (OFF)
        - score: average likes [0.0...1.0]
        """
        power_on_db = self.threshold_power * 0.93
        power_off_db = self.threshold_power * 1.1
        power_db = compute_band_power_db(psd, sample_spacing=1)
        logger.debug(
            f"Power: {power_db:.2f}dB, Range ON/OFF: [ {power_on_db:.2f} | {self.threshold_power:.2f} | {power_off_db:.2f} ]dB"
        )
        is_on = bool(power_db >= self.threshold_power)
        score = float(
            np.clip(
                (power_db - power_off_db) / (power_on_db - power_off_db + 1e-6),
                0.0,
                1.0,
            )
        )  # margin_db = power_db - self.threshold_power
        # score = max(0.0, 10 ** (margin_db / 10))
        return is_on, score
