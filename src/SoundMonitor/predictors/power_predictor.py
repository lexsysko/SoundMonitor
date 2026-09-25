import logging
import numpy as np

from SoundMonitor.analize.analizer import compute_band_power_db
from SoundMonitor.predictors.base_predictor import BasePredictor
from SoundMonitor.settings import THRESHOLD_ON_POWER_DB, THRESHOLD_ON_POWER_UP, THRESHOLD_ON_POWER_DOWN

logger = logging.getLogger(__name__)


class PowerPredictor(BasePredictor):
    def __init__(self, threshold_power: float | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold_power = threshold_power or (THRESHOLD_ON_POWER_DB or -61)
        self.power_on_db = self.threshold_power * (THRESHOLD_ON_POWER_UP or 0.93)
        self.power_off_db = self.threshold_power * (THRESHOLD_ON_POWER_DOWN or 1.1)

    def predict(self, psd: np.ndarray, *args, **kwargs) -> tuple[int, float]:
        """
        :param psd: np.ndarray of shape (n_samples, n_dimensions)
        :return:
        - state: 1 (ON) or 0 (OFF)
        - score: average likes [0.0...1.0]
        """
        smoothed = kwargs.get("smoothed")
        power_db = compute_band_power_db(psd, sample_spacing=1)
        logger.debug(
            f"Power: {power_db:.2f}dB, Range ON/OFF: [ {self.power_on_db:.2f} | {self.threshold_power:.2f} | {self.power_off_db:.2f} ]dB"
        )
        is_on = bool(power_db >= self.threshold_power)
        score = float((power_db - self.power_off_db) / (self.power_on_db - self.power_off_db + 1e-6))
        if smoothed is not None:
            logger.debug(f"raw score {score:.4f} {smoothed=}")
            if not smoothed and score > 1.3:
                score = 0.0
                is_on = False
                logger.warning(f"raw score penalties to 0.0000 as too high power level")
        # clip to 0...1
        score = max(0.0, min(score, 1.0))

        # margin_db = power_db - self.threshold_power
        # score = max(0.0, 10 ** (margin_db / 10))
        return is_on, score
