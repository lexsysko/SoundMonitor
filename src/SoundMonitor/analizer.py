import logging
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

from SoundMonitor.normalizer import normalize_psd
from SoundMonitor.psd_knn import PSD_KNN
from SoundMonitor.settings import WELCH_WINDOW_SEC, get_effective_sr, FREQ_RANGE, NORMALIZE_METHOD

logger = logging.getLogger(__name__)


def nperseg_for(sr: int) -> int:
    """Welch window ≈ WELCH_WINDOW_SEC, rounded to the nearest power of 2."""
    target = int(sr * WELCH_WINDOW_SEC)
    # next power of 2 (or nearest – pick what you prefer)
    n = 1 << (target - 1).bit_length()  # ceil to 2^N
    # optional: also allow previous power of 2 if closer
    prev = n >> 1
    if prev >= 256 and abs(target - prev) < abs(target - n):
        n = prev
    return max(256, min(n, 16384))


def compute_psd(
    audio: np.ndarray, sr: int | None = None, normalize: bool = True, normalize_value: float | None = None
) -> tuple[np.ndarray, np.ndarray, bool]:
    if sr is None:
        sr = get_effective_sr()
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    sos = butter(2, 25, btype="high", fs=sr, output="sos")
    audio = sosfiltfilt(sos, audio.astype(np.float64))

    nperseg = min(nperseg_for(sr), max(256, len(audio) // 4))
    # nperseg = nperseg_for(sr)

    # logger.debug(f"compute_psd {nperseg=}  {len(audio)=}  {(len(audio)/sr)=} ")

    freqs, psd = welch(audio, fs=sr, nperseg=nperseg, scaling="density", average="median")

    logger.debug(f"compute_psd welch {len(psd)=} {len(freqs)=} {nperseg=}")

    mask = (freqs >= FREQ_RANGE[0]) & (freqs <= FREQ_RANGE[1])
    freqs, psd = freqs[mask], psd[mask]

    # logger.debug(f"compute_psd masked {len(psd)=} {len(freqs)=}  {FREQ_RANGE[0]} - {FREQ_RANGE[1]}")

    if not normalize:
        return freqs, psd.astype(np.float32), True

    return freqs, *normalize_psd(psd=psd, method=NORMALIZE_METHOD, normalize_value=normalize_value)[:2]


def predict_psd(audio: np.ndarray, psd_knn: PSD_KNN, sr: int | None = None):
    freq, psd_live, success = compute_psd(audio, sr=sr, normalize=False, normalize_value=psd_knn.global_norm)

    if not success:
        return freq, psd_live, success, 0, 0

    is_on, score = psd_knn.predict(psd_live, normalize=True)

    return freq, psd_live, success, is_on, score


def spectral_distance(psd_a: np.ndarray, psd_b: np.ndarray) -> float:
    n = min(len(psd_a), len(psd_b))
    cos = np.dot(psd_a[:n], psd_b[:n])
    return float(1.0 - np.clip(cos, -1.0, 1.0))
