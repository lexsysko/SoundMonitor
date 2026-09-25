import logging
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

from SoundMonitor.base_predictor import BasePredictor
from SoundMonitor.normalizer import normalize_psd
from SoundMonitor.psd_knn import PSD_KNN
from SoundMonitor.settings import (
    WELCH_WINDOW_SEC,
    get_effective_sr,
    FREQ_RANGE,
    NORMALIZE_METHOD,
    THRESHOLD_ON_POWER_DB,
)

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

    # freqs, psd = welch(audio, fs=sr, nperseg=nperseg, scaling="density", average="median")
    freqs, psd = welch(audio, fs=sr, nperseg=nperseg, scaling="density")

    logger.debug(f"compute_psd welch {len(psd)=} {len(freqs)=} {nperseg=}")

    mask = (freqs >= FREQ_RANGE[0]) & (freqs <= FREQ_RANGE[1])
    freqs, psd = freqs[mask], psd[mask]

    # logger.debug(f"compute_psd masked {len(psd)=} {len(freqs)=}  {FREQ_RANGE[0]} - {FREQ_RANGE[1]}")

    if not normalize:
        return freqs, psd.astype(np.float32), True

    return freqs, *normalize_psd(psd=psd, method=NORMALIZE_METHOD, normalize_value=normalize_value)[:2]


def predict_psd(audio: np.ndarray, predictor: BasePredictor, sr: int | None = None, *args, **kwargs):
    freq, psd_live, success = compute_psd(audio, sr=sr, normalize=False)

    if not success:
        return freq, psd_live, success, 0, 0

    is_on, score = predictor.predict(psd_live, *args, **kwargs)

    return freq, psd_live, success, is_on, score


def spectral_distance(psd_a: np.ndarray, psd_b: np.ndarray) -> float:
    n = min(len(psd_a), len(psd_b))
    cos = np.dot(psd_a[:n], psd_b[:n])
    return float(1.0 - np.clip(cos, -1.0, 1.0))


def extract_coarse_bands(psd_array, num_bands=4):
    """Splits a wide ~100Hz spectrum into broad sub-bands.

    Converts shape (N, M) -> (N, num_bands)
    """
    if psd_array.ndim == 1:
        psd_array = psd_array[None, :]

    # Split array into equal broad frequency bands and sum energy
    band_chunks = np.array_split(psd_array, num_bands, axis=1)
    coarse_features = np.column_stack([np.sum(chunk, axis=1) for chunk in band_chunks])

    # Normalize feature vectors for KNN
    norms = np.linalg.norm(coarse_features, axis=1, keepdims=True)
    return coarse_features / (norms + 1e-10)


def compute_band_power_db(psd_array: np.ndarray, sample_spacing: float = 0.5) -> float:
    """Calculates total integrated power in a wide frequency band in dB.

    psd_array: (N, M) matrix or (M,) vector of PSD values
    sample_spacing: frequency bin width (Hz)

    Returns: Scalar or (N,) array of total band powers
    """
    # Total area under the PSD curve using Trapezoidal integration
    total_power = np.trapezoid(psd_array, dx=sample_spacing, axis=-1)
    power_db = 10 * np.log10(total_power + 1e-12)
    return power_db
