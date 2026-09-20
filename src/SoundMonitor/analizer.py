import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

from SoundMonitor.settings import WELCH_WINDOW_SEC, get_effective_sr, FREQ_RANGE


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
    audio: np.ndarray, sr: int | None = None, normalize: bool = True, normalize_log: bool = True
) -> tuple[np.ndarray, np.ndarray, bool]:
    if sr is None:
        sr = get_effective_sr()
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    sos = butter(2, 25, btype="high", fs=sr, output="sos")
    audio = sosfiltfilt(sos, audio.astype(np.float64))

    nperseg = min(nperseg_for(sr), max(256, len(audio) // 4))
    freqs, psd = welch(audio, fs=sr, nperseg=nperseg, scaling="density", average="median")

    mask = (freqs >= FREQ_RANGE[0]) & (freqs <= FREQ_RANGE[1])
    freqs, psd = freqs[mask], psd[mask]

    if not normalize:
        return freqs, psd.astype(np.float32), True

    if not normalize_log:
        peak_val = psd.max()
        if peak_val < 1e-7:  # tune to your system
            return freqs, np.zeros_like(psd), False
        psd_norm = np.linalg.norm(psd)
        return freqs, psd_norm.astype(np.float32), True

    # Normalize with log

    psd_log = 10 * np.log10(psd + 1e-12)

    if psd_log.max() - psd_log.min() > 3:  # dynamic range check
        norm = np.linalg.norm(psd_log)
        if norm > 0:
            psd_norm = psd_log / norm
            success = True
        else:
            psd_norm = np.zeros_like(psd_log)
            success = False
    else:
        psd_norm = np.zeros_like(psd_log)
        success = False

    return freqs, psd_norm.astype(np.float32), success


def spectral_distance(psd_a: np.ndarray, psd_b: np.ndarray) -> float:
    n = min(len(psd_a), len(psd_b))
    cos = np.dot(psd_a[:n], psd_b[:n])
    return float(1.0 - np.clip(cos, -1.0, 1.0))
