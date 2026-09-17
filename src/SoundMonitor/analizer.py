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


def compute_psd(audio: np.ndarray, sr: int | None = None) -> tuple[np.ndarray, np.ndarray]:

    if sr is None:
        sr = get_effective_sr()
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # 1. High-pass filter with zero phase distortion
    sos = butter(2, 25, btype="high", fs=sr, output="sos")
    audio = sosfiltfilt(sos, audio.astype(np.float64))

    # 2. Welch PSD with Median segment averaging
    nperseg = min(nperseg_for(sr), max(256, len(audio) // 4))
    freqs, psd = welch(audio, fs=sr, nperseg=nperseg, scaling="density", average="median")

    # 3. Frequency masking
    mask = (freqs >= FREQ_RANGE[0]) & (freqs <= FREQ_RANGE[1])
    freqs = freqs[mask]
    psd = psd[mask]

    # 4. Threshold-aware normalization to avoid boosting background noise
    peak_val = psd.max()
    if peak_val > 1e-7:  # Tune this noise floor to your system
        psd = psd / peak_val
    else:
        psd = np.zeros_like(psd)

    return freqs, psd.astype(np.float32)


def spectral_distance(psd_a: np.ndarray, psd_b: np.ndarray) -> float:
    a = psd_a.astype(np.float64)
    b = psd_b.astype(np.float64)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    cos = np.dot(a, b) / (na * nb)
    return float(1.0 - cos)
