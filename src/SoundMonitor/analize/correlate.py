import numpy as np
from scipy.signal import correlate, correlation_lags, fftconvolve


def compute_aligned_signals_batch(A, b, sample_spacing=0.1):
    """Calculates optimal frequency shifts and aligns a 2D matrix of templates A (N, M)

    against a single 1D live signal b (M).

    Parameters:
    - A: 2D array of shape (N, M) representing N template PSD signals
    - b: 1D array of shape (M,) representing 1 live PSD signal
    - sample_spacing: Frequency step size in Hz

    Returns:
    - aligned_B: 2D array of shape (N, M) - live signal aligned to each template
    - freq_shifts: 1D array of shape (N,) - detected shift in Hz for each template
    """
    # Force 2D input for A if 1D array is passed
    if A.ndim == 1:
        A = np.atleast_2d(A)

    N, M = A.shape

    # 1. Zero-mean normalization along feature axis (M)
    A_norm = A - np.mean(A, axis=1, keepdims=True)
    b_norm = b - np.mean(b)

    # 2. Vectorized 1D cross-correlation along axis=1 using FFT
    # Cross-correlation between A[i] and b is equal to convolution with flipped b
    b_flipped_2d = b_norm[::-1][None, :]
    corr = fftconvolve(A_norm, b_flipped_2d, mode="full", axes=1)
    lags = correlation_lags(M, M, mode="full")

    # 3. Find optimal lag index for each template
    best_lag_indices = np.argmax(corr, axis=1)
    best_lags = lags[best_lag_indices]
    freq_shifts = best_lags * sample_spacing

    # 4. Construct fixed (N, M) aligned matrices
    aligned_B = np.zeros((N, M))

    for i in range(N):
        lag = best_lags[i]

        if lag > 0:
            # Live signal b shifted right
            aligned_B[i] = np.pad(b, (lag, 0))[:M]
        elif lag < 0:
            # Live signal b shifted left
            b_shifted = b[-lag:]
            aligned_B[i] = np.pad(b_shifted, (0, -lag))[:M]
        else:
            aligned_B[i] = b

    return aligned_B, freq_shifts


def compute_aligned_signals(a, b, sample_spacing=0.1):
    """Calculates peak frequency shift and shape similarity using cross-correlation.

    Parameters:
    - a: 1D array of template PSD values
    - b: 1D array of live PSD values
    - sample_spacing: Frequency step size (e.g., 0.1 Hz per point)

    Returns:
    - aligned_a: values
    - aligned_b: values
    - freq_shift: Detected frequency shift in Hz
    """

    # 2. Compute Cross-Correlation
    # Subtract means for unbiased shape correlation
    t_norm = a - np.mean(a)
    l_norm = b - np.mean(b)

    corr = correlate(t_norm, l_norm, mode="full")
    lags = correlation_lags(len(a), len(b), mode="full")

    # 3. Find optimal lag (shift index)
    best_lag_idx = np.argmax(corr)
    best_lag = lags[best_lag_idx]
    freq_shift = best_lag * sample_spacing

    # 4. Shift live signal to align with the template
    if best_lag > 0:
        aligned_b = np.pad(b, (best_lag, 0))[: len(a)]
        aligned_a = a
    elif best_lag < 0:
        aligned_b = b[-best_lag:]
        aligned_a = a[: len(aligned_b)]
    else:
        aligned_b = b
        aligned_a = a

    return aligned_a, aligned_b, freq_shift


def compute_aligned_similarity(template, live_signal, sample_spacing=0.1):
    """Calculates peak frequency shift and shape similarity using cross-correlation.

    Parameters:
    - template: 1D array of template PSD values
    - live_signal: 1D array of live PSD values
    - sample_spacing: Frequency step size (e.g., 0.1 Hz per point)

    Returns:
    - max_cosine_sim: Cosine similarity after optimal alignment (0.0 to 1.0)
    - unaligned_sim: Cosine similarity without alignment
    - freq_shift: Detected frequency shift in Hz
    """
    # 1. Standard Cosine Similarity without alignment
    unaligned_sim = np.dot(template, live_signal) / (np.linalg.norm(template) * np.linalg.norm(live_signal))

    aligned_template, aligned_live, freq_shift = compute_aligned_signals(template, live_signal)

    # 5. Compute Cosine Similarity on aligned signals
    max_cosine_sim = np.dot(aligned_template, aligned_live) / (
        np.linalg.norm(aligned_template) * np.linalg.norm(aligned_live)
    )

    return max_cosine_sim, unaligned_sim, freq_shift


# --- Example Usage with Mock Data ---
if __name__ == "__main__":
    # Generate mock frequency axis (86 Hz to 105 Hz)
    freqs = np.arange(86, 105, 0.1)

    # Simulated template (peak around 96.9 Hz)
    template = np.exp(-((freqs - 96.9) ** 2) / (2 * 2.5**2))

    # Simulated live signal (shifted peak around 94.2 Hz -> ~2.7 Hz shift)
    live_signal = 0.94 * np.exp(-((freqs - 94.2) ** 2) / (2 * 2.5**2))

    # Run comparison
    aligned_sim, unaligned_sim, shift = compute_aligned_similarity(template, live_signal, sample_spacing=0.1)

    print(f"Unaligned Cosine Similarity : {unaligned_sim:.4f}")
    print(f"Aligned Cosine Similarity   : {aligned_sim:.4f}")
    print(f"Detected Frequency Shift     : {shift:+.2f} Hz")
