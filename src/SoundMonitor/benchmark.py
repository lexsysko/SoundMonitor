#!/usr/bin/env python3
"""Compare compute_psd + spectral_distance: direct vs asyncio.to_thread."""

import asyncio
import time
import numpy as np
from scipy.signal import welch, butter, sosfilt

SR = 44100
WINDOW_SEC = 1.5
N = int(SR * WINDOW_SEC)
WELCH_WINDOW_SEC = 0.256
FREQ_RANGE = (60.0, 1000.0)
ROUNDS = 1000


def nperseg_for(sr: int) -> int:
    target = int(sr * WELCH_WINDOW_SEC)
    n = 1 << (target - 1).bit_length()
    prev = n >> 1
    if prev >= 256 and abs(target - prev) < abs(target - n):
        n = prev
    return max(256, min(n, 16384))


def compute_psd(audio: np.ndarray, sr: int = SR):
    sos = butter(2, 25, btype="high", fs=sr, output="sos")
    audio = sosfilt(sos, audio.astype(np.float64))
    nperseg = min(nperseg_for(sr), max(256, len(audio) // 4))
    freqs, psd = welch(audio, fs=sr, nperseg=nperseg, scaling="density")
    mask = (freqs >= FREQ_RANGE[0]) & (freqs <= FREQ_RANGE[1])
    psd = psd[mask]
    psd = psd / (psd.max() + 1e-12)
    return freqs[mask], psd.astype(np.float32)


def spectral_distance(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    return float(
        1.0 - np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12))
    )


# fake templates + audio
rng = np.random.default_rng(0)
audio = rng.standard_normal(N).astype(np.float32) * 0.1
_, psd_on = compute_psd(audio)
psd_off = psd_on * 0.7 + 0.1
audio2 = rng.standard_normal(N).astype(np.float32) * 0.1


def run_direct():
    _, psd = compute_psd(audio2, SR)
    n = min(len(psd), len(psd_on), len(psd_off))
    d_on = spectral_distance(psd[:n], psd_on[:n])
    d_off = spectral_distance(psd[:n], psd_off[:n])
    return d_on, d_off


async def run_threaded(spectral_distance_threaded: bool = False):
    _, psd = await asyncio.to_thread(compute_psd, audio2, SR)
    n = min(len(psd), len(psd_on), len(psd_off))
    if spectral_distance_threaded:
        d_on, d_off = await asyncio.gather(
            asyncio.to_thread(spectral_distance, psd[:n], psd_on[:n]),
            asyncio.to_thread(spectral_distance, psd[:n], psd_off[:n]),
        )
    else:
        d_on = spectral_distance(psd[:n], psd_on[:n])
        d_off = spectral_distance(psd[:n], psd_off[:n])
    return d_on, d_off


def bench_sync(fn, rounds=30):
    warmup = max(5, int(rounds * 0.15))
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(rounds):
        fn()
    return (time.perf_counter() - t0) / rounds * 1000  # ms


async def bench_async(coro_fn, rounds=30, *args, **kwargs):
    warmup = max(5, int(rounds * 0.15))
    for _ in range(warmup):
        await coro_fn(*args, **kwargs)
    t0 = time.perf_counter()
    for _ in range(rounds):
        await coro_fn(*args, **kwargs)
    return (time.perf_counter() - t0) / rounds * 1000


async def main():
    print(
        f"SR={SR}, window={WINDOW_SEC}s, samples={N}, nperseg≈{nperseg_for(SR)}, {ROUNDS=}"
    )
    ms_direct = bench_sync(run_direct, rounds=ROUNDS)
    print(f"direct (blocking):     {ms_direct:7.2f} ms / analysis")
    ms_thread_0 = await bench_async(
        run_threaded, rounds=ROUNDS, spectral_distance_threaded=False
    )
    print(f"\nto_thread compute_psd:    {ms_thread_0:7.2f} ms / analysis")
    print(
        f"difference:            {ms_thread_0 - ms_direct:+.2f} ms  "
        f"({(ms_thread_0 / ms_direct - 1) * 100:+.0f}%)"
    )
    ms_thread_1 = await bench_async(
        run_threaded, rounds=ROUNDS, spectral_distance_threaded=True
    )
    print(
        f"\nto_thread + gather spectral_distances:    {ms_thread_1:7.2f} ms / analysis"
    )
    print(
        f"difference:            {ms_thread_1 - ms_direct:+.2f} ms  "
        f"({(ms_thread_1 / ms_direct - 1) * 100:+.0f}%)"
    )
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
