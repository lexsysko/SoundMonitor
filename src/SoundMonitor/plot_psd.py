from datetime import datetime

import logging
from pathlib import Path

import numpy as np


logger = logging.getLogger(__name__)


def plot_psd_comparison(
    freqs: np.ndarray,
    psd_on: np.ndarray,
    psd_off: np.ndarray,
    psd_live: np.ndarray | None = None,
    title: str = "PSD Spectrum Comparison",
    save_path: Path | str | None = None,
):
    import matplotlib

    # Set non-interactive backend BEFORE importing pyplot or using Figure
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    # Create Figure directly (Thread-safe, bypasses Matplotlib GUI main-thread checks)
    fig = Figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    # 1. Format current timestamp
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Determine window slicing
    arrays = [freqs, psd_on, psd_off]
    if psd_live is not None:
        arrays.append(psd_live)
    n = min(len(arr) for arr in arrays)
    f = freqs[:n]

    # Plot Templates
    ax.plot(
        f,
        psd_on[:n],
        label="ON Template",
        color="crimson",
        linewidth=2,
        alpha=0.8,
    )
    ax.plot(
        f,
        psd_off[:n],
        label="OFF Template",
        color="dodgerblue",
        linewidth=2,
        alpha=0.8,
    )
    ax.fill_between(f, psd_on[:n], alpha=0.12, color="crimson")
    ax.fill_between(f, psd_off[:n], alpha=0.12, color="dodgerblue")

    # Mark ON Template Max Peak
    max_idx_on = np.argmax(psd_on)
    ax.plot(
        f[max_idx_on],
        psd_on[max_idx_on],
        "o",
        color="crimson",
        markersize=7,
        label=f"ON Max: {psd_on[max_idx_on]:.2f} @ {f[max_idx_on]:.1f}Hz",
    )

    # Plot Live Signal (if provided)
    if psd_live is not None:
        ax.plot(
            f,
            psd_live[:n],
            label="Live Signal",
            color="black",
            linestyle="--",
            linewidth=1.5,
        )
        # Mark ON Template Max Peak
        max_idx_live = np.argmax(psd_live)
        ax.plot(
            f[max_idx_live],
            psd_live[max_idx_live],
            "o",
            color="black",
            markersize=7,
            label=f"Live Max: {psd_live[max_idx_live]:.2f} @ {f[max_idx_live]:.1f}Hz",
        )

    # Formatting
    ax.set_title(title, fontsize=11, fontweight="bold")
    fig.text(0.98, 0.01, f"Generated: {timestamp_str}", fontsize=8, color="gray", ha="right")
    ax.set_xlabel("Frequency (Hz)", fontsize=10)
    ax.set_ylabel("Normalized Power", fontsize=10)
    ax.set_xlim(f[0], f[-1])
    ax.set_ylim(-0.005, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()

    # Save and cleanup explicitly
    save_path = save_path or "live.png"
    fig.savefig(save_path, dpi=150)
    logger.info(f"Saved figure to {save_path}")
