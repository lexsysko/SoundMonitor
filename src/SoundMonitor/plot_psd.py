from pathlib import Path

import logging
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
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))

    # Calculate valid slice length
    arrays = [freqs, psd_on, psd_off]
    if psd_live is not None:
        arrays.append(psd_live)

    n = min(len(arr) for arr in arrays)
    f = freqs[:n]

    # --- 1. Plot Templates ---
    p_on = psd_on[:n]
    p_off = psd_off[:n]

    plt.plot(f, p_on, label="ON Template", color="crimson", linewidth=2, alpha=0.8)
    plt.plot(f, p_off, label="OFF Template", color="dodgerblue", linewidth=2, alpha=0.8)
    plt.fill_between(f, p_on, alpha=0.12, color="crimson")
    plt.fill_between(f, p_off, alpha=0.12, color="dodgerblue")

    # Mark ON Template Max Peak
    max_idx_on = np.argmax(p_on)
    plt.plot(
        f[max_idx_on],
        p_on[max_idx_on],
        "o",
        color="crimson",
        markersize=7,
        label=f"ON Max: {p_on[max_idx_on]:.2f} @ {f[max_idx_on]:.1f}Hz",
    )

    # --- 2. Plot Live Signal (Optional) ---
    if psd_live is not None:
        p_live = psd_live[:n]

        plt.plot(
            f,
            p_live,
            label="Live Signal",
            color="black",
            linestyle="--",
            linewidth=1.5,
        )

        # Mark Live Max Peak
        max_idx_live = np.argmax(p_live)
        max_freq_live = f[max_idx_live]
        max_val_live = p_live[max_idx_live]

        # Draw a point and a dashed line at the max peak
        plt.plot(
            max_freq_live,
            max_val_live,
            "r*",
            markersize=12,
            label=f"Live Max: {max_val_live:.2f} @ {max_freq_live:.1f}Hz",
        )
        plt.axhline(
            y=max_val_live,
            color="gray",
            linestyle=":",
            alpha=0.5,
            label=f"Max Level ({max_val_live:.2f})",
        )

    # Formatting
    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel("Frequency (Hz)", fontsize=10)
    plt.ylabel("Normalized Power", fontsize=10)
    plt.xlim(f[0], f[-1])
    # 1. Find max power among all provided arrays
    max_val = max(
        psd_on[:n].max(),
        psd_off[:n].max(),
        psd_live[:n].max() if psd_live is not None else 0.0,
    )

    # 2. Add 15% headroom above the highest peak
    y_upper = float(max(max_val * 1.15, 0.1))  # Keeps a minimum scale of 0.1 for visibility
    logger.debug(f"{y_upper=}")

    plt.ylim(-0.005, y_upper)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"Saved figure to {save_path}")
    else:
        plt.show()
