from datetime import datetime

import logging
from pathlib import Path

import numpy as np


logger = logging.getLogger(__name__)


def plot_psd_comparison(
    freqs: np.ndarray,
    templates: dict[str, np.ndarray],
    psd_live: np.ndarray | None = None,
    title: str = "PSD Spectrum Comparison",
    save_path: Path | str | None = None,
):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    fig = Figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    arrays = [freqs] + list(templates.values())
    if psd_live is not None:
        arrays.append(psd_live)
    n = min(len(arr) for arr in arrays)
    f = freqs[:n]

    # Plot each template
    colors = {"on": "crimson", "off": "dodgerblue"}
    for label, psd in templates.items():
        psd = psd[:n]
        ax.plot(f, psd, label=f"{label.upper()} Template", color=colors.get(label, None), linewidth=2, alpha=0.8)
        ax.fill_between(f, psd, alpha=0.12, color=colors.get(label, "gray"))
        if len(psd) > 0:
            max_idx = np.argmax(psd)
            ax.plot(
                f[max_idx],
                psd[max_idx],
                "o",
                color=colors.get(label, "gray"),
                markersize=7,
                label=f"{label.upper()} Max: {psd[max_idx]:.2f} @ {f[max_idx]:.1f}Hz",
            )

    # Plot live signal
    if psd_live is not None and len(psd_live) > 0:
        psd_live = psd_live[:n]
        ax.plot(f, psd_live, label="Live Signal", color="black", linestyle="--", linewidth=1.5)
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
    # Dynamic y-limits
    ymin = min(np.min(psd[:n]) for psd in templates.values())
    ymax = max(np.max(psd[:n]) for psd in templates.values())
    if psd_live is not None:
        ymin = min(ymin, np.min(psd_live[:n]))
        ymax = max(ymax, np.max(psd_live[:n]))
    ax.set_ylim(ymin - 0.05 * abs(ymin), ymax + 0.05 * abs(ymax))
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    save_path = save_path or "live.png"
    fig.savefig(save_path, dpi=150)
    logger.info(f"Saved figure to {save_path}")
