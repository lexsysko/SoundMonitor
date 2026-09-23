from datetime import datetime

import logging
from pathlib import Path

import numpy as np

from SoundMonitor.enums import NormalizeMethod
from SoundMonitor.normalizer import normalize_psd, get_concatenated_norm
from SoundMonitor.settings import NORMALIZE_METHOD

logger = logging.getLogger(__name__)


def plot_psd_comparison(
    freqs: np.ndarray,
    templates: dict[str, np.ndarray],
    psd_live: np.ndarray | None = None,
    title: str = "PSD Spectrum Comparison",
    save_path: Path | str | None = None,
    normalize_value: float | None = None,
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

    normalize_method = NormalizeMethod.NONE

    normalize_value = normalize_value or get_concatenated_norm(*templates.values(), method=normalize_method)
    # n_on, success_on, normalize_value_on = normalize_psd(
    #     templates["on"], method=NORMALIZE_METHOD, normalize_value=normalize_value
    # )
    # n_off, success_off, normalize_value_off = normalize_psd(templates["off"], method=NORMALIZE_METHOD)
    # logger.debug(
    #     f"{NORMALIZE_METHOD.value=} {normalize_value=} {normalize_value_on=} {success_on=}  {normalize_value_off=} {success_off=} "
    # )
    # logger.debug(f"{templates["on"].max()=} {templates["on"].min()=} ")
    # logger.debug(f"{templates["off"].max()=} {templates["off"].min()=} ")
    # logger.debug(f"{n_on.max()=} {n_on.min()=} ")
    # logger.debug(f"{n_off.max()=} {n_off.min()=} ")

    # Plot each template
    colors = {"on": "crimson", "off": "dodgerblue"}
    y_min = 0
    y_max = 0
    for label, psd in templates.items():
        psd = normalize_psd(psd[:n], normalize_value=normalize_value, method=normalize_method)[0]
        y_min = min(psd.min(), y_min)
        y_max = max(psd.max(), y_max)
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
        psd_live = normalize_psd(psd_live[:n], normalize_value=normalize_value, method=normalize_method)[0]
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
    if psd_live is not None and len(psd_live) > 0:
        y_min = min(y_min, np.min(psd_live[:n]))
        y_max = max(y_max, np.max(psd_live[:n]))
    # logger.debug(f"{y_min=} {y_max=}")
    ax.set_ylim(y_min - 0.05 * abs(y_min), y_max + 0.05 * abs(y_max))
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    save_path = save_path or "live.png"
    fig.savefig(save_path, dpi=150)
    logger.info(f"Saved figure to {save_path}")
