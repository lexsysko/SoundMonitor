from pathlib import Path

import time

import logging

import sys

import json
import numpy as np

from SoundMonitor.enums import NormalizeMethod
from SoundMonitor.settings import (
    THRESHOLD_FILE,
    DEFAULT_THRESHOLD,
    TEMPLATE_DIR,
    TEMPLATE_FILE_MAKS,
    TEMPLATE_LABELS,
)

logger = logging.getLogger(__name__)


def load_threshold(override: float | None = None) -> float:
    if override is not None:
        return override
    if THRESHOLD_FILE.exists():
        try:
            with open(THRESHOLD_FILE, encoding="utf-8") as f:
                data = json.load(f)
            t = float(data["threshold"])
            logger.info(f"  Using calibrated threshold: {t:.3f}")
            return t
        except Exception as e:
            logger.info(f"  Warning: could not read {THRESHOLD_FILE}: {e}")
    logger.info(f"  Using default threshold: {DEFAULT_THRESHOLD:.3f}")
    return DEFAULT_THRESHOLD


def load_template(label="on"):
    files = TEMPLATE_DIR.glob(TEMPLATE_FILE_MAKS.format(label=label, idx="*"))
    templates = []
    for f in files:
        arr = np.load(f, allow_pickle=True)
        # If saved with savez, arr is a np.lib.npyio.NpzFile
        if isinstance(arr, np.lib.npyio.NpzFile):
            psd = arr["psd"]
            templates.append(
                {
                    "psd": np.array(psd, dtype=np.float32),
                    "freqs": arr.get("freqs"),
                    "sr": arr.get("sr"),
                    "label": arr.get("label", label),
                }
            )
        else:
            # If saved with plain np.save
            templates.append(
                {
                    "psd": np.array(arr, dtype=np.float32),
                    "freqs": None,
                    "sr": None,
                    "label": label,
                }
            )
    return templates


def load_templates() -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, int]:
    labels = TEMPLATE_LABELS
    templates = {}
    for label in labels:
        if loaded := load_template(label=label):
            templates[label] = loaded

    if not all(label in templates for label in TEMPLATE_LABELS):
        logger.error(f"Templates not found in {TEMPLATE_DIR}")
        logger.error("Run:  python main.py train")
        # raise RuntimeError()
        sys.exit(1)

    first_label = labels[0]
    first_template = templates[first_label][0]

    tmpl_sr = int(first_template["sr"]) if "sr" in first_template else 16000
    # Extract frequencies with fallback check
    if first_template and "freqs" in first_template:
        freqs = first_template["freqs"]
    else:
        logger.warning("Frequencies missing in template file. Re-run 'python main.py train'.")
        freqs = np.array([])  # Or handle accordingly

    psd_on = [np.array(item["psd"], dtype=np.float32) for item in templates["on"]]
    psd_off = [np.array(item["psd"], dtype=np.float32) for item in templates["off"]]

    return psd_on, psd_off, freqs, tmpl_sr


def save_template(
    freqs: np.ndarray,
    psd: np.ndarray,
    sr: int,
    duration: float,
    label="on",
    idx=0,
    normalize: bool = True,
    normalize_method: NormalizeMethod = NormalizeMethod.NONE,
):
    idx_str = f"{idx:03d}"
    path = TEMPLATE_DIR / TEMPLATE_FILE_MAKS.format(label=label.lower(), idx=idx_str)
    np.savez_compressed(
        path,
        freqs=freqs,
        psd=psd,
        sr=sr,
        duration=duration,
        label=label,
        normalize=int(normalize),
        normalize_method=str(normalize_method),
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    logger.info(f"  Saved template → {path}  (sr={sr})")


def prune_templates(label, idx=None) -> int:

    idx_str = "*" if idx is None else f"{idx:03d}"
    count = 0
    for file in TEMPLATE_DIR.glob(TEMPLATE_FILE_MAKS.format(label=label.lower(), idx=idx_str)):
        try:
            file.unlink()
            count += 1
        except (FileNotFoundError, OSError):
            ...
    return count
