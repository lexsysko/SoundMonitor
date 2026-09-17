from pathlib import Path

import time

import logging

import sys

import json
import numpy as np


from SoundMonitor.settings import THRESHOLD_FILE, DEFAULT_THRESHOLD, ON_FILE, OFF_FILE, TEMPLATE_DIR

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


def load_templates() -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if not ON_FILE.exists() or not OFF_FILE.exists():
        logger.error(f"Templates not found in {TEMPLATE_DIR}")
        logger.error("Run:  python main.py train")
        sys.exit(1)
    on = np.load(ON_FILE)
    off = np.load(OFF_FILE)
    tmpl_sr = int(on["sr"]) if "sr" in on else 16000
    # Extract frequencies with fallback check
    if "freqs" in on:
        freqs = on["freqs"]
    else:
        logger.warning("Frequencies missing in template file. Re-run 'python main.py train'.")
        freqs = np.array([])  # Or handle accordingly
    return on["psd"], off["psd"], freqs, tmpl_sr


def save_template(path: Path, freqs: np.ndarray, psd: np.ndarray, sr: int, duration: float, label: str):
    np.savez_compressed(
        path, freqs=freqs, psd=psd, sr=sr, duration=duration, label=label, created=time.strftime("%Y-%m-%d %H:%M:%S")
    )
