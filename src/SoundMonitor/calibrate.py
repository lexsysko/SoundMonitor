import logging

import json

import time

import numpy as np

from SoundMonitor.analizer import spectral_distance
from SoundMonitor.settings import ON_FILE, OFF_FILE, SAFETY_MARGIN, TEMPLATE_DIR, THRESHOLD_FILE

logger = logging.getLogger(__name__)


def calibrate_from_templates(verbose: bool = True) -> dict:
    if not ON_FILE.exists() or not OFF_FILE.exists():
        raise FileNotFoundError(f"Need both {ON_FILE.name} and {OFF_FILE.name}. Run 'train' first.")

    on = np.load(ON_FILE)
    off = np.load(OFF_FILE)
    d_on_off = spectral_distance(on["psd"], off["psd"])
    threshold = float(np.clip(1.0 * (1.0 - SAFETY_MARGIN), 0.35, 0.90))

    result = {
        "threshold": threshold,
        "distance_on_off": round(d_on_off, 4),
        "safety_margin": SAFETY_MARGIN,
        "method": "midpoint_with_safety_margin",
        "rule": "is_on = (d_on < d_off * threshold)",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "template_sr": int(on["sr"]) if "sr" in on else None,
    }

    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(THRESHOLD_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if verbose:
        logger.info("=" * 60)
        logger.info("AUTOMATIC THRESHOLD CALIBRATION")
        logger.info("=" * 60)
        logger.info(f"  Spectral distance ON ↔ OFF : {d_on_off:.4f}")
        logger.info(f"  Safety margin              : {SAFETY_MARGIN:.2f}")
        logger.info(f"  Calibrated threshold       : {threshold:.3f}")
        if result["template_sr"]:
            logger.info(f"  Template sample rate       : {result['template_sr']} Hz")
        logger.info(f"  Saved to                   : {THRESHOLD_FILE}")
        if d_on_off < 0.12:
            logger.warning("  ⚠  Templates very close – detection may be unreliable.")
        elif d_on_off < 0.20:
            logger.info("  ⚠  Moderate separation.")
        else:
            logger.info("  ✓  Good separation between ON and OFF templates.")
        logger.info("=" * 60)

    return result
