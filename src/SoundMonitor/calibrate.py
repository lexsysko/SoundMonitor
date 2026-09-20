import json
import logging
import numpy as np
import time

from SoundMonitor.analizer import spectral_distance
from SoundMonitor.settings import TEMPLATE_DIR, THRESHOLD_FILE
from SoundMonitor.templates import load_templates

logger = logging.getLogger(__name__)


def calibrate_from_templates(verbose: bool = True) -> dict:
    psd_on, psd_off, freqs, tmpl_sr = load_templates()

    # 1. Calculate pairwise spectral distances (Cosine Distance: 0 = identical, 1 = orthogonal)
    d_on_on = [spectral_distance(a, b) for i, a in enumerate(psd_on) for j, b in enumerate(psd_on) if i < j]
    d_off_off = [spectral_distance(a, b) for i, a in enumerate(psd_off) for j, b in enumerate(psd_off) if i < j]
    d_on_off = [spectral_distance(a, b) for a in psd_on for b in psd_off]

    # 2. Calculate mean intra-class distance (variation within the same state)
    intra_means = []
    if d_on_on:
        intra_means.append(np.mean(d_on_on))
    if d_off_off:
        intra_means.append(np.mean(d_off_off))

    mean_intra = float(np.mean(intra_means)) if intra_means else 0.0

    # 3. Calculate mean inter-class distance (separation between ON and OFF)
    if d_on_off:
        mean_on_off = float(np.mean(d_on_off))
    else:
        # Fallback if either ON or OFF templates are missing
        mean_on_off = 0.50

    # 4. Safety margin based on cluster separation
    if mean_on_off > mean_intra and mean_on_off > 0:
        SAFETY_MARGIN = (mean_on_off - mean_intra) / mean_on_off
        SAFETY_MARGIN = float(np.clip(SAFETY_MARGIN, 0.05, 0.50))
    else:
        SAFETY_MARGIN = 0.15

    # 5. Convert Cosine Distance to Cosine Similarity (Score)
    # Average similarity between distinct ON and OFF templates:
    inter_similarity = 1.0 - mean_on_off

    # Minimum similarity threshold for accepting a frame into history:
    # We set min_score above the inter-class similarity + safety margin.
    min_score = float(np.clip(inter_similarity + (SAFETY_MARGIN * 0.5), 0.55, 0.85))

    # Hysteresis Thresholds for State Filter
    on_thresh = float(np.clip(1.0 - (mean_intra * 1.5), 0.70, 0.90))
    off_thresh = float(np.clip(inter_similarity + 0.10, 0.20, 0.50))

    result = {
        "min_score": round(min_score, 3),
        "on_thresh": round(on_thresh, 3),
        "off_thresh": round(off_thresh, 3),
        "distance_on_off": round(mean_on_off, 4),
        "distance_intra": round(mean_intra, 4),
        "safety_margin": round(SAFETY_MARGIN, 2),
        "freq_count": len(freqs) if freqs is not None else 0,
        "method": "similarity_gap_calibration",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "template_sr": tmpl_sr,
    }

    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(THRESHOLD_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if verbose:
        logger.info("=" * 60)
        logger.info("AUTOMATIC THRESHOLD CALIBRATION")
        logger.info("=" * 60)
        logger.info(f"  Spectral Distance ON ↔ OFF : {mean_on_off:.4f}")
        logger.info(f"  Intra-class Distance       : {mean_intra:.4f}")
        logger.info(f"  Safety Margin              : {SAFETY_MARGIN:.2f}")
        logger.info(f"  Confidence Gate (min_score): {min_score:.3f}")
        logger.info(f"  Hysteresis ON / OFF        : {on_thresh:.2f} / {off_thresh:.2f}")
        if result["template_sr"]:
            logger.info(f"  Template Sample Rate       : {tmpl_sr} Hz")
        logger.info(f"  Saved to                   : {THRESHOLD_FILE}")

        if mean_on_off < 0.12:
            logger.warning("  ⚠  Templates very close – detection may be unreliable.")
        elif mean_on_off < 0.20:
            logger.info("  ⚠  Moderate separation between states.")
        else:
            logger.info("  ✓  Good separation between ON and OFF templates.")
        logger.info("=" * 60)

    return result
