from pathlib import Path

import logging
import numpy as np

from SoundMonitor.analizer import compute_psd
from SoundMonitor.audio_device import list_input_devices, record_seconds
from SoundMonitor.calibrate import calibrate_from_templates
from SoundMonitor.settings import MIN_RECORD_SEC, PREFERRED_RATES, ON_FILE, OFF_FILE, TRAIN_DURATION
from SoundMonitor.templates import save_template

logger = logging.getLogger(__name__)


def train(device_index: int | None = None, duration: float = TRAIN_DURATION) -> None:
    list_input_devices()

    def capture_label(label: str, path: Path) -> None:
        input(f"\n>>> Prepare '{label}' state, then press Enter to start recording…")
        audio, sr = record_seconds(duration, device_index)
        if len(audio) < sr * MIN_RECORD_SEC:
            logger.warning("Recording too short – try again.")
            return
        freqs, psd = compute_psd(audio, sr=sr)
        save_template(path, freqs=freqs, psd=psd, sr=sr, duration=duration, label=label)
        logger.info(f"  Saved template → {path}  (sr={sr})")
        peaks = freqs[np.argsort(psd)[::-1][:5]]
        logger.info(f"  Top peaks (Hz): {', '.join(f'{p:.0f}' for p in peaks)}")

    buff = "\n" + "=" * 60
    buff += "\nTRAINING MODE"
    buff += "\n  1. Compressor ON"
    buff += "\n  2. Compressor OFF"
    buff += f"\nEach recording lasts {duration:.0f} seconds."
    buff += f"\nWill try sample rates: {PREFERRED_RATES}\n"
    buff += "=" * 60
    logger.info(buff)

    capture_label("ON", ON_FILE)
    capture_label("OFF", OFF_FILE)

    if ON_FILE.exists() and OFF_FILE.exists():
        calibrate_from_templates(verbose=True)
    logger.info("Training + calibration finished.")
    logger.info("Run:  python main.py detect")
