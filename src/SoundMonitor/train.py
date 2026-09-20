from time import sleep

import logging
import numpy as np

from SoundMonitor.analizer import compute_psd
from SoundMonitor.audio_device import list_input_devices, record_seconds
from SoundMonitor.calibrate import calibrate_from_templates
from SoundMonitor.settings import MIN_RECORD_SEC, PREFERRED_RATES, TRAIN_DURATION, TEMPLATE_LABELS
from SoundMonitor.templates import save_template

logger = logging.getLogger(__name__)


def train(device_index: int | None = None, duration: float = TRAIN_DURATION, train_mode: str = "auto") -> None:
    list_input_devices()
    counters: dict[str, int] = {
        "on": 30,
        "off": 5,
    }
    counters_delay_sec: dict[str, int] = {
        "on": 30,
        "off": 60,
    }

    def capture_label(label: str, idx, confirm: bool = True) -> bool:
        if confirm:
            input(f"\n>>> Prepare '{label} [{idx}]' state, then press Enter to start recording…")
        else:
            logger.info(f"Start recording for'{label} [{idx}]' ...")
        audio, sr = record_seconds(duration, device_index)
        if len(audio) < sr * MIN_RECORD_SEC:
            logger.warning("Recording too short – try again.")
            return False
        freqs, psd, success = compute_psd(audio, sr=sr)
        if not success:
            logger.warning("Recording may be muted, try again.")
            return False
        save_template(freqs=freqs, psd=psd, sr=sr, duration=duration, label=label, idx=idx)
        peaks = freqs[np.argsort(psd)[::-1][:5]]
        logger.info(f"  Top peaks (Hz): {', '.join(f'{p:.0f}' for p in peaks)}")
        return True

    buff = "\n" + "=" * 60
    buff += f"\nTRAINING MODE: {train_mode}"
    buff += f"\nEach recording lasts {duration:.0f} seconds."
    buff += f"\nWill try sample rates: {PREFERRED_RATES}\n"
    buff += "=" * 60
    logger.info(buff)

    match train_mode:
        case "auto":
            buff = "\n" + "=" * 60
            buff += f"\n  1. Compressor ON. Total templates: ({counters['on']}) with delay {counters_delay_sec['on']} seconds"
            buff += f"\n  2. Compressor OFF. Total templates: ({counters['off']}) with delay {counters_delay_sec['off']} seconds"

            logger.info(buff)
            for label in TEMPLATE_LABELS:
                for idx in range(counters[label]):
                    while True:
                        if capture_label(label, idx, confirm=(idx == 0)):
                            sleep(counters_delay_sec[label])
                            break
                        else:
                            logger.info("Sleeping 5 seconds ...")
                            sleep(5)

    calibrate_from_templates(verbose=True)
    logger.info("Training + calibration finished.")
    logger.info("Run:  python main.py detect")
