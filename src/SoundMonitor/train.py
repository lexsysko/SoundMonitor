import logging
import numpy as np
from time import sleep

from SoundMonitor.analizer import compute_psd
from SoundMonitor.audio_device import list_input_devices, record_seconds, play_beep
from SoundMonitor.calibrate import calibrate_from_templates
from SoundMonitor.enums import NormalizeMethod
from SoundMonitor.settings import (
    MIN_RECORD_SEC,
    PREFERRED_RATES,
    TRAIN_DURATION,
    TEMPLATE_COUNTERS_DELAY_SEC,
    TEMPLATE_COUNTERS,
    TEMPLATE_LABELS,
    NORMALIZE_METHOD,
)
from SoundMonitor.templates import save_template, prune_templates

logger = logging.getLogger(__name__)


def capture_label(
    duration, device_index, label: str, idx, confirm: bool = True, normalize: bool = False, beep: bool = False
) -> bool:
    if confirm:
        input(f"\n>>> Prepare '{label} [{idx}]' state, then press Enter to start recording…")
    else:
        logger.info(f"Start recording for '{label} [{idx}]' {normalize=}...")
    if beep:
        play_beep()
    audio, sr = record_seconds(duration, device_index)
    if len(audio) < sr * MIN_RECORD_SEC:
        logger.warning("Recording too short – try again.")
        return False
    freqs, psd, success = compute_psd(audio, sr=sr, normalize=normalize)
    if not success:
        logger.warning("Recording may be muted, try again.")
        return False
    save_template(
        freqs=freqs,
        psd=psd,
        sr=sr,
        duration=duration,
        label=label,
        idx=idx,
        normalize=normalize,
        normalize_method=NORMALIZE_METHOD if normalize else NormalizeMethod.NONE,
    )
    peaks = freqs[np.argsort(psd)[::-1][:5]]
    logger.info(f"  Top peaks (Hz) from {len(freqs)}: {', '.join(f'{p:.0f}' for p in peaks)}")
    return True


def train(
    device_index: int | None = None,
    duration: float = TRAIN_DURATION,
    mode: str = "auto",
    count: int | None = None,
    delay: int | None = None,
    prune: bool = False,
    beep: bool = True,
) -> None:
    list_input_devices()

    buff = "\n" + "=" * 60
    buff += f"\nTRAINING MODE: {mode}"
    buff += f"\nEach recording lasts {duration:.0f} seconds."
    buff += f"\nWill try sample rates: {PREFERRED_RATES}\n"
    buff += "=" * 60
    logger.info(buff)

    match mode:
        case "on":
            labels = ("on",)
        case "off":
            labels = ("off",)
        case _:
            labels: tuple[str] = TEMPLATE_LABELS

    for label in labels:
        train_count = count or TEMPLATE_COUNTERS.get(label, 2)
        train_delay = delay or TEMPLATE_COUNTERS_DELAY_SEC.get(label, 30)
        buff = "\n" + "=" * 60
        buff += f"\n  Compressor {label.upper()}. Total templates: ({train_count}) with delay {train_delay} seconds"
        if prune:
            count = prune_templates(label=label)
            buff += f"\n Pruned total: {count} previous templates for label: {label}"
        logger.info(buff)
        for idx in range(train_count):
            while True:
                if capture_label(duration, device_index, label, idx, confirm=(idx == 0), beep=beep):
                    logger.info(f"Sleeping {train_delay} seconds ...")
                    if idx < train_count - 1:
                        sleep(train_delay)
                    break
                else:
                    logger.info("Sleeping 5 seconds ...")
                    sleep(5)

    calibrate_from_templates(verbose=True)
    logger.info("Training + calibration finished.")
    logger.info("Run:  python main.py detect")
