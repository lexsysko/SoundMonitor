import logging

import time

import asyncio
from collections import deque

import numpy as np

from dataclasses import dataclass

from SoundMonitor.analizer import compute_psd, spectral_distance
from SoundMonitor.async_mic import AsyncMic

from SoundMonitor.settings import (
    SMOOTH_WINDOWS,
    MIN_CONFIRM,
    THRESHOLD_ON,
    THRESHOLD_OFF,
    ANALYSIS_WINDOW_SEC,
    DETECT_INTERVAL_SEC,
    PERIODIC_WRITE_TIME,
    DATA_PATH,
    PLOT_LIVE_FILENAME,
)
from SoundMonitor.plot_psd import plot_psd_comparison

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    timestamp: float
    state: bool
    d_on: float
    d_off: float


async def detection_loop(
    mic: AsyncMic,
    psd_on: np.ndarray,
    psd_off: np.ndarray,
    threshold: float,
    result_queue: asyncio.Queue,
    window_sec: float = ANALYSIS_WINDOW_SEC,
    interval_sec: float = DETECT_INTERVAL_SEC,
    plot: bool = False,
) -> None:
    history: deque[bool] = deque(maxlen=SMOOTH_WINDOWS)
    last_smoothed: bool | None = None
    last_written: bool | None = None  # last state sent to DB
    min_samples = int(mic.sr * 0.8)
    last_written_time: float = 0

    logger.info(f"  Detection loop running @ {mic.sr} Hz with threshold: {threshold:.4f}, {plot=} (Ctrl+C to stop)")
    try:
        while True:
            audio = mic.get_recent(window_sec)
            if len(audio) < min_samples:
                await asyncio.sleep(0.2)
                continue

            freq, psd = await asyncio.to_thread(compute_psd, audio, sr=mic.sr)
            n = min(len(psd), len(psd_on), len(psd_off))
            # fast, can stay in-loop
            d_on = spectral_distance(psd[:n], psd_on[:n])
            d_off = spectral_distance(psd[:n], psd_off[:n])
            logger.debug(f"freq Hz: {[float(f'{x:10.6f}'[:10]) for x in freq[2:8]]}")
            logger.debug(f"psd    : {psd[2:8]}")
            logger.debug(f"psd_on : {psd_on[2:8]}")
            logger.debug(f"psd_off: {psd_off[2:8]}")

            # Hysteresis: different decision boundary depending on current state
            if last_smoothed:  # currently ON → make it harder to turn OFF
                d_off_tr = d_off * threshold * THRESHOLD_OFF
                is_on = d_on < d_off_tr
            else:  # currently OFF (or first sample) → make it harder to turn ON
                d_off_tr = d_off * threshold * THRESHOLD_ON
                is_on = d_on < d_off_tr

            history.append(is_on)
            history_ready = len(history) == SMOOTH_WINDOWS

            # Strong majority required to change state
            votes_on = sum(history)
            votes_min_confirmed = votes_on >= MIN_CONFIRM
            if votes_min_confirmed:
                smoothed = True
            elif votes_on <= (SMOOTH_WINDOWS - MIN_CONFIRM):
                smoothed = False
            else:
                # stay in previous state (hysteresis zone)
                smoothed = last_smoothed if last_smoothed is not None else False

            # Logging
            state_str = "ON " if is_on else "off"
            marker = " <<<" if smoothed != last_smoothed and last_smoothed is not None else ""

            logger.debug(
                # f"[{time.strftime('%H:%M:%S')}]  "
                f"d_on={d_on:.3f}  d_off={d_off:.3f}  d_off_tr={d_off_tr:.3f} → {state_str} {smoothed} {marker}",
            )

            last_smoothed = smoothed
            periodic_write = (time.monotonic() - last_written_time) > PERIODIC_WRITE_TIME

            # --- DB only on trusted state change ---
            if history_ready and (last_written is None or smoothed != last_written or periodic_write):
                result = DetectionResult(
                    timestamp=time.time(),
                    state=smoothed,  # trusted state
                    d_on=d_on,
                    d_off=d_off,
                )
                await result_queue.put(result)
                last_written_time = time.monotonic()
                last_written = smoothed
                if plot:
                    asyncio.create_task(
                        asyncio.to_thread(
                            plot_psd_comparison,
                            freqs=freq,
                            psd_on=psd_on,
                            psd_off=psd_off,
                            psd_live=psd,
                            save_path=DATA_PATH / PLOT_LIVE_FILENAME,
                        )
                    )

            await asyncio.sleep(interval_sec)
    except asyncio.CancelledError:
        ...
