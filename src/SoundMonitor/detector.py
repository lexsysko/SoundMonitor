from collections import deque

import asyncio
import logging
import numpy as np
import time
from dataclasses import dataclass

from SoundMonitor.analizer import compute_psd, predict_psd
from SoundMonitor.async_mic import AsyncMic
from SoundMonitor.plot_psd import plot_psd_comparison
from SoundMonitor.power_predictor import PowerPredictor
from SoundMonitor.psd_knn import PSD_KNN
from SoundMonitor.settings import (
    SMOOTH_WINDOWS,
    ANALYSIS_WINDOW_SEC,
    DETECT_INTERVAL_SEC,
    PERIODIC_WRITE_TIME,
    DATA_PATH,
    PLOT_LIVE_FILENAME,
    NORMALIZE_LOG,
    FILTER_METHOD,
    NORMALIZE_METHOD,
    FREQ_RANGE,
    CROP_LIVE_FREQ_RANGE,
)
from SoundMonitor.state_filter import StateFilter

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    timestamp: float
    state: bool
    d_on: float
    d_off: float

    def to_sql_list(self):
        return (
            self.timestamp,
            int(self.state),
            self.d_on,
            self.d_off,
        )


async def detection_loop(
    mic: AsyncMic,
    psd_on: list[np.ndarray],
    psd_off: list[np.ndarray],
    threshold: float,
    result_queue: asyncio.Queue,
    freq: np.ndarray | None = None,
    window_sec: float = ANALYSIS_WINDOW_SEC,
    interval_sec: float = DETECT_INTERVAL_SEC,
    plot: bool = False,
) -> None:
    last_smoothed: bool | int | None = None
    last_written: bool | None = None  # last state sent to DB
    min_samples = int(mic.sr * window_sec)
    last_written_time: float = 0
    # predictor = PSD_KNN(crop_live_freq_range=CROP_LIVE_FREQ_RANGE)
    predictor = PowerPredictor()
    state_filter = StateFilter(method=FILTER_METHOD)
    logger.info(f"  Starting learning psd_knn")
    freq_templates = freq
    # predictor.fit(on_templates=psd_on, freq=freq, freq_low=FREQ_RANGE[0], freq_high=FREQ_RANGE[1])
    logger.info(
        f"  Detection loop running @ {mic.sr} Hz with threshold: {threshold:.4f}, "
        f"{plot=}, Crop Live Freq: {str(CROP_LIVE_FREQ_RANGE)}. (Ctrl+C to stop)"
    )
    try:
        while True:
            audio = mic.get_recent(window_sec)
            if len(audio) < min_samples:
                await asyncio.sleep(0.2)
                logger.warning(f"Recording failed: audio device is short {len(audio)} < {min_samples=}")
                continue

            freq, psd_live, success, is_on, score = await asyncio.to_thread(
                predict_psd, audio=audio, predictor=predictor, sr=mic.sr
            )

            freq_list = [str(round(f, 1)) for f in freq if f]
            logger.debug(f"{success=}, Analyzed frequencies ({len(freq)}): {', '.join(freq_list)} Hz")
            if not success:
                logger.warning("Recording failed: audio device might be muted. Retrying after 10 seconds...")
                await asyncio.sleep(10)
                continue

            # print(f"{psd=}")
            # is_on, score = psd_knn.predict(psd, normalize=True)
            # logger.debug(f"{is_on=} {score=}")

            smoothed = bool(state_filter.update(is_on, score))

            # n = min(len(psd), len(psd_on[0]), len(psd_off[0]))
            # # fast, can stay in-loop
            # d_on = spectral_distance(psd[:n], psd_on[:n])
            # d_off = spectral_distance(psd[:n], psd_off[:n])
            # logger.debug(f"freq Hz: [{' '.join([str(f'{x:10.9f}'[:10]) for x in freq[2:8]])}] ({len(freq)})")
            logger.info(f"{is_on=} score={score:.4f} {smoothed=}")

            # logger.debug(f"psd    : {psd[2:8]}")
            # logger.debug(f"psd_on : {psd_on[2:8]}")
            # logger.debug(f"psd_off: {psd_off[2:8]}")

            # d_off_tr = d_off * threshold
            # is_on = d_on < d_off_tr
            # smoothed = bool(state_filter.update(is_on, min(d_on, d_off)))

            #
            # # Hysteresis: different decision boundary depending on current state
            # if last_smoothed:  # currently ON → make it harder to turn OFF
            #     d_off_tr = d_off * threshold * THRESHOLD_OFF
            #     is_on = d_on < d_off_tr
            # else:  # currently OFF (or first sample) → make it harder to turn ON
            #     d_off_tr = d_off * threshold * THRESHOLD_ON
            #     is_on = d_on < d_off_tr
            #
            # history.append(is_on)
            # history_ready = len(history) == SMOOTH_WINDOWS
            #
            # # Strong majority required to change state
            # votes_on = sum(history)
            # votes_min_confirmed = votes_on >= MIN_CONFIRM
            # if votes_min_confirmed:
            #     smoothed = True
            # elif votes_on <= (SMOOTH_WINDOWS - MIN_CONFIRM):
            #     smoothed = False
            # else:
            #     # stay in previous state (hysteresis zone)
            #     smoothed = last_smoothed if last_smoothed is not None else False

            # Logging
            state_str = "ON " if is_on else "off"
            marker = " <<<" if smoothed != last_smoothed and last_smoothed is not None else ""

            # logger.debug(
            #     # f"[{time.strftime('%H:%M:%S')}]  "
            #     f"d_on={d_on:.3f}  d_off={d_off:.3f}  d_off_tr={d_off_tr:.3f} → {state_str} {smoothed} {marker}",
            # )

            last_smoothed = smoothed
            periodic_write = (time.monotonic() - last_written_time) > PERIODIC_WRITE_TIME

            # --- DB only on trusted state change ---
            if state_filter.is_history_ready() and (last_written is None or smoothed != last_written or periodic_write):
                result = DetectionResult(
                    timestamp=time.time(),
                    state=smoothed,  # trusted state
                    d_on=score,
                    d_off=score,
                )
                # await result_queue.put(result.to_sql_list())
                try:
                    result_queue.put_nowait(result.to_sql_list())
                except asyncio.QueueFull:
                    logger.warning("Queue full, skipping")

                logger.debug(f"Added to result_queue: {result_queue.qsize()}")
                last_written_time = time.monotonic()
                last_written = smoothed
                if plot:
                    idx = 0
                    # if freq_templates is not None:
                    #     mask = (freq_templates >= FREQ_RANGE[0]) & (freq_templates <= FREQ_RANGE[1])
                    #     psd_on_t = psd_on[idx][mask]
                    #     psd_off_t = psd_off[idx][mask]
                    # else:
                    #     psd_on_t = psd_on[idx]
                    #     psd_off_t = psd_off[idx]
                    # templates = {"on": psd_on_t, "off": psd_off_t}
                    # templates = {"on": psd_on_t}
                    templates = {}
                    asyncio.create_task(
                        asyncio.to_thread(
                            plot_psd_comparison,
                            freqs=freq,
                            templates=templates,
                            psd_live=psd_live,
                            save_path=DATA_PATH / PLOT_LIVE_FILENAME,
                        )
                    )

            await asyncio.sleep(interval_sec)
    except asyncio.CancelledError:
        ...
