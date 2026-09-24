import asyncio
import logging
import numpy as np
import time
from dataclasses import dataclass

from SoundMonitor.analizer import predict_psd
from SoundMonitor.async_mic import AsyncMic
from SoundMonitor.plot_psd import plot_psd_comparison
from SoundMonitor.power_predictor import PowerPredictor
from SoundMonitor.settings import (
    ANALYSIS_WINDOW_SEC,
    DETECT_INTERVAL_SEC,
    PERIODIC_WRITE_TIME,
    DATA_PATH,
    PLOT_LIVE_FILENAME,
    FILTER_METHOD,
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
    result_queue: asyncio.Queue,
    psd_on: list[np.ndarray] | None = None,
    psd_off: list[np.ndarray] | None = None,
    threshold: float | None = None,
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
    # logger.info(f"  Starting learning psd_knn")
    # freq_templates = freq
    # predictor.fit(on_templates=psd_on, freq=freq, freq_low=FREQ_RANGE[0], freq_high=FREQ_RANGE[1])
    logger.info(f"  Detection loop running @ {mic.sr} Hz , {plot=}. (Ctrl+C to stop)")
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

            smoothed = bool(state_filter.update(is_on, score))

            # Logging
            state_str = "ON " if is_on else "OFF"
            marker = " <<<" if smoothed != last_smoothed and last_smoothed is not None else ""

            trusted_state_str = "ON " if smoothed else "OFF"
            logger.info(f"STATE: {state_str}, SCORE: {score:.4f}, TRUSTED STATE: {trusted_state_str}{marker}")

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
