import asyncio
import logging
import numpy as np
import time
from dataclasses import dataclass

from SoundMonitor.analize.analizer import predict_psd
from SoundMonitor.async_mic import AsyncMic
from SoundMonitor.filters.confirmed_state_filter import ConfirmedStateFilter, StateCommit
from SoundMonitor.filters.state_filter import StateFilter
from SoundMonitor.plot_psd import plot_psd_comparison
from SoundMonitor.predictors.power_predictor import PowerPredictor
from SoundMonitor.services.audio_device import play_beep
from SoundMonitor.settings import (
    ANALYSIS_WINDOW_SEC,
    DETECT_INTERVAL_SEC,
    PERIODIC_WRITE_TIME,
    DATA_PATH,
    PLOT_LIVE_FILENAME,
    FILTER_METHOD,
    FILTER_CONFIRMED_STATE_CONFIRM_DURATION,
    FILTER_CONFIRMED_STATE_MIN_INTERVAL,
)

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    timestamp: float
    state: bool | int
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
    beep: bool = False,
) -> None:
    last_smoothed_psd: dict[str, np.ndarray] = {}
    last_smoothed: bool | int | None = None
    last_written: bool | int | None = None  # last state sent to DB
    min_samples = int(mic.sr * window_sec)
    last_written_time: float = 0
    # predictor = PSD_KNN(crop_live_freq_range=CROP_LIVE_FREQ_RANGE)
    predictor = PowerPredictor()
    state_filter = StateFilter(method=FILTER_METHOD)
    # logger.info(f"  Starting learning psd_knn")
    # freq_templates = freq
    # predictor.fit(on_templates=psd_on, freq=freq, freq_low=FREQ_RANGE[0], freq_high=FREQ_RANGE[1])
    logger.info(f"  Detection loop running @ {mic.sr} Hz , {plot=}. (Ctrl+C to stop)")
    confirmed_state_filter = ConfirmedStateFilter(
        confirm_duration=FILTER_CONFIRMED_STATE_CONFIRM_DURATION,
        min_interval=FILTER_CONFIRMED_STATE_MIN_INTERVAL,
        initial_state=0,
    )
    baseline: StateCommit = confirmed_state_filter.initial_commit()
    smoothed: int = baseline.state
    last_smoothed_raw: int = smoothed
    detection_result: DetectionResult | None = None
    logger.debug(f"confirmed_state_filter t=:00  -> WRITE ({baseline.kind}) state={baseline.state}")
    try:
        while True:
            audio = mic.get_recent(window_sec)
            if len(audio) < min_samples:
                await asyncio.sleep(0.2)
                logger.warning(f"Recording failed: audio device is short {len(audio)} < {min_samples=}")
                continue

            freq, psd_live, success, is_on, score = await asyncio.to_thread(
                predict_psd, audio=audio, predictor=predictor, sr=mic.sr, smoothed=smoothed
            )

            freq_list = [str(round(f, 1)) for f in freq if f]
            logger.debug(f"{success=}, Analyzed frequencies ({len(freq)}): {', '.join(freq_list)} Hz")
            if not success:
                logger.warning("Recording failed: audio device might be muted. Retrying after 10 seconds...")
                await asyncio.sleep(10)
                continue

            smoothed_raw = state_filter.update(is_on, score)
            commit_state: StateCommit | None = confirmed_state_filter.update(smoothed_raw)
            # result commit_state=StateCommit(state=1, event_at=1790370382.512521, committed_at=1790370442.7113695, kind='change')

            # logger.debug(f"{commit_state=}")

            if commit_state is not None:
                is_changed = commit_state.kind == "change"
                if is_changed:
                    smoothed = commit_state.state
                    detection_result = DetectionResult(
                        timestamp=commit_state.event_at,
                        state=smoothed,  # trusted state
                        d_on=score,
                        d_off=score,
                    )
            else:
                is_changed = False

            # Logging
            state_str = "ON " if is_on else "OFF"
            # is_changed = smoothed != last_smoothed and last_smoothed is not None
            marker = " <<<" if is_changed else ""

            if is_changed:
                sm_state_str = "on" if smoothed else "off"
                if sm_state_str in last_smoothed_psd:
                    # average value of all previous psd stated
                    last_smoothed_psd[sm_state_str] = np.mean(
                        np.vstack([psd_live, last_smoothed_psd[sm_state_str]]), axis=0
                    )
                else:
                    last_smoothed_psd[sm_state_str] = psd_live

            trusted_state_str = "ON " if smoothed else "OFF"
            logger.info(
                f"STATE: {state_str}, SCORE: {score:.4f}, {smoothed_raw=}, TRUSTED STATE: {trusted_state_str}{marker}"
            )

            periodic_write = (time.monotonic() - last_written_time) > PERIODIC_WRITE_TIME

            if state_filter.is_history_ready() and periodic_write:
                hb = confirmed_state_filter.heartbeat(min_gap=60)
                if hb is not None:
                    detection_result = DetectionResult(
                        timestamp=hb.event_at,
                        state=smoothed,  # trusted state
                        d_on=score,
                        d_off=score,
                    )

                logger.debug(
                    f"confirmed_state_filter  (heartbeat tick) -> {'WRITE ' + hb.kind if hb else '(suppressed)'}  {hb=}"
                )

            # --- DB only on trusted state change ---
            if detection_result is not None:
                try:
                    result_queue.put_nowait(detection_result.to_sql_list())
                    detection_result = None
                except asyncio.QueueFull:
                    logger.warning("Queue full, skipping")

                logger.debug(f"Added to result_queue: {result_queue.qsize()}")
                last_written_time = time.monotonic()
                last_written = smoothed
                if plot:
                    asyncio.create_task(
                        asyncio.to_thread(
                            plot_psd_comparison,
                            freqs=freq,
                            templates=last_smoothed_psd,
                            psd_live=psd_live,
                            save_path=DATA_PATH / PLOT_LIVE_FILENAME,
                        )
                    )

            if beep and smoothed_raw != last_smoothed_raw:
                last_smoothed_raw = smoothed_raw
                asyncio.create_task(asyncio.to_thread(play_beep, pa=mic.pa, device_index=mic.device_index))

            await asyncio.sleep(interval_sec)
    except asyncio.CancelledError:
        ...
