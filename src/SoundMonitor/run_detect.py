import asyncio
import logging

from SoundMonitor.async_mic import AsyncMic
from SoundMonitor.audio_device import list_input_devices
from SoundMonitor.db_writer_worker import db_writer_worker, db_cleanup_worker, init_db
from SoundMonitor.detector import detection_loop
from SoundMonitor.handler_signal import setup_signal_handlers
from SoundMonitor.settings import ANALYSIS_WINDOW_SEC, TEMPLATE_DIR, DB_FILE, PREFERRED_RATES
from SoundMonitor.templates import load_templates, load_threshold

logger = logging.getLogger(__name__)


async def run_detect(
    device_index: int | None = None,
    threshold_override: float | None = None,
    window_sec: float = ANALYSIS_WINDOW_SEC,
    plot: bool = False,
) -> None:
    psd_on, psd_off, freq, tmpl_sr = load_templates()
    thr = load_threshold(override=threshold_override)

    logger.info("=" * 60)
    logger.info("LIVE ASYNC DETECTION")
    logger.info(f"  Template dir : {TEMPLATE_DIR}")
    logger.info(f"  Template SR  : {tmpl_sr} Hz")
    logger.info(f"  Threshold    : {thr:.3f}")
    logger.info(f"  Window       : {window_sec:.1f}s")
    logger.info(f"  SQLite DB    : {DB_FILE}")
    logger.info(f"  Will try rates: {PREFERRED_RATES}")
    logger.info("  Ctrl+C to stop")
    logger.info("=" * 60)
    list_input_devices()
    logger.info("")

    shutdown_event = asyncio.Event()
    setup_signal_handlers(shutdown_event, asyncio.get_running_loop())

    result_queue: asyncio.Queue = asyncio.Queue(maxsize=256)
    await init_db(db_path=DB_FILE)

    writer_task = asyncio.create_task(
        db_writer_worker(queue=result_queue, shutdown_event=shutdown_event, db_path=DB_FILE)
    )
    cleanup_task = asyncio.create_task(db_cleanup_worker(shutdown_event=shutdown_event, db_path=DB_FILE))
    await asyncio.sleep(0.5)

    mic = AsyncMic(device_index=device_index)
    live_sr = mic.start()

    if live_sr != tmpl_sr:
        logger.info(
            f"  ⚠  Live SR ({live_sr}) ≠ template SR ({tmpl_sr}). "
            "PSD shapes should still match (normalized), but for best "
            "results re-train at the same rate."
        )

    detect_task = asyncio.create_task(
        detection_loop(
            mic=mic,
            psd_on=psd_on,
            psd_off=psd_off,
            freq=freq,
            threshold=thr,
            result_queue=result_queue,
            window_sec=window_sec,
            plot=plot,
        )
    )

    await shutdown_event.wait()

    logger.info("Shutting down…")
    detect_task.cancel()
    try:
        await detect_task
    except asyncio.CancelledError:
        ...

    mic.stop()
    await result_queue.put(None)
    await writer_task
    await cleanup_task
    logger.info("Done.")
