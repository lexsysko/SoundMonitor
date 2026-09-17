#!/usr/bin/env python3
"""
Refrigerator Compressor ON/OFF Detector (async)
===============================================
- Training: record ON/OFF → .npz templates + auto threshold calibration
- Live detection: non-blocking PyAudio callback + asyncio
- Results → asyncio.Queue → background worker → aiosqlite
- Auto sample-rate fallback: tries 16000, then 44100, then 48000

Install:
    sudo apt update
    sudo apt install -y portaudio19-dev python3-pyaudio
    pip install numpy scipy aiosqlite

Usage:
    python main.py train
    python main.py calibrate
    python main.py detect
    python main.py devices
"""

from pathlib import Path

import argparse
import asyncio
import logging
import sys

from SoundMonitor.audio_device import list_input_devices
from SoundMonitor.calibrate import calibrate_from_templates
from SoundMonitor.plot_psd import plot_psd_comparison
from SoundMonitor.run_detect import run_detect
from SoundMonitor.settings import (
    ANALYSIS_WINDOW_SEC,
    DATA_PATH,
    APP_VERSION,
)
from SoundMonitor.templates import load_templates
from SoundMonitor.train import train


def setup_logger(level: str | int = logging.INFO) -> logging.Logger:
    """Configure and return the application logger."""
    if isinstance(level, str):
        try:
            resolved_level = int(level)
        except ValueError:
            resolved_level = getattr(logging, level.upper(), logging.INFO)
    elif isinstance(level, int):
        resolved_level = level
    else:
        resolved_level = logging.INFO

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    log = logging.getLogger(Path(__file__).parent.name)
    log.setLevel(resolved_level)
    return log


logger = setup_logger()

try:
    import pyaudio
except ImportError:
    logger.info("ERROR: pyaudio is not installed.\n  sudo apt install -y portaudio19-dev\n  uv add pyaudio\n")
    sys.exit(1)

try:
    import aiosqlite
except ImportError:
    logger.error("ERROR: aiosqlite is not installed.\n  uv add aiosqlite\n")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Open stream with rate fallback
# ---------------------------------------------------------------------------


async def init_db(db_path: Path) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   REAL    NOT NULL,
                state       INTEGER NOT NULL,
                d_on        REAL    NOT NULL,
                d_off       REAL    NOT NULL
            )
            """
        )
        await db.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)")
        await db.execute("PRAGMA journal_mode=WAL;")
        await db.execute("PRAGMA busy_timeout=5000;")
        await db.execute("PRAGMA synchronous=NORMAL;")
        await db.commit()


async def db_writer_worker(
    queue: asyncio.Queue,
    db_path: Path,
) -> None:
    await init_db(db_path)
    logger.info(f"  SQLite writer ready → {db_path}")

    async with aiosqlite.connect(db_path) as db:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    queue.task_done()
                    break

                await db.execute(
                    """
                    INSERT INTO events (timestamp, state, d_on, d_off)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        item.timestamp,
                        int(item.state),
                        item.d_on,
                        item.d_off,
                    ),
                )
                await db.commit()
                logger.debug(f"[db worker] saved data: {item}")
            except Exception as e:
                logger.error(f"[db worker] error: {e}")
            finally:
                try:
                    queue.task_done()
                except ValueError:
                    ...

    logger.info("  SQLite writer stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Async refrigerator compressor ON/OFF detector "
        "(PyAudio callback + asyncio queue + aiosqlite, auto SR fallback)"
    )
    parser.add_argument("--version", action="version", version=f"App version: {APP_VERSION}")
    parser.add_argument("--loglevel", type=str, default="INFO")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Record ON/OFF templates + auto-calibrate")
    p_train.add_argument("-d", "--device", type=int, default=None)
    p_train.add_argument("-t", "--duration", type=float, default=8.0)

    sub.add_parser("calibrate", help="Re-compute threshold from existing templates")
    p_plot = sub.add_parser("plot", help="plot diagram from existing PSD on templates")
    p_plot.add_argument("-f", "--filename", type=str, default="diagram.png")

    p_det = sub.add_parser("detect", help="Live async detection → SQLite")
    p_det.add_argument("-d", "--device", type=int, default=None)
    p_det.add_argument("--threshold", type=float, default=None)
    p_det.add_argument("--window", type=float, default=ANALYSIS_WINDOW_SEC)
    p_det.add_argument("--plot", action="store_true")

    sub.add_parser("devices", help="List microphone devices")

    args = parser.parse_args()
    setup_logger(args.loglevel)

    match args.cmd:
        case "devices":
            list_input_devices()
        case "train":
            try:
                train(device_index=args.device, duration=args.duration)
            except KeyboardInterrupt:
                print()
                logger.error("Interrupted.")
        case "calibrate":
            try:
                calibrate_from_templates(verbose=True)
            except KeyboardInterrupt:
                logger.error("Interrupted.")
        case "detect":
            try:
                asyncio.run(
                    run_detect(
                        device_index=args.device,
                        threshold_override=args.threshold,
                        window_sec=args.window,
                        plot=args.plot,
                    )
                )
            except KeyboardInterrupt:
                logger.error("Interrupted.")
        case "plot":
            psd_on, psd_off, freqs, tmpl_sr = load_templates()
            save_path = DATA_PATH / args.filename
            plot_psd_comparison(freqs=freqs, psd_on=psd_on, psd_off=psd_off, save_path=save_path)


if __name__ == "__main__":
    main()
