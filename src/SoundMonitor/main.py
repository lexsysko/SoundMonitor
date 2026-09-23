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
    p_train.add_argument(
        "-c", "--count", type=int, default=None, help="by default used preconfigured values from settings"
    )
    p_train.add_argument(
        "-s", "--sleep", type=int, default=None, help="by default used preconfigured values from settings"
    )
    p_train.add_argument("-m", "--mode", choices=("auto", "on", "off"), default="auto")
    p_train.add_argument("--prune", action="store_true", help="Clearing all previous template files")
    p_train.add_argument("--nobeep", action="store_true", help="Disable beep sound before start record")

    sub.add_parser("calibrate", help="Re-compute threshold from existing templates")
    p_plot = sub.add_parser("plot", help="plot diagram from existing PSD on templates")
    p_plot.add_argument("-f", "--filename", type=str, default="diagram{}.png")
    p_plot.add_argument("-i", "--idx", type=int, default=0, help="idx of templates to plot")

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
                train(
                    device_index=args.device,
                    duration=args.duration,
                    count=args.count,
                    delay=args.sleep,
                    mode=args.mode,
                    prune=args.prune,
                    beep=not args.nobeep,
                )
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
            idx = args.idx
            psd_on, psd_off, freqs, tmpl_sr = load_templates()
            save_path = DATA_PATH / args.filename.format(f"_{idx:03d}")
            # templates = {"on": psd_on[min(idx, len(psd_on) - 1)], "off": psd_off[min(idx, len(psd_off) - 1)]}
            templates = {"on": psd_on[min(idx, len(psd_on) - 1)]}

            plot_psd_comparison(freqs=freqs, templates=templates, save_path=save_path)


if __name__ == "__main__":
    main()
