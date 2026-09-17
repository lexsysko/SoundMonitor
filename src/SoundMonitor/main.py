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

from collections import deque
from pathlib import Path
from threading import Lock

import argparse
import asyncio
import functools
import json
import logging
import numpy as np
import sys
import time
from dataclasses import dataclass
from scipy.signal import butter, sosfilt, welch
from typing import Deque, Tuple

from __init__ import __version__


def time_it(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # High-resolution clock
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' took {execution_time:.6f} seconds")
        return result

    return wrapper


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
    logger.info(
        "ERROR: pyaudio is not installed.\n"
        "  sudo apt install -y portaudio19-dev\n"
        "  uv add pyaudio\n"
    )
    sys.exit(1)

try:
    import aiosqlite
except ImportError:
    logger.error("ERROR: aiosqlite is not installed.\n  uv add aiosqlite\n")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Preferred rates in order. First one that the device accepts is used.
# PREFERRED_RATES = [16000, 44100, 48000]
PREFERRED_RATES = [44100, 48000, 16000]

CHANNELS = 1
CHUNK = 1024

# Target Welch window length in seconds (adapted to actual SR)
WELCH_WINDOW_SEC = 0.256
FREQ_RANGE = (60.0, 1000.0)

DEFAULT_THRESHOLD = 0.60
SMOOTH_WINDOWS = 5
MIN_RECORD_SEC = 3.0
SAFETY_MARGIN = 0.18

ANALYSIS_WINDOW_SEC = 1.5
DETECT_INTERVAL_SEC = 0.5
BUFFER_SEC = 4.0

BASE_PATH = Path(__file__).parent.parent.parent
DATA_DIR = BASE_PATH / "data"
DB_FILE = DATA_DIR / "snd_data.db"
if not DB_FILE.parent.exists():
    # Safe for both CLI package execution and local development
    DB_FILE = Path.cwd() / "data/snd_data.db"

DB_FILE.parent.mkdir(exist_ok=True, parents=True)
APP_VERSION = __version__

TEMPLATE_DIR = DATA_DIR / "templates"
ON_FILE = TEMPLATE_DIR / "compressor_on.npz"
OFF_FILE = TEMPLATE_DIR / "compressor_off.npz"
THRESHOLD_FILE = TEMPLATE_DIR / "threshold.json"

# Runtime sample rate (set after successful open)
EFFECTIVE_SR: int = PREFERRED_RATES[0]


def nperseg_for(sr: int) -> int:
    """Welch window ≈ WELCH_WINDOW_SEC, rounded to the nearest power of 2."""
    target = int(sr * WELCH_WINDOW_SEC)
    # next power of 2 (or nearest – pick what you prefer)
    n = 1 << (target - 1).bit_length()  # ceil to 2^N
    # optional: also allow previous power of 2 if closer
    prev = n >> 1
    if prev >= 256 and abs(target - prev) < abs(target - n):
        n = prev
    return max(256, min(n, 16384))


@dataclass
class DetectionResult:
    timestamp: float
    state: bool
    d_on: float
    d_off: float


# ---------------------------------------------------------------------------
# Open stream with rate fallback
# ---------------------------------------------------------------------------
def open_input_stream(
    pa: pyaudio.PyAudio,
    device_index: int | None = None,
    callback=None,
) -> Tuple[pyaudio.Stream, int]:
    """
    Try PREFERRED_RATES in order until the device accepts one.
    Returns (stream, effective_sr).
    """
    last_err: Exception | None = None
    for rate in PREFERRED_RATES:
        kwargs = dict(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=rate,
            input=True,
            frames_per_buffer=CHUNK,
        )
        if device_index is not None:
            kwargs["input_device_index"] = device_index
        if callback is not None:
            kwargs["stream_callback"] = callback

        try:
            stream = pa.open(**kwargs)
            logger.info(f"  Audio opened at {rate} Hz")
            return stream, rate
        except Exception as e:
            last_err = e
            logger.error(f"  Rate {rate} Hz not accepted: {e}")

    raise RuntimeError(
        f"Could not open input stream at any of {PREFERRED_RATES}. "
        f"Last error: {last_err}"
    )


def list_input_devices() -> None:
    pa = pyaudio.PyAudio()
    buff = "\nAvailable input devices:"
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            buff += (
                f"\n  [{i}] {info['name']}  "
                f"(max in={info['maxInputChannels']}, "
                f"default rate={int(info['defaultSampleRate'])})"
            )
    pa.terminate()
    logger.info(buff + "\n")


def record_seconds(
    seconds: float,
    device_index: int | None = None,
) -> Tuple[np.ndarray, int]:
    """
    Blocking record for training.
    Returns (audio_float32, effective_sr).
    """
    global EFFECTIVE_SR
    pa = pyaudio.PyAudio()
    stream, sr = open_input_stream(pa, device_index=device_index)
    EFFECTIVE_SR = sr

    frames = []
    n_chunks = max(1, int(sr / CHUNK * seconds))
    logger.info(f"  Recording {seconds:.1f}s @ {sr} Hz …")
    for _ in range(n_chunks):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(np.frombuffer(data, dtype=np.int16))
    stream.stop_stream()
    stream.close()
    pa.terminate()
    logger.info("  Recording done")

    audio = np.concatenate(frames).astype(np.float32)
    audio /= 32768.0
    return audio, sr


# @time_it
def compute_psd(
    audio: np.ndarray,
    sr: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if sr is None:
        sr = EFFECTIVE_SR
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    sos = butter(2, 25, btype="high", fs=sr, output="sos")
    audio = sosfilt(sos, audio.astype(np.float64))

    nperseg = min(nperseg_for(sr), max(256, len(audio) // 4))
    freqs, psd = welch(audio, fs=sr, nperseg=nperseg, scaling="density")
    mask = (freqs >= FREQ_RANGE[0]) & (freqs <= FREQ_RANGE[1])
    freqs = freqs[mask]
    psd = psd[mask]
    psd = psd / (psd.max() + 1e-12)
    return freqs, psd.astype(np.float32)


# @time_it
def spectral_distance(psd_a: np.ndarray, psd_b: np.ndarray) -> float:
    a = psd_a.astype(np.float64)
    b = psd_b.astype(np.float64)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    cos = np.dot(a, b) / (na * nb)
    return float(1.0 - cos)


def calibrate_from_templates(verbose: bool = True) -> dict:
    if not ON_FILE.exists() or not OFF_FILE.exists():
        raise FileNotFoundError(
            f"Need both {ON_FILE.name} and {OFF_FILE.name}. Run 'train' first."
        )

    on = np.load(ON_FILE)
    off = np.load(OFF_FILE)
    d_on_off = spectral_distance(on["psd"], off["psd"])
    threshold = float(np.clip(1.0 * (1.0 - SAFETY_MARGIN), 0.35, 0.90))

    result = {
        "threshold": threshold,
        "distance_on_off": round(d_on_off, 4),
        "safety_margin": SAFETY_MARGIN,
        "method": "midpoint_with_safety_margin",
        "rule": "is_on = (d_on < d_off * threshold)",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "template_sr": int(on["sr"]) if "sr" in on else None,
    }

    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(THRESHOLD_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if verbose:
        logger.info("=" * 60)
        logger.info("AUTOMATIC THRESHOLD CALIBRATION")
        logger.info("=" * 60)
        logger.info(f"  Spectral distance ON ↔ OFF : {d_on_off:.4f}")
        logger.info(f"  Safety margin              : {SAFETY_MARGIN:.2f}")
        logger.info(f"  Calibrated threshold       : {threshold:.3f}")
        if result["template_sr"]:
            logger.info(f"  Template sample rate       : {result['template_sr']} Hz")
        logger.info(f"  Saved to                   : {THRESHOLD_FILE}")
        if d_on_off < 0.12:
            logger.warning("  ⚠  Templates very close – detection may be unreliable.")
        elif d_on_off < 0.20:
            logger.info("  ⚠  Moderate separation.")
        else:
            logger.info("  ✓  Good separation between ON and OFF templates.")
        logger.info("=" * 60)

    return result


def load_threshold(override: float | None = None) -> float:
    if override is not None:
        return override
    if THRESHOLD_FILE.exists():
        try:
            with open(THRESHOLD_FILE, encoding="utf-8") as f:
                data = json.load(f)
            t = float(data["threshold"])
            logger.info(f"  Using calibrated threshold: {t:.3f}")
            return t
        except Exception as e:
            logger.info(f"  Warning: could not read {THRESHOLD_FILE}: {e}")
    logger.info(f"  Using default threshold: {DEFAULT_THRESHOLD:.3f}")
    return DEFAULT_THRESHOLD


def load_templates() -> tuple[np.ndarray, np.ndarray, int]:
    if not ON_FILE.exists() or not OFF_FILE.exists():
        logger.error(f"Templates not found in {TEMPLATE_DIR}")
        logger.error("Run:  python main.py train")
        sys.exit(1)
    on = np.load(ON_FILE)
    off = np.load(OFF_FILE)
    tmpl_sr = int(on["sr"]) if "sr" in on else 16000
    return on["psd"], off["psd"], tmpl_sr


def train(device_index: int | None = None, duration: float = 8.0) -> None:
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    list_input_devices()

    def capture_label(label: str, path: Path) -> None:
        input(f"\n>>> Prepare '{label}' state, then press Enter to start recording…")
        audio, sr = record_seconds(duration, device_index)
        if len(audio) < sr * MIN_RECORD_SEC:
            logger.info("Recording too short – try again.")
            return
        freqs, psd = compute_psd(audio, sr=sr)
        np.savez_compressed(
            path,
            freqs=freqs,
            psd=psd,
            sr=sr,
            duration=duration,
            label=label,
            created=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
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
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)"
        )
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


class AsyncMic:
    """Non-blocking mic via PyAudio callback; auto rate fallback."""

    def __init__(self, device_index: int | None = None):
        self.device_index = device_index
        self.sr: int = PREFERRED_RATES[0]
        self.buffer: Deque[float] = deque()
        self._pa = None
        self._stream = None
        self._lock = Lock()

    def _callback(self, in_data, frame_count, time_info, status) -> Tuple[None, int]:
        audio = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        with self._lock:
            self.buffer.extend(audio.tolist())
        return None, pyaudio.paContinue

    def start(self) -> int:
        global EFFECTIVE_SR
        self._pa = pyaudio.PyAudio()
        self._stream, self.sr = open_input_stream(
            self._pa,
            device_index=self.device_index,
            callback=self._callback,
        )
        EFFECTIVE_SR = self.sr
        # size ring buffer for BUFFER_SEC at the real rate
        with self._lock:
            self.buffer = deque(maxlen=int(self.sr * BUFFER_SEC))
        self._stream.start_stream()
        logger.info(f"  Microphone stream started (callback, {self.sr} Hz)")
        return self.sr

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None
        logger.info("  Microphone stream stopped")

    def get_recent(self, seconds: float) -> np.ndarray:
        n = int(seconds * self.sr)
        with self._lock:
            data = list(self.buffer)
        if len(data) < n:
            return np.array(data, dtype=np.float32)
        return np.array(data[-n:], dtype=np.float32)


async def detection_loop(
    mic: AsyncMic,
    psd_on: np.ndarray,
    psd_off: np.ndarray,
    threshold: float,
    result_queue: asyncio.Queue,
    window_sec: float = ANALYSIS_WINDOW_SEC,
    interval_sec: float = DETECT_INTERVAL_SEC,
) -> None:
    history: list[bool] = []
    last_smoothed: bool | None = None
    last_written: bool | None = None  # last state sent to DB
    min_samples = int(mic.sr * 0.8)

    logger.info(f"  Detection loop running @ {mic.sr} Hz (Ctrl+C to stop)")
    try:
        while True:
            audio = mic.get_recent(window_sec)
            if len(audio) < min_samples:
                await asyncio.sleep(0.2)
                continue

            _, psd = await asyncio.to_thread(compute_psd, audio, sr=mic.sr)
            n = min(len(psd), len(psd_on), len(psd_off))
            # fast, can stay in-loop
            d_on = spectral_distance(psd[:n], psd_on[:n])
            d_off = spectral_distance(psd[:n], psd_off[:n])

            is_on = d_on < (d_off * threshold)

            history.append(is_on)
            if len(history) > SMOOTH_WINDOWS:
                history.pop(0)
            smoothed = sum(history) > len(history) / 2

            state_str = "ON " if smoothed else "off"
            marker = (
                " <<<"
                if smoothed != last_smoothed and last_smoothed is not None
                else ""
            )

            logger.debug(
                # f"[{time.strftime('%H:%M:%S')}]  "
                f"d_on={d_on:.3f}  d_off={d_off:.3f}  → {state_str} {smoothed} {marker}",
            )

            last_smoothed = smoothed

            # --- DB only on trusted state change ---
            if last_written is None or smoothed != last_written:
                result = DetectionResult(
                    timestamp=time.time(),
                    state=smoothed,  # trusted state
                    d_on=d_on,
                    d_off=d_off,
                )
                await result_queue.put(result)
                last_written = smoothed

            await asyncio.sleep(interval_sec)
    except asyncio.CancelledError:
        ...


async def run_detect(
    device_index: int | None = None,
    threshold_override: float | None = None,
    window_sec: float = ANALYSIS_WINDOW_SEC,
) -> None:
    psd_on, psd_off, tmpl_sr = load_templates()
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

    result_queue: asyncio.Queue = asyncio.Queue(maxsize=256)
    writer_task = asyncio.create_task(db_writer_worker(result_queue, DB_FILE))

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
            threshold=thr,
            result_queue=result_queue,
            window_sec=window_sec,
        )
    )

    try:
        await detect_task
    except (asyncio.CancelledError, KeyboardInterrupt):
        ...
    finally:
        logger.info("Shutting down…")
        detect_task.cancel()
        try:
            await detect_task
        except asyncio.CancelledError:
            ...

        mic.stop()
        await result_queue.put(None)
        await writer_task
        logger.info("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Async refrigerator compressor ON/OFF detector "
        "(PyAudio callback + asyncio queue + aiosqlite, auto SR fallback)"
    )
    parser.add_argument("--loglevel", type=str, default="INFO")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Record ON/OFF templates + auto-calibrate")
    p_train.add_argument("-d", "--device", type=int, default=None)
    p_train.add_argument("-t", "--duration", type=float, default=8.0)

    sub.add_parser("calibrate", help="Re-compute threshold from existing templates")

    p_det = sub.add_parser("detect", help="Live async detection → SQLite")
    p_det.add_argument("-d", "--device", type=int, default=None)
    p_det.add_argument("--threshold", type=float, default=None)
    p_det.add_argument("--window", type=float, default=ANALYSIS_WINDOW_SEC)

    sub.add_parser("devices", help="List microphone devices")

    args = parser.parse_args()
    setup_logger(args.loglevel)

    if args.cmd == "devices":
        list_input_devices()
    elif args.cmd == "train":
        try:
            train(device_index=args.device, duration=args.duration)
        except KeyboardInterrupt:
            print()
            logger.error("Interrupted.")
    elif args.cmd == "calibrate":
        try:
            calibrate_from_templates(verbose=True)
        except KeyboardInterrupt:
            logger.error("Interrupted.")
    elif args.cmd == "detect":
        try:
            asyncio.run(
                run_detect(
                    device_index=args.device,
                    threshold_override=args.threshold,
                    window_sec=args.window,
                )
            )
        except KeyboardInterrupt:
            logger.error("Interrupted.")


if __name__ == "__main__":
    main()
