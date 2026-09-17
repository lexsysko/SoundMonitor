from os import environ
from pathlib import Path

import asyncio
import time
from dotenv import load_dotenv

from SoundMonitor import __version__

BASE_PATH = Path(__file__).parent.parent.parent

for folder in (Path.cwd(), BASE_PATH):
    if (folder / ".env").exists():
        load_dotenv()
        break


data_path_str: str | None = environ.get("DATA_PATH")

if data_path_str:
    DATA_PATH = Path(data_path_str).expanduser().resolve()
else:
    DATA_PATH = BASE_PATH / "data"
    if not DATA_PATH.exists():
        # Safe for both CLI package execution and local development
        DATA_PATH = Path.cwd() / "data"

DATA_PATH.parent.mkdir(exist_ok=True, parents=True)
DB_FILE = DATA_PATH / environ.get("DB_FILE", "snd_data.db")

APP_VERSION = __version__
LOGLEVEL: str = environ.get("LOGLEVEL", "INFO").upper()
PLOT_FILENAME: str = environ.get("PLOT_FILENAME", "diagram.png")
PLOT_LIVE_FILENAME: str = environ.get("PLOT_LIVE_FILENAME", "live.png")
CLEANUP_TIMEOUT: int = int(environ.get("CLEANUP_TIMEOUT", 60 * 60 * 24))
CLEANUP_PERIOD_DAYS: int = int(environ.get("CLEANUP_PERIOD_DAYS", 30))
BATCH_FLUSH_DB_TIMEOUT: int = int(environ.get("BATCH_FLUSH_DB_TIMEOUT", 3))


PREFERRED_RATES = [44100, 48000, 16000]
CHANNELS = 1
CHUNK = 1024
WELCH_WINDOW_SEC = 0.256
FREQ_RANGE = (80.0, 500.0)
DEFAULT_THRESHOLD = 0.60
SMOOTH_WINDOWS = 30
MIN_CONFIRM = int(SMOOTH_WINDOWS * 0.85)  # how many votes needed to change state (out of 30)
THRESHOLD_ON = 1.05  # harder to turn ON
THRESHOLD_OFF = 0.85  # harder to turn OFF
MIN_RECORD_SEC = 3.0
SAFETY_MARGIN = 0.18
ANALYSIS_WINDOW_SEC = 3.0
DETECT_INTERVAL_SEC = 1.2
BUFFER_SEC = 6.0
PERIODIC_WRITE_TIME = 120

TRAIN_DURATION = 8.0

TEMPLATE_DIR = DATA_PATH / "templates"
TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)

ON_FILE = TEMPLATE_DIR / "compressor_on.npz"
OFF_FILE = TEMPLATE_DIR / "compressor_off.npz"
THRESHOLD_FILE = TEMPLATE_DIR / "threshold.json"

_EFFECTIVE_SR: int = PREFERRED_RATES[0]


_shutdown_event = None
_db_queue = None
_sr: int

last_frame_counter = {}
last_packet_time = time.time()
last_counter_data = {"last_packet_time": time.time()}


def get_shutdown_event() -> asyncio.Event:
    global _shutdown_event
    if _shutdown_event is None:
        _shutdown_event = asyncio.Event()
    return _shutdown_event


def get_db_queue() -> asyncio.Queue:
    global _db_queue
    if _db_queue is None:
        _db_queue = asyncio.Queue()
    return _db_queue


def get_effective_sr():
    return _EFFECTIVE_SR


def set_effective_sr(value: int):
    global _EFFECTIVE_SR
    _EFFECTIVE_SR = value
