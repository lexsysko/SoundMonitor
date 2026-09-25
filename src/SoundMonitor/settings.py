from os import environ
from pathlib import Path

from dotenv import load_dotenv

from SoundMonitor import __version__
from SoundMonitor.enums import AnalyzeMethods, NormalizeMethod

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
DB_FILE = DATA_PATH / Path(environ.get("DB_FILE", "snd_data.db")).name

APP_VERSION = __version__
LOGLEVEL: str = environ.get("LOGLEVEL", "INFO").upper()
PLOT_FILENAME: str = environ.get("PLOT_FILENAME", "diagram.png")
PLOT_LIVE_FILENAME: str = environ.get("PLOT_LIVE_FILENAME", "live.png")
CLEANUP_TIMEOUT: int = int(environ.get("CLEANUP_TIMEOUT", 60 * 60 * 24))
CLEANUP_PERIOD_DAYS: int = int(environ.get("CLEANUP_PERIOD_DAYS", 30))
BATCH_FLUSH_DB_TIMEOUT: int = int(environ.get("BATCH_FLUSH_DB_TIMEOUT", 2))


PREFERRED_RATES = [44100, 16000]
# PREFERRED_RATES = [16000, 44100]
CHANNELS = 1
CHUNK = 1024
WELCH_WINDOW_SEC = 0.1857596371882086 * 2

FREQ_RANGE = (float(environ.get("FREQ_RANGE_LOW", 80)), float(environ.get("FREQ_RANGE_HIGH", 108)))

DEFAULT_THRESHOLD = 0.60
SMOOTH_WINDOWS = int(environ.get("SMOOTH_WINDOWS", 30))
MIN_CONFIRM = int(SMOOTH_WINDOWS * 0.85)  # how many votes needed to change state (out of 30)
THRESHOLD_ON = float(environ.get("THRESHOLD_ON", 0.9))
THRESHOLD_OFF = float(environ.get("THRESHOLD_OFF", 1.8))
THRESHOLD_ON_POWER_ALPHA = float(environ.get("THRESHOLD_ON_POWER_ALPHA", 0.4))
THRESHOLD_ON_POWER_DB = float(environ.get("THRESHOLD_ON_POWER_DB", -60.3))
THRESHOLD_ON_POWER_UP = float(environ.get("THRESHOLD_ON_POWER_UP", 0.95))
THRESHOLD_ON_POWER_DOWN = float(environ.get("THRESHOLD_ON_POWER_DOWN", 1.1))
THRESHOLD_ON_SIM = float(environ.get("THRESHOLD_ON_SIM", 0.45))
MIN_RECORD_SEC = 3.0
SAFETY_MARGIN = 0.18
ANALYSIS_WINDOW_SEC = 3.0
DETECT_INTERVAL_SEC = 1.2
BUFFER_SEC = 6.0
PERIODIC_WRITE_TIME = int(environ.get("PERIODIC_WRITE_TIME", 120))
SILENCE_THRESHOLD = 1e-3
NORMALIZE_LOG: bool = False
try:
    NORMALIZE_METHOD: NormalizeMethod = NormalizeMethod(environ.get("NORMALIZE_METHOD", "MAX").lower())
except ValueError as e:
    raise ValueError(str(e))


FILTER_WINDOW_SIZE = int(environ.get("FILTER_WINDOW_SIZE", 30))
FILTER_THRESH_ON_DENSITY: float = float(environ.get("FILTER_THRESH_ON_DENSITY", 0.8))
FILTER_THRESH_OFF_DENSITY: float = float(environ.get("FILTER_THRESH_OFF_DENSITY", 0.7))
FILTER_THRESH_STREAK_PERC: int = int(environ.get("FILTER_THRESH_STREAK_PERC", 50))
FILTER_THRESH_WEIGHTS_ON_DENSITY: float = float(environ.get("FILTER_THRESH_WEIGHTS_ON_DENSITY", 0.5))
FILTER_THRESH_WEIGHTS_OFF_DENSITY: float = float(environ.get("FILTER_THRESH_WEIGHTS_OFF_DENSITY", 0.2))
FILTER_THRESH_MIN_SCORE: float = float(environ.get("FILTER_THRESH_MIN_SCORE", 0.65))
FILTER_METHOD: str = environ.get("FILTER_METHOD", "SCORED_WEIGHTED_DENSITY")
if FILTER_METHOD.lower() not in AnalyzeMethods.__members__.values():
    raise ValueError(f"{FILTER_METHOD} is not a valid AnalyzeMethods")


TRAIN_DURATION = 8.0

TEMPLATE_DIR = DATA_PATH / "templates"
TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATE_LABELS = ("on", "off")

TEMPLATE_FILE_MAKS = "template_{label}_{idx}.npz"
THRESHOLD_FILE = TEMPLATE_DIR / "threshold.json"

TEMPLATE_COUNTERS: dict[str, int] = {
    "on": 30,
    "off": 5,
}
TEMPLATE_COUNTERS_DELAY_SEC: dict[str, int] = {
    "on": 10,
    "off": 30,
}

_EFFECTIVE_SR: int = PREFERRED_RATES[0]

_sr: int


def get_effective_sr():
    return _EFFECTIVE_SR


def set_effective_sr(value: int):
    global _EFFECTIVE_SR
    _EFFECTIVE_SR = value
