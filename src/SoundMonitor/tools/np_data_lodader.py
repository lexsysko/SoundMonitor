import logging
import re
from pathlib import Path

import numpy as np

from SoundMonitor.settings import DATA_PATH

logger = logging.getLogger(__name__)


def get_audio_files(
    data_path: Path,
    state_str: str,
    prefix: str = "audio",
    extension: str = "npz",
) -> list[Path]:
    clean_state = state_str.strip().lower()
    pattern = f"{prefix}_{clean_state}_*.{extension}"
    files: list = list(data_path.glob(pattern))
    return files


def get_next_audio_filepath(
    data_path: Path,
    state_str: str,
    prefix: str = "audio",
    extension: str = "npz",
    padding: int = 4,
    init_id: int = 0,
    limit: int = 10,
) -> Path | None:
    """Finds the highest index for files matching pattern '{prefix}_{state}_*.{extension}'

    and returns a Path object for the next incremented filename.
    """
    clean_state = state_str.strip().lower()
    pattern = f"{prefix}_{clean_state}_*.{extension}"

    # Regex to capture the trailing numbers right before .npz
    number_pattern = re.compile(rf"^{prefix}_{clean_state}_(\d+)\.{extension}$")

    max_id = init_id
    match = False
    for file_path in data_path.glob(pattern):
        match = number_pattern.match(file_path.name)
        if match:
            max_id = max(max_id, int(match.group(1)))

    next_id = max_id + 1 if match else max_id
    if next_id > limit:
        return None
    # Zero-pad the ID (e.g., 1 -> '0001')
    filename = f"{prefix}_{clean_state}_{next_id:0{padding}d}.{extension}"
    return data_path / filename


def save_audio(audio: np.ndarray, sr: int, filename: Path | None = None):
    filename = filename or DATA_PATH / "audio.npz"
    np.savez_compressed(filename, audio=audio, sr=sr)
    logger.info(f"Saved audio file : {filename}")


def load_audio(filename: Path | None = None) -> tuple[np.ndarray, int]:
    filename = filename or DATA_PATH / "audio.npz"
    if not filename.exists():
        raise FileNotFoundError("Audio file does not exist")
    data = np.load(filename)
    logger.info(f"Loaded audio file : {filename}")
    return data["audio"], data["sr"]


def save_smoothed_audio(
    audio: np.ndarray, sr: int, trusted_state_str: str, limit: int = 10, data_path: Path | None = None
) -> None:
    data_path = data_path or DATA_PATH
    data_path.mkdir(parents=True, exist_ok=True)
    filename = get_next_audio_filepath(data_path, trusted_state_str, limit=limit)
    if filename:
        save_audio(audio=audio, sr=sr, filename=filename)
