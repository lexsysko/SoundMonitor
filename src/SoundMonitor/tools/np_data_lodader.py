import logging
from pathlib import Path

import numpy as np

from SoundMonitor.settings import DATA_PATH

logger = logging.getLogger(__name__)


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
