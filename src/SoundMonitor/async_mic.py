import logging
from collections import deque

import numpy as np
import pyaudio
from threading import Lock

from typing import Deque, Tuple

from SoundMonitor.audio_device import open_input_stream
from SoundMonitor.settings import PREFERRED_RATES, set_effective_sr, BUFFER_SEC

logger = logging.getLogger(__name__)


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
        self._pa = pyaudio.PyAudio()
        self._stream, self.sr = open_input_stream(
            self._pa,
            device_index=self.device_index,
            callback=self._callback,
        )
        set_effective_sr(self.sr)
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
