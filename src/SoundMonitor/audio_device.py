import logging
import numpy as np
from typing import Tuple

from SoundMonitor.main import logger
from SoundMonitor.settings import CHUNK, set_effective_sr, PREFERRED_RATES, CHANNELS

logger = logging.getLogger(__name__)

try:
    import pyaudio
except ImportError:
    logger.info("ERROR: pyaudio is not installed.\n  sudo apt install -y portaudio19-dev\n  uv add pyaudio\n")
    raise


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

    raise RuntimeError(f"Could not open input stream at any of {PREFERRED_RATES}. Last error: {last_err}")


def record_seconds(
    seconds: float,
    device_index: int | None = None,
) -> Tuple[np.ndarray, int]:
    """
    Blocking record for training.
    Returns (audio_float32, effective_sr).
    """
    pa = pyaudio.PyAudio()
    stream, sr = open_input_stream(pa, device_index=device_index)
    set_effective_sr(sr)

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
