import os

import sys

from time import sleep

import logging
import numpy as np
from typing import Tuple

from SoundMonitor.settings import CHUNK, set_effective_sr, PREFERRED_RATES, CHANNELS

logger = logging.getLogger(__name__)

try:
    import pyaudio
except ImportError:
    logger.info("ERROR: pyaudio is not installed.\n  sudo apt install -y portaudio19-dev\n  uv add pyaudio\n")
    raise


class suppress_fd_stderr:
    """Redirects C-level stderr (fd 2) to /dev/null during execution."""

    def __enter__(self):
        sys.stderr.flush()
        self.err_fd = 2
        self.null_fd = os.open(os.devnull, os.O_WRONLY)
        self.saved_err_fd = os.dup(self.err_fd)
        os.dup2(self.null_fd, self.err_fd)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.dup2(self.saved_err_fd, self.err_fd)
        os.close(self.null_fd)
        os.close(self.saved_err_fd)


def list_input_devices() -> None:
    with suppress_fd_stderr():
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
    with suppress_fd_stderr():
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

    audio = np.concatenate(frames)

    # force to convert to mono if audio has more than one channel
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Normalize audio to range [-1.0, 1.0]
    audio = audio.astype(np.float32) / 32768.0

    return audio, sr


def play_beep(
    melody: list[int] | None = None,
    duration: float = 0.2,
    pause: float = 0.3,
    sample_rate=None,
    device_index: int | None = None,
):
    melody = melody or [1000, 2000, 1000, 1500]
    with suppress_fd_stderr():
        p = pyaudio.PyAudio()

        rates = (sample_rate,) if sample_rate else PREFERRED_RATES

        # Open output stream
        for rate in rates:
            kwargs = dict(format=pyaudio.paInt16, channels=1, rate=rate, output=True)
            if device_index is not None:
                kwargs["input_device_index"] = device_index
            stream = p.open(**kwargs)
            logger.info("Beep sound opened at {rate} Hz")
            try:
                for frequency in melody:
                    # Generate sine wave
                    t = np.linspace(0, duration, int(rate * duration), False)
                    tone = np.sin(frequency * 2 * np.pi * t)

                    # Convert to 16-bit PCM
                    audio = (tone * 32767).astype(np.int16).tobytes()

                    stream.write(audio)
                    sleep(pause)
                stream.stop_stream()
                stream.close()
                logger.info(f"  Audio opened at {rate} Hz")
                break

            except Exception as e:
                last_err = e
                logger.error(f"  Rate {rate} Hz not accepted: {e}")

        p.terminate()


if __name__ == "__main__":
    play_beep()
