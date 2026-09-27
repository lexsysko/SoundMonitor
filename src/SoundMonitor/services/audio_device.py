import logging
import numpy as np
import os

import sys
from scipy.signal import resample
from time import sleep
from typing import Tuple, Mapping

from SoundMonitor.settings import CHUNK, set_effective_sr, PREFERRED_RATES, CHANNELS, DATA_PATH


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


class PA:
    _pa: pyaudio.PyAudio | None = None

    @classmethod
    def create(cls) -> pyaudio.PyAudio:
        if cls._pa is None:
            try:
                with suppress_fd_stderr():
                    cls._pa = pyaudio.PyAudio()
            except OSError as e:
                logger.error(str(e))
        return cls._pa  # noqa

    @classmethod
    def terminate(cls) -> None:
        if cls._pa is not None:
            try:
                cls._pa.terminate()
            except OSError as e:
                logger.error(str(e))
            cls._pa = None
        return None

    def __enter__(self):
        return self.create()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.terminate()


def list_input_devices(pa: pyaudio.PyAudio) -> list[Mapping]:
    device_count = pa.get_device_count()
    buff = f"\nAvailable input devices of {device_count} :"
    input_devices: list[Mapping] = []
    for i in range(device_count):
        try:
            # with suppress_fd_stderr():
            info: Mapping = pa.get_device_info_by_index(i)
        except IOError:
            continue
        if info["maxInputChannels"] > 0:
            default_sample_rate = int(info.get("defaultSampleRate", 0))
            input_devices.append(info)
            buff += f"\n  [{i}] {info['name']}  (max in={info['maxInputChannels']}, default rate={default_sample_rate})"
    logger.info(buff + "\n")
    return input_devices


def list_output_devices(pa: pyaudio.PyAudio) -> list[Mapping]:
    device_count = pa.get_device_count()
    buff = f"\nAvailable output devices of {device_count} :"
    output_devices: list[Mapping] = []
    for i in range(device_count):
        try:
            # with suppress_fd_stderr():
            info: Mapping = pa.get_device_info_by_index(i)
        except IOError:
            continue
        if info["maxOutputChannels"] > 0:
            default_sample_rate = int(info.get("defaultSampleRate", 0))
            output_devices.append(info)
            buff += (
                f"\n  [{i}] {info['name']}  (max out={info['maxOutputChannels']}, default rate={default_sample_rate})"
            )
    logger.info(buff + "\n")
    return output_devices


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
            with suppress_fd_stderr():
                stream = pa.open(**kwargs)
            logger.info(f"  Audio opened at {rate} Hz")
            return stream, rate
        except Exception as e:
            last_err = e
            logger.error(f"  Rate {rate} Hz not accepted: {e}")

    raise RuntimeError(f"Could not open input stream at any of {PREFERRED_RATES}. Last error: {last_err}")


def record_seconds(
    pa: pyaudio.PyAudio,
    seconds: float,
    device_index: int | None = None,
) -> Tuple[np.ndarray, int]:
    """
    Blocking record for training.
    Returns (audio_float32, effective_sr).
    """
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

    logger.info("  Recording done")

    audio = np.concatenate(frames)

    # force to convert to mono if audio has more than one channel
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Normalize audio to range [-1.0, 1.0]
    audio = audio.astype(np.float32) / 32768.0

    return audio, sr


def play_beep(
    pa: pyaudio.PyAudio,
    melody: list[int] | None = None,
    duration: float = 0.2,
    pause: float = 0.3,
    sample_rate=None,
    device_index: int | None = None,
):
    melody = melody or [1000, 2000, 1000, 1500]

    rates = (sample_rate,) if sample_rate else PREFERRED_RATES

    # Open output stream
    stream = None
    for rate in rates:
        kwargs = dict(format=pyaudio.paInt16, channels=1, rate=rate, output=True)
        if device_index is not None:
            kwargs["output_device_index"] = device_index
        try:
            with suppress_fd_stderr():
                stream = pa.open(**kwargs)
            logger.info(f"Beep sound opened at {rate} Hz {device_index=}")
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
            logger.info(f"  Beep sound finished")
            break

        except Exception as e:
            logger.error(f"  Rate {rate} Hz not accepted: {e}")


def play_raw_sound(
    pa: pyaudio.PyAudio,
    audio: np.ndarray,
    audio_rate: int | None = None,
    device_rate: int | None = None,
    device_index: int | None = None,
):

    device_rates = (device_rate,) if device_rate else PREFERRED_RATES

    # force to convert to mono if audio has more than one channel
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Open output stream
    stream = None
    for rate in device_rates:
        kwargs = dict(format=pyaudio.paFloat32, channels=1, rate=rate, output=True)
        if device_index is not None:
            kwargs["output_device_index"] = device_index
        try:
            with suppress_fd_stderr():
                stream = pa.open(**kwargs)

            play_audio = audio

            if audio_rate and rate != audio_rate:
                target_len = int(len(audio) * rate / audio_rate)
                play_audio = resample(audio, target_len)

            stream.write(np.ascontiguousarray(np.clip(play_audio, -1.0, 1.0).astype(np.float32)).tobytes())

            stream.stop_stream()
            stream.close()
            logger.info(f"  Play sound finished")
            break

        except Exception as e:
            logger.error(f"  Rate {rate} Hz not accepted: {e}")


def convert_npz_to_wav(audio_folder: str = "audio", id: int | None = None, max_play: int | None = None):
    from SoundMonitor.tools.np_data_lodader import load_audio, get_audio_files, save_audio

    data_path = DATA_PATH / audio_folder
    files = get_audio_files(data_path=data_path, state_str="*", extension="npz")
    max_play = max_play if id is None else None
    for i, filename in enumerate(files[:max_play]):
        if id is not None and not filename.stem.endswith(str(id)):
            continue
        dest_filename = filename.with_suffix(".wav")
        if dest_filename.exists():
            continue
        logger.info(f"{i}. {filename} ")
        audio, sr = load_audio(filename)
        save_audio(audio=audio, sr=sr, filename=dest_filename)


def play_list_states(
    audio_folder: str = "audio",
    id: int | None = None,
    max_play: int = 10,
    save_wav: bool = True,
    device_index: int | None = None,
):
    from SoundMonitor.tools.np_data_lodader import load_audio, get_audio_files, save_audio

    data_path = DATA_PATH / audio_folder
    files_on = get_audio_files(data_path=data_path, state_str="on")
    files_off = get_audio_files(data_path=data_path, state_str="off")
    max_play = max_play if id is None else None

    with PA() as pa:
        list_output_devices(pa)
        # play_beep(pa, device_index=device_index)
        logger.info(f"ON. {max_play} / {len(files_on)} ")
        for i, filename in enumerate(files_on[:max_play]):
            if id is not None and not filename.stem.endswith(str(id)):
                continue
            logger.info(f"{i}. {filename} ")
            audio, sr = load_audio(filename)
            play_raw_sound(pa, audio, sr, device_index=device_index)
            if save_wav:
                save_audio(audio=audio, sr=sr, filename=filename.with_suffix(".wav"))

            sleep(1)

        logger.info(f"OFF. {max_play} / {len(files_on)} ")
        for i, filename in enumerate(files_off[:max_play]):
            if id is not None and not filename.stem.endswith(str(id)):
                continue
            logger.info(f"{i}. {filename} ")
            audio, sr = load_audio(filename)
            play_raw_sound(pa, audio, sr)
            if save_wav:
                save_audio(audio=audio, sr=sr, filename=filename.with_suffix(".wav"))
            sleep(1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        # stream=sys.stdout,
        force=True,
    )
    convert_npz_to_wav()
    # id = int(sys.argv[1]) if len(sys.argv) > 1 else None
    # play_list_states(id=id)
