import logging
import sys

import numpy as np
import torch
import torchaudio.transforms as T

from SoundMonitor.tools.np_data_lodader import load_audio

logger = logging.getLogger(__name__)

# 1. Load VGGish model via PyTorch Hub
vggish_model = torch.hub.load("harritaylor/torchvggish", "vggish")
vggish_model.eval()

# 2. Dynamically add local Torch Hub cache directory to sys.path
hub_dir = torch.hub.get_dir()
vggish_repo_path = f"{hub_dir}/harritaylor_torchvggish_master"

if vggish_repo_path not in sys.path:
    sys.path.append(vggish_repo_path)

# print(f"{vggish_repo_path}")

# Now standard imports work directly from Torch Hub cache
try:
    import torchvggish.vggish_input as vggish_input
except ImportError:
    raise


def resample_audio(raw_pcm_array: np.ndarray, orig_freq=44100, new_freq=16000) -> torch.Tensor:
    waveform = torch.from_numpy(raw_pcm_array).float()
    resampler = T.Resample(orig_freq=orig_freq, new_freq=new_freq)
    return resampler(waveform)


def get_audio_embedding(raw_pcm_array: np.ndarray, raw_pcm_sr: int = 16000, model_sr: int = 16000):
    """
    Converts raw 16-bit PCM numpy array directly to a feature embedding.
    """
    # 1. Convert PCM to float32 normalized between -1.0 and 1.0
    if raw_pcm_array.dtype == np.int16:
        audio_float = raw_pcm_array.astype(np.float32) / 32768.0
    elif raw_pcm_array.dtype == np.float32:
        audio_float = raw_pcm_array
    else:
        audio_float = raw_pcm_array.astype(np.float32)

    # 2. Resample if necessary
    if raw_pcm_sr != model_sr:
        waveform = resample_audio(audio_float, orig_freq=raw_pcm_sr, new_freq=model_sr)
    else:
        waveform = torch.from_numpy(audio_float).float()

    logger.debug(f"Input Tensor Shape: {type(waveform)} {waveform.shape}")

    # 3. Convert tensor to NumPy array for vggish_input
    waveform_np = waveform.detach().cpu().numpy()

    # 4. Generate log-mel spectrogram tensor (shape: [N, 1, 96, 64])
    input_batch = vggish_input.waveform_to_examples(waveform_np, model_sr)

    # 5. Extract embeddings bypassing raw _preprocess
    with torch.no_grad():
        # Option A: Pass through features CNN + embeddings head directly
        x = vggish_model.features(input_batch)
        x = torch.flatten(x, 1)
        embedding = vggish_model.embeddings(x)

    # 6. Mean-pool across time frames to get a single 1D vector (128 dimensions)
    return embedding.mean(dim=0).cpu().numpy()

    # # 3. Extract embeddings: Pass model_sr as the 2nd parameter!
    # with torch.no_grad():
    #     # Passing sample rate solves the AttributeError in _preprocess
    #     embedding = model(waveform, model_sr)
    #
    # # 4. Take the mean across time frames -> 1D vector (128 dimensions for VGGish)
    # embedding_vector = embedding.mean(dim=0).cpu().numpy()
    #
    # return embedding_vector


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    raw_pcm, sr = load_audio()

    # Handle numpy array scalar for sample rate if present
    sr = int(sr) if isinstance(sr, np.ndarray) else sr

    duration = len(raw_pcm) / sr
    logger.info(f"Raw PCM : {type(raw_pcm)} {raw_pcm.shape}, sr={sr}, duration={duration:.2f}s")

    em = get_audio_embedding(raw_pcm, raw_pcm_sr=sr)
    logger.info(f"Embedding shape: {em.shape}")
    logger.info(f"Embedding vector preview: {em[:5]}...")

    """
2026-09-26 04:06:56 [INFO] __main__: Raw PCM : <class 'numpy.ndarray'> (132300,), sr=44100, duration=3.00s
2026-09-26 04:06:56 [DEBUG] __main__: Input Tensor Shape: <class 'torch.Tensor'> torch.Size([48000])
2026-09-26 04:06:56 [INFO] __main__: Embedding shape: (128,)
2026-09-26 04:06:56 [INFO] __main__: Embedding vector preview: [0.60298216 0.         0.10870192 0.         0.44294918]...
    """
