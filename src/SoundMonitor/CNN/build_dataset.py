import logging
import os
import numpy as np
import torch
import torchaudio.transforms as T

# Import your existing get_audio_embedding function
from SoundMonitor.CNN.audio_embedding import get_audio_embedding

logger = logging.getLogger(__name__)


def mix_audio(clean_pcm: np.ndarray, noise_pcm: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Overlays noise onto clean audio at a specific Signal-to-Noise Ratio (SNR).
    A lower snr_db (e.g. 0 or -5) means stronger background noise.
    """
    # Equalize lengths
    if len(noise_pcm) < len(clean_pcm):
        # Repeat noise if it's shorter than clean audio
        repeats = int(np.ceil(len(clean_pcm) / len(noise_pcm)))
        noise_pcm = np.tile(noise_pcm, repeats)[: len(clean_pcm)]
    else:
        # Crop noise if it's longer
        start_idx = np.random.randint(0, len(noise_pcm) - len(clean_pcm) + 1)
        noise_pcm = noise_pcm[start_idx : start_idx + len(clean_pcm)]

    # Calculate signal and noise power
    clean_power = np.mean(clean_pcm**2)
    noise_power = np.mean(noise_pcm**2)

    if noise_power == 0 or clean_power == 0:
        return clean_pcm

    # Calculate required noise gain for target SNR
    # SNR = 10 * log10(clean_power / (gain^2 * noise_power))
    desired_noise_power = clean_power / (10 ** (snr_db / 10))
    gain = np.sqrt(desired_noise_power / noise_power)

    mixed = clean_pcm + gain * noise_pcm

    # Prevent clipping by normalizing if peak exceeds 1.0 (or 32767 for int16)
    max_val = np.max(np.abs(mixed))
    if max_val > 1.0 and clean_pcm.dtype == np.float32:
        mixed = mixed / max_val

    return mixed.astype(clean_pcm.dtype)


def generate_dataset(
    clean_on_files: list[str],
    clean_off_files: list[str],
    noise_files: list[str],
    sample_rate: int = 44100,
    snr_levels: list[int] = [15, 10, 5, 0, -5],  # Decibels of noise strength
):
    X = []
    y = []

    # Helper function to load audio file (replace with your custom np_data_loader)
    def load_pcm(file_path):
        data = np.load(file_path)
        return data["raw_pcm"]  # Assuming your .npz contains 'raw_pcm'

    # --- 1. Process "Compressor ON" (Label 1) ---
    logger.info("Processing Compressor ON samples...")
    for file_path in clean_on_files:
        clean_pcm = load_pcm(file_path)

        # A. Add clean sample
        emb = get_audio_embedding(clean_pcm, raw_pcm_sr=sample_rate)
        X.append(emb)
        y.append(1)

        # B. Generate noisy variations
        for noise_path in noise_files:
            noise_pcm = load_pcm(noise_path)
            for snr in snr_levels:
                mixed_pcm = mix_audio(clean_pcm, noise_pcm, snr_db=snr)
                emb = get_audio_embedding(mixed_pcm, raw_pcm_sr=sample_rate)
                X.append(emb)
                y.append(1)

    # --- 2. Process "Compressor OFF" / Ambient Only (Label 0) ---
    logger.info("Processing Compressor OFF / Noise samples...")
    for file_path in clean_off_files:
        clean_pcm = load_pcm(file_path)

        # A. Add clean ambient OFF sample
        emb = get_audio_embedding(clean_pcm, raw_pcm_sr=sample_rate)
        X.append(emb)
        y.append(0)

        # B. Generate noisy variations for OFF state
        for noise_path in noise_files:
            noise_pcm = load_pcm(noise_path)
            for snr in snr_levels:
                mixed_pcm = mix_audio(clean_pcm, noise_pcm, snr_db=snr)
                emb = get_audio_embedding(mixed_pcm, raw_pcm_sr=sample_rate)
                X.append(emb)
                y.append(0)

    # C. Add pure noise files as Label 0 (Compressor OFF)
    for noise_path in noise_files:
        noise_pcm = load_pcm(noise_path)
        emb = get_audio_embedding(noise_pcm, raw_pcm_sr=sample_rate)
        X.append(emb)
        y.append(0)

    X = np.array(X)
    y = np.array(y)

    logger.info(f"Dataset generated successfully! X shape: {X.shape}, y shape: {y.shape}")
    return X, y


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example Paths
    on_files = ["data/compressor_on_1.npz", "data/compressor_on_2.npz"]
    off_files = ["data/compressor_off_1.npz"]
    noise_files = ["data/water_noise.npz", "data/washing_machine.npz"]

    # Generate X and y
    X, y = generate_dataset(
        clean_on_files=on_files,
        clean_off_files=off_files,
        noise_files=noise_files,
        sample_rate=44100,
    )

    # Save final vectors to disk
    np.savez("data/dataset_embeddings.npz", X=X, y=y)
    logger.info("Saved dataset to data/dataset_embeddings.npz")
