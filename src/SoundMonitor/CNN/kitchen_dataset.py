import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T

"""
Raw Audio (3s) ──► Log-Mel Spectrogram (Image) ──► 2D CNN ──► [Compressor ON / OFF]
"""


class KitchenAudioDataset(Dataset):
    def __init__(self, file_paths, labels, sample_rate=22050, duration=3):
        self.file_paths = file_paths
        self.labels = labels
        self.target_samples = sample_rate * duration

        self.mel_spectrogram = T.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, hop_length=512, n_mels=64)
        self.amplitude_to_db = T.AmplitudeToDB()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        waveform, sr = torchaudio.load(path)

        # Convert stereo to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Pad or truncate waveform to standard length
        if waveform.shape[1] < self.target_samples:
            padding = self.target_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            waveform = waveform[:, : self.target_samples]

        # Compute Log-Mel Spectrogram
        spec = self.mel_spectrogram(waveform)
        spec = self.amplitude_to_db(spec)

        return spec, torch.tensor(self.labels[idx], dtype=torch.long)
