import torch
import torchaudio.transforms as T
import numpy as np

# Setup the Spectrogram transformer once
mel_spectrogram = T.MelSpectrogram(sample_rate=22050, n_fft=1024, hop_length=512, n_mels=64)
amplitude_to_db = T.AmplitudeToDB()


def classify_buffer(deque_buffer, model, device="cpu"):
    model.eval()

    # 1. Combine all blocks in your deque into a single continuous 6-second numpy array
    raw_audio = np.concatenate(list(deque_buffer))

    # 2. Normalize to float (-1.0 to 1.0)
    audio_float = raw_audio.astype(np.float32) / 32768.0

    # 3. Convert to PyTorch Tensor and add channel dimension: shape (1, num_samples)
    waveform = torch.tensor(audio_float).unsqueeze(0)

    # 4. Generate the Log-Mel Spectrogram image on the fly
    spec = mel_spectrogram(waveform)
    spec_db = amplitude_to_db(spec)

    # 5. Add batch dimension for PyTorch: shape (1, 1, n_mels, time_frames)
    input_tensor = spec_db.unsqueeze(0).to(device)

    # 6. Run prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    return predicted_class, confidence
