import torch
import numpy as np
import librosa
from torch_vggish_yamnet import yamnet, input_proc

# 1. Load the pre-trained YAMNet model
model = yamnet.yamnet(pretrained=True)
model.eval()


# Helper to process raw audio float array -> 1024-D embedding vector
def get_audio_embedding(y, sr=16000):
    # YAMNet expects 16kHz audio
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)

    converter = input_proc.WaveformToInput()
    input_tensor = converter(torch.from_numpy(y).float(), 16000)

    with torch.no_grad():
        # Get embeddings from intermediate feature layer
        embeddings, _ = model(input_tensor)

    # Pool across time frames to create a single 1D vector per audio clip
    embedding_vector = embeddings.mean(dim=0).cpu().numpy()
    return embedding_vector
