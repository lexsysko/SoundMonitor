import torch

import torch.optim as optim
import torch.nn as nn

from SoundMonitor.CNN.cnn_detector import CompressorDetectorCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assume train_loader is initialized with KitchenAudioDataset
model = CompressorDetectorCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training execution
epochs = 15
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for specs, labels in train_loader:
        specs, labels = specs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(specs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(train_loader):.4f}")
