"""VT-CNN2-inspired 1D-CNN for 24-class Automatic Modulation Classification.

Architecture designed for RadioML 2018.01A:
  Input:  (batch, 2, 1024) — I/Q channels × 1024 time samples
  Output: (batch, 24) — 24 modulation class logits

Three Conv1D blocks with BatchNorm + ReLU + MaxPool + Dropout,
followed by two Dense layers. Dynamically computes the flatten size
so the architecture is robust to input length changes.

This model is designed to be trainable on CPU in ~20-30 minutes with
a subsampled dataset (50K-100K samples), while still achieving useful
accuracy (>60% overall at SNR >= 10 dB).
"""
import os
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .radioml_loader import RADIOML_CLASSES, NUM_CLASSES

# Default location for saved model weights
WEIGHTS_DIR = Path(__file__).parent / "weights"
DEFAULT_WEIGHTS_PATH = WEIGHTS_DIR / "amc_radioml2018.pt"


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class VTCNN2(nn.Module):
    """ResNet1D variant for 24-class modulation classification to achieve >95% accuracy.
    (Kept name VTCNN2 for compatibility with existing imports)

    Input: (batch, 2, 1024) tensor of I/Q samples.
    Output: (batch, num_classes) logits.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, input_length: int = 1024):
        super().__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(ResBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get softmax probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    save_path: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple["VTCNN2", dict]:
    """Train the AMC CNN model.

    Args:
        X_train: Training data, shape (N, 2, 1024).
        y_train: Training labels, shape (N,) int64.
        X_val: Validation data.
        y_val: Validation labels.
        num_epochs: Number of training epochs.
        batch_size: Batch size.
        learning_rate: Initial learning rate.
        save_path: Path to save best model weights.
        device: 'cuda' or 'cpu'. Auto-detected if None.

    Returns:
        (trained_model, training_history_dict)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    input_length = X_train.shape[2]
    num_classes = int(y_train.max()) + 1

    print(f"[AMC] Training on device: {device}")
    print(f"[AMC] Training samples: {len(X_train)}, Validation: {len(X_val)}")
    print(f"[AMC] Input shape: (2, {input_length}), Classes: {num_classes}")

    model = VTCNN2(num_classes=num_classes, input_length=input_length).to(device)

    train_ds = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train),
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5,
    )

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
    }
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y_batch)
            train_correct += (logits.argmax(1) == y_batch).sum().item()
            train_total += len(y_batch)

        # --- Validation ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * len(y_batch)
                val_correct += (logits.argmax(1) == y_batch).sum().item()
                val_total += len(y_batch)

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        avg_train_loss = train_loss / max(train_total, 1)
        avg_val_loss = val_loss / max(val_total, 1)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        scheduler.step(avg_val_loss)

        print(
            f"  Epoch {epoch + 1:3d}/{num_epochs}: "
            f"train_loss={avg_train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={avg_val_loss:.4f} val_acc={val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_path:
                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"  [v] Saved best model (val_acc={val_acc:.4f})")

    print(f"[AMC] Training complete. Best val accuracy: {best_val_acc:.4f}")

    # Load best model if saved
    if save_path and os.path.exists(save_path):
        model.load_state_dict(
            torch.load(save_path, map_location=device, weights_only=True)
        )

    return model, history


def load_model(
    weights_path: Optional[str] = None,
    num_classes: int = NUM_CLASSES,
    input_length: int = 1024,
    device: Optional[str] = None,
) -> "VTCNN2":
    """Load a trained model from saved weights.

    Args:
        weights_path: Path to .pt weights file. Defaults to built-in path.
        num_classes: Number of output classes.
        input_length: Expected input sequence length.
        device: 'cuda' or 'cpu'.

    Returns:
        Loaded model in eval mode, or None if weights not found.
    """
    if weights_path is None:
        weights_path = str(DEFAULT_WEIGHTS_PATH)

    if not os.path.exists(weights_path):
        return None

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = VTCNN2(num_classes=num_classes, input_length=input_length).to(device)
    model.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True)
    )
    model.eval()
    return model


def classify_iq_cnn(
    iq_complex: np.ndarray,
    model: Optional["VTCNN2"] = None,
    device: Optional[str] = None,
    frame_length: int = 1024,
) -> dict:
    """Classify modulation of a complex IQ signal using the trained CNN.

    Takes arbitrary-length complex IQ, segments into frames, runs inference,
    and aggregates predictions.

    Args:
        iq_complex: 1D complex64 array of IQ samples
        model: pre-loaded model (loads default if None)
        device: 'cuda' or 'cpu'
        frame_length: frame size for segmentation

    Returns:
        dict with 'modulation', 'confidence', 'evidence', 'probabilities'
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model is None:
        model = load_model(device=device)
        if model is None:
            return {
                "modulation": None,
                "confidence": 0.0,
                "evidence": "No trained CNN model available",
                "probabilities": {},
            }

    model.eval()

    # Segment into frames of (2, frame_length)
    I = np.real(iq_complex).astype(np.float32)
    Q = np.imag(iq_complex).astype(np.float32)
    n_frames = len(I) // frame_length
    if n_frames == 0:
        return {
            "modulation": None,
            "confidence": 0.0,
            "evidence": f"Signal too short for CNN ({len(I)} < {frame_length})",
            "probabilities": {},
        }

    I_frames = I[: n_frames * frame_length].reshape(n_frames, frame_length)
    Q_frames = Q[: n_frames * frame_length].reshape(n_frames, frame_length)
    frames = np.stack([I_frames, Q_frames], axis=1)  # (n_frames, 2, frame_length)

    # Normalize each frame
    energy = np.sqrt(np.mean(frames ** 2, axis=(1, 2), keepdims=True))
    frames /= np.maximum(energy, 1e-10)

    # Run inference
    batch = torch.FloatTensor(frames).to(device)
    with torch.no_grad():
        probas = model.predict_proba(batch).cpu().numpy()  # (n_frames, num_classes)

    # Aggregate: average probabilities across frames
    avg_probas = np.mean(probas, axis=0)
    std_probas = np.std(probas, axis=0)
    consistency = float(1.0 - np.mean(std_probas))

    best_idx = int(np.argmax(avg_probas))
    best_prob = float(avg_probas[best_idx])
    best_mod = RADIOML_CLASSES[best_idx]

    # Confidence = probability * consistency factor
    confidence = best_prob * (0.6 + 0.4 * consistency)

    # Top-5 probabilities for evidence
    top5_idx = np.argsort(avg_probas)[::-1][:5]
    top5 = {RADIOML_CLASSES[i]: float(avg_probas[i]) for i in top5_idx}

    return {
        "modulation": best_mod.lower(),
        "confidence": confidence,
        "evidence": {
            "method": "CNN (VT-CNN2)",
            "frames_analyzed": n_frames,
            "frame_consistency": consistency,
            "top5_probabilities": top5,
        },
        "probabilities": {RADIOML_CLASSES[i]: float(avg_probas[i]) for i in range(NUM_CLASSES)},
    }
