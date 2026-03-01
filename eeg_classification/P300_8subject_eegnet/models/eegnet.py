"""
EEGNet Model for P300 Classification
=====================================
Based on: Lawhern et al. (2018) - EEGNet: A Compact Convolutional Network for EEG-based BCIs

Architecture:
  1. Temporal Conv: Captures waveform patterns (kernel size 64 samples ~256ms)
  2. Depthwise Spatial Conv: Channel interactions (learns spatial filters)
  3. Separable Conv: Depthwise + Pointwise (feature refinement)
  4. FC Classifier: Binary classification (target/non-target)

Key Features:
  - Learns spatial/temporal filters end-to-end (no manual xDAWN needed)
  - Depthwise convolutions: fewer parameters, less overfitting
  - Dropout: regularization for limited training data
  - Batch normalization: training stability

Input: Raw EEG epochs (B, 1, n_channels, n_samples)
Output: Class logits (B, n_classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    def __init__(self, n_channels=8, n_samples=250, n_classes=1, F1=8, D=2, dropout=0.25):
        """
        EEGNet model for EEG classification.

        Args:
            n_channels: Number of EEG channels (default: 8)
            n_samples: Number of timepoints per epoch (default: 250)
            n_classes: Number of output units (default: 1 for binary with BCEWithLogitsLoss)
            F1: Number of temporal filters (default: 8)
            D: Depth multiplier for spatial filters (default: 2)
            dropout: Dropout rate (default: 0.25)
        """
        super().__init__()

        # Block 1: Temporal convolution
        # Captures P300 waveform patterns over ~256ms window
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, 64),      # Kernel size: 64 samples @ 250Hz = 256ms
            padding='same',
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # Block 2: Depthwise spatial convolution
        # Learns spatial filters (channel combinations) for each temporal filter
        self.conv2 = nn.Conv2d(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(n_channels, 1),  # Spatial: combines all channels
            groups=F1,                     # Depthwise: one spatial filter per temporal
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))  # Downsample time by 4

        # Block 3: Separable convolution
        # Depthwise temporal + pointwise channel mixing
        self.conv3 = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F1 * D,
            kernel_size=(1, 16),
            padding='same',
            groups=F1 * D,  # Depthwise
            bias=False
        )
        self.conv4 = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F1 * D,
            kernel_size=1,  # Pointwise: 1×1 conv
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(F1 * D)
        self.pool2 = nn.AvgPool2d((1, 8))  # Downsample time by 8

        # Dropout regularization
        self.dropout = nn.Dropout(dropout)

        # Classifier
        # After 2 pooling layers (4×8=32), timepoints: 250 // 32 ≈ 7-8
        n_out_features = F1 * D * (n_samples // 32)
        self.fc = nn.Linear(n_out_features, n_classes)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, 1, n_channels, n_samples)

        Returns:
            logits: Class logits (batch_size, n_classes)
        """
        # Block 1: Temporal convolution + ELU + Dropout
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)

        # Block 2: Spatial convolution + ELU + Pooling + Dropout
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.dropout(x)

        # Block 3: Separable convolution + ELU + Pooling + Dropout
        x = self.conv3(x)
        x = F.elu(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout(x)

        # Flatten and classify
        x = x.flatten(1)
        logits = self.fc(x)

        return logits


class EEGNetConfig:
    """Default EEGNet configuration for P300 classification."""

    # Model architecture
    n_channels = 8
    n_samples = 250
    n_classes = 1  # Binary classification with BCEWithLogitsLoss
    F1 = 8          # Temporal filters
    D = 2           # Depth multiplier
    dropout = 0.25

    # Training
    batch_size = 128
    learning_rate = 0.001
    weight_decay = 1e-4
    n_epochs = 25
    grad_clip = 1.0

    # Class imbalance (targets:non-targets = 1:5)
    pos_weight = 4.0  # Penalize target misclassification more

    # Device (supports Apple Silicon MPS)
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'  # Apple Silicon GPU
    else:
        device = 'cpu'


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model instantiation
    config = EEGNetConfig()
    model = EEGNet(
        n_channels=config.n_channels,
        n_samples=config.n_samples,
        n_classes=config.n_classes,
        F1=config.F1,
        D=config.D,
        dropout=config.dropout
    )

    print("=" * 70)
    print("  EEGNet Model")
    print("=" * 70)
    print(f"  Channels: {config.n_channels}")
    print(f"  Samples:  {config.n_samples}")
    print(f"  F1 (temporal filters): {config.F1}")
    print(f"  D (spatial depth): {config.D}")
    print(f"  Dropout: {config.dropout}")
    print(f"  Parameters: {count_parameters(model):,}")
    print("=" * 70)

    # Test forward pass
    batch_size = 16
    x = torch.randn(batch_size, 1, config.n_channels, config.n_samples)
    logits = model(x)
    print(f"\n  Input shape:  {x.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Expected:     ({batch_size}, {config.n_classes}) or ({batch_size},)")

    expected_shape = (batch_size, config.n_classes) if config.n_classes > 1 else (batch_size,)
    if logits.dim() == 1:
        assert logits.shape == (batch_size,), "Output shape mismatch!"
    else:
        assert logits.shape == (batch_size, config.n_classes), "Output shape mismatch!"
    print("\n  ✓ Model test passed")
