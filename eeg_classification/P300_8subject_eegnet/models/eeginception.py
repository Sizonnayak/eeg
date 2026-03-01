"""
EEG-Inception Model Implementation

Based on:
Zhang et al. (2020) "EEG-Inception: An Accurate and Robust End-to-End Neural Network
for EEG-based Motor Imagery Classification"

Architecture uses Inception modules with multiple kernel sizes to capture
multi-scale temporal patterns in EEG data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock(nn.Module):
    """
    Inception module with multiple parallel convolutional paths.

    Captures multi-scale temporal features by using kernels of different sizes
    (short, medium, long patterns) simultaneously.
    """
    def __init__(self, in_channels, n_filters=8):
        super().__init__()

        # Branch 1: Small kernel (captures fast dynamics, ~20ms)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, n_filters, kernel_size=(1, 5), padding='same', bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ELU()
        )

        # Branch 2: Medium kernel (captures medium dynamics, ~40ms)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, n_filters, kernel_size=(1, 9), padding='same', bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ELU()
        )

        # Branch 3: Large kernel (captures slow dynamics, ~80ms)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, n_filters, kernel_size=(1, 17), padding='same', bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ELU()
        )

        # Branch 4: Average pooling path (preserves strong features)
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.Conv2d(in_channels, n_filters, kernel_size=1, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ELU()
        )

    def forward(self, x):
        # Apply all branches in parallel
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        # Concatenate along channel dimension
        return torch.cat([b1, b2, b3, b4], dim=1)


class EEGInception(nn.Module):
    """
    EEG-Inception network for P300 classification.

    Architecture:
    1. Initial temporal convolution (learns basic temporal patterns)
    2. Depthwise spatial convolution (learns channel combinations)
    3. Multiple Inception blocks (multi-scale temporal feature extraction)
    4. Average pooling and classification

    Args:
        n_channels: Number of EEG channels (8 for our data)
        n_samples: Number of time samples per epoch (250 for our data)
        n_classes: Number of output classes (1 for binary with BCEWithLogitsLoss)
        F1: Number of temporal filters in first layer
        D: Depth multiplier for spatial filters
        n_inception_blocks: Number of Inception modules
        n_filters_inception: Filters per Inception branch
        dropout: Dropout probability
    """
    def __init__(self, n_channels=8, n_samples=250, n_classes=1,
                 F1=8, D=2, n_inception_blocks=2, n_filters_inception=8, dropout=0.5):
        super().__init__()

        # Block 1: Temporal convolution (learns basic temporal patterns)
        # Kernel size 64 = 256ms @ 250Hz, captures P300 latency range
        self.conv1 = nn.Conv2d(1, F1, kernel_size=(1, 64), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        # Block 2: Depthwise spatial convolution (learns channel combinations)
        self.conv2 = nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 4))  # Downsample 250 -> 62
        self.dropout2 = nn.Dropout(dropout)

        # Calculate shape after spatial conv and first pooling
        # (batch, F1*D, 1, n_samples//4)
        n_samples_after_pool = n_samples // 4

        # Block 3: Inception modules for multi-scale temporal features
        self.inception_blocks = nn.ModuleList()
        in_ch = F1 * D
        for i in range(n_inception_blocks):
            self.inception_blocks.append(InceptionBlock(in_ch, n_filters_inception))
            in_ch = n_filters_inception * 4  # 4 branches concatenated

        # Pooling between Inception blocks
        self.pool_inception = nn.AvgPool2d((1, 2))  # 62 -> 31 after first, 31 -> 15 after second
        self.dropout_inception = nn.Dropout(dropout)

        # Calculate final feature size
        # After 2 Inception blocks with pooling: 250 -> 62 -> 31 -> 15
        n_samples_final = n_samples_after_pool
        for _ in range(n_inception_blocks):
            n_samples_final = n_samples_final // 2

        # Final feature dimension: (n_filters_inception * 4) * n_samples_final
        n_features = (n_filters_inception * 4) * n_samples_final

        # Global average pooling + classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(n_filters_inception * 4, n_classes)

    def forward(self, x):
        # Input: (batch, 1, n_channels, n_samples)

        # Block 1: Temporal convolution
        x = self.conv1(x)
        x = self.bn1(x)
        # (batch, F1, n_channels, n_samples)

        # Block 2: Depthwise spatial convolution
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        # (batch, F1*D, 1, n_samples//4)

        # Block 3: Inception modules
        for inception_block in self.inception_blocks:
            x = inception_block(x)
            x = self.pool_inception(x)
            x = self.dropout_inception(x)
        # (batch, n_filters*4, 1, n_samples_final)

        # Global average pooling
        x = self.global_pool(x)  # (batch, n_filters*4, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, n_filters*4)

        # Classification
        x = self.fc(x)  # (batch, n_classes)

        return x


class EEGInceptionConfig:
    """Configuration for EEG-Inception model training."""
    def __init__(self):
        # Model architecture
        self.n_channels = 8
        self.n_samples = 250
        self.n_classes = 1  # Binary classification with BCEWithLogitsLoss
        self.F1 = 8
        self.D = 2
        self.n_inception_blocks = 2
        self.n_filters_inception = 8
        self.dropout = 0.5

        # Training hyperparameters
        self.batch_size = 128
        self.learning_rate = 0.001
        self.n_epochs = 30  # Slightly more epochs for deeper model
        self.pos_weight = 4.0  # Handle class imbalance

        # Data preprocessing
        self.normalized = True  # Z-score normalization per channel


if __name__ == '__main__':
    # Test model instantiation
    config = EEGInceptionConfig()
    model = EEGInception(
        n_channels=config.n_channels,
        n_samples=config.n_samples,
        n_classes=config.n_classes,
        F1=config.F1,
        D=config.D,
        n_inception_blocks=config.n_inception_blocks,
        n_filters_inception=config.n_filters_inception,
        dropout=config.dropout
    )

    # Test forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)

    # Create dummy batch
    batch_size = 16
    x = torch.randn(batch_size, 1, config.n_channels, config.n_samples).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(x)

    print(f"Model: EEG-Inception")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Device: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\nModel architecture test passed!")
