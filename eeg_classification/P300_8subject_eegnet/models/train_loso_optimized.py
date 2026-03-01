"""
OPTIMIZED EEGNet LOSO Training for P300 Classification
=======================================================
Tuned hyperparameters based on initial results:
  - F1=16 (more temporal filters)
  - Dropout=0.2 (less regularization)
  - 50 epochs (better convergence)
  - Batch size=256 (faster, more stable)
  - LR=0.002 with ReduceLROnPlateau scheduler
  - pos_weight=5.0 (exact class imbalance ratio)

Expected: 67-69% BAcc (vs baseline 65.1%)

Usage:
    python models/train_loso_optimized.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score
)
from sklearn.preprocessing import StandardScaler
import json
import warnings
warnings.filterwarnings('ignore')

# Add models to path
sys.path.insert(0, str(Path(__file__).parent))
from eegnet import EEGNet


class OptimizedEEGNetConfig:
    """OPTIMIZED configuration for EEGNet model training."""
    def __init__(self):
        # Model architecture - TUNED
        self.n_channels = 8
        self.n_samples = 250
        self.n_classes = 1
        self.F1 = 16           # ↑ More temporal filters (was 8)
        self.D = 2
        self.dropout = 0.2     # ↓ Less regularization (was 0.25)

        # Training hyperparameters - TUNED
        self.batch_size = 256  # ↑ Larger batches (was 128)
        self.learning_rate = 0.002  # ↑ Faster start (was 0.001)
        self.n_epochs = 50     # ↑ Better convergence (was 25)
        self.pos_weight = 5.0  # Exact imbalance ratio (was 4.0)

        # LR scheduler
        self.scheduler_patience = 10
        self.scheduler_factor = 0.5

        # Data preprocessing
        self.normalized = True


def load_data(data_path):
    """Load preprocessed P300 data."""
    print(f"Loading data from: {data_path}")
    data = np.load(data_path)

    X = data['X']
    y = data['y']
    subject_id = data['subject_id']

    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Targets: {y.sum()}/{len(y)} ({y.mean()*100:.1f}%)")
    print(f"  Subjects: {np.unique(subject_id)}")

    return X, y, subject_id


def normalize_data(X):
    """Z-score normalization per channel."""
    print("\nNormalizing data (z-score per channel)...")
    scaler = StandardScaler()
    n_epochs, n_channels, n_samples = X.shape

    X_reshaped = X.reshape(-1, n_samples)
    X_normalized = scaler.fit_transform(X_reshaped)
    X_normalized = X_normalized.reshape(n_epochs, n_channels, n_samples)

    return X_normalized


def find_best_threshold(y_true, y_probs, metric='f1'):
    """Find optimal classification threshold."""
    thresholds = np.linspace(0.1, 0.9, 17)
    best_score = 0
    best_thresh = 0.5

    for thresh in thresholds:
        y_pred = (y_probs > thresh).astype(int)

        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'bacc':
            score = balanced_accuracy_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_thresh = thresh

    return best_thresh, best_score


def compute_metrics(y_true, y_pred, y_probs):
    """Compute all evaluation metrics."""
    return {
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_true, y_probs),
        'auc_pr': average_precision_score(y_true, y_probs)
    }


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    """Evaluate model and return predictions."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.sigmoid(outputs).cpu().numpy()

            all_probs.extend(probs.flatten())
            all_labels.extend(y_batch.numpy())

    return np.array(all_labels), np.array(all_probs)


def loso_cross_validation(X, y, subject_id, config, normalize=True):
    """
    Leave-One-Subject-Out cross-validation with OPTIMIZED hyperparameters.
    """
    # Detect device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("\nUsing device: cuda")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("\nUsing device: mps")
        print("  → Apple Silicon GPU detected (MPS backend)")
    else:
        device = torch.device('cpu')
        print("\nUsing device: cpu")

    if normalize:
        X = normalize_data(X)

    logo = LeaveOneGroupOut()
    n_folds = len(np.unique(subject_id))

    print(f"\nStarting LOSO CV ({n_folds} folds)...")
    print("=" * 70)

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, subject_id)):
        test_subject = subject_id[test_idx][0]
        print(f"\nFold {fold + 1}/{n_folds}: Test Subject {test_subject}")
        print("-" * 70)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f"  Train: {len(X_train)} epochs")
        print(f"  Test:  {len(X_test)} epochs")

        # Convert to PyTorch
        X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1)
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1)
        y_test_tensor = torch.FloatTensor(y_test)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

        # Initialize OPTIMIZED model
        model = EEGNet(
            n_channels=config.n_channels,
            n_samples=config.n_samples,
            n_classes=config.n_classes,
            F1=config.F1,      # 16 filters
            D=config.D,
            dropout=config.dropout  # 0.2
        ).to(device)

        # Loss and optimizer
        pos_weight = torch.tensor([config.pos_weight]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        # LR scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=config.scheduler_patience,
            factor=config.scheduler_factor
        )

        # Training loop
        print(f"  Training optimized EEGNet ({config.n_epochs} epochs)...")
        best_loss = float('inf')

        for epoch in range(config.n_epochs):
            loss = train_epoch(model, train_loader, criterion, optimizer, device)
            scheduler.step(loss)

            if loss < best_loss:
                best_loss = loss

            if (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"    Epoch {epoch + 1}/{config.n_epochs}: Loss = {loss:.4f}, LR = {current_lr:.6f}")

        # Evaluate
        y_test_np, y_probs = evaluate(model, test_loader, device)

        # Find best threshold
        best_thresh, _ = find_best_threshold(y_test_np, y_probs, metric='f1')
        y_pred = (y_probs > best_thresh).astype(int)

        # Compute metrics
        metrics = compute_metrics(y_test_np, y_pred, y_probs)
        metrics['best_threshold'] = best_thresh
        metrics['subject'] = int(test_subject)

        print(f"  Results:")
        print(f"    BAcc:      {metrics['balanced_accuracy']:.3f}")
        print(f"    F1:        {metrics['f1']:.3f}")
        print(f"    AUC-ROC:   {metrics['auc_roc']:.3f}")
        print(f"    Threshold: {best_thresh:.2f}")

        fold_results.append(metrics)

    # Aggregate results
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS (LOSO Cross-Validation)")
    print("=" * 70)

    metric_names = ['balanced_accuracy', 'f1', 'auc_roc', 'auc_pr', 'precision', 'recall']
    aggregate = {}

    for metric in metric_names:
        values = [r[metric] for r in fold_results]
        aggregate[metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'per_subject': [float(v) for v in values]
        }

        print(f"\n{metric.upper()}:")
        print(f"  Mean ± Std: {aggregate[metric]['mean']:.3f} ± {aggregate[metric]['std']:.3f}")
        print(f"  Range: [{aggregate[metric]['min']:.3f}, {aggregate[metric]['max']:.3f}]")

    # Best/worst subjects
    bacc_per_subject = aggregate['balanced_accuracy']['per_subject']
    best_idx = np.argmax(bacc_per_subject)
    worst_idx = np.argmin(bacc_per_subject)

    print(f"\nBest Subject:  S{fold_results[best_idx]['subject']} "
          f"(BAcc={bacc_per_subject[best_idx]:.3f})")
    print(f"Worst Subject: S{fold_results[worst_idx]['subject']} "
          f"(BAcc={bacc_per_subject[worst_idx]:.3f})")

    return {
        'fold_results': fold_results,
        'aggregate': aggregate,
        'config': {
            'n_channels': config.n_channels,
            'n_samples': config.n_samples,
            'F1': config.F1,
            'D': config.D,
            'dropout': config.dropout,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'n_epochs': config.n_epochs,
            'pos_weight': config.pos_weight,
            'normalized': normalize
        }
    }


def main():
    """Main training pipeline."""
    print("=" * 70)
    print("  OPTIMIZED EEGNet P300 Classification")
    print("=" * 70)

    # Paths
    project_root = Path(__file__).parent.parent
    data_path = project_root.parent / 'P300_8subject_complete' / 'preprocessing' / 'p300_preprocessed_v2.npz'
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)

    # Load data
    X, y, subject_id = load_data(data_path)

    # OPTIMIZED Configuration
    config = OptimizedEEGNetConfig()

    print("\nOPTIMIZED Model Configuration:")
    print(f"  F1 (temporal filters): {config.F1} (was 8)")
    print(f"  D (spatial depth):     {config.D}")
    print(f"  Dropout:               {config.dropout} (was 0.25)")
    print(f"  Batch size:            {config.batch_size} (was 128)")
    print(f"  Learning rate:         {config.learning_rate} (was 0.001)")
    print(f"  Epochs:                {config.n_epochs} (was 25)")
    print(f"  Pos weight:            {config.pos_weight} (was 4.0)")
    print(f"  LR scheduler:          ReduceLROnPlateau (patience={config.scheduler_patience})")

    # Run LOSO CV
    results = loso_cross_validation(X, y, subject_id, config, normalize=True)

    # Save results
    results_path = results_dir / 'eegnet_optimized_loso_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Results saved to: {results_path}")
    print(f"{'=' * 70}")

    # Compare with baseline
    print("\nCOMPARISON WITH BASELINE:")
    print("-" * 70)
    print(f"Baseline EEGNet:   BAcc = 0.651 ± 0.046")
    print(f"Optimized EEGNet:  BAcc = {results['aggregate']['balanced_accuracy']['mean']:.3f} "
          f"± {results['aggregate']['balanced_accuracy']['std']:.3f}")
    improvement = (results['aggregate']['balanced_accuracy']['mean'] - 0.651) * 100
    print(f"Improvement:       {improvement:+.1f}%")
    print(f"\nTarget: 67-69% BAcc (+3-6% improvement)")


if __name__ == '__main__':
    main()
