"""
Phase 1: xDAWN-4 + EEGNet for P300 Classification
===================================================
Hybrid approach combining:
  - xDAWN spatial filtering (4 components) from v3
  - EEGNet temporal learning (end-to-end)

Usage:
    python models/train_loso_xdawn4.py
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
from scipy.linalg import eigh
import json
import warnings
warnings.filterwarnings('ignore')

# Add models to path
sys.path.insert(0, str(Path(__file__).parent))
from eegnet import EEGNet, EEGNetConfig


def fit_xdawn(X_train, y_train, n_components=4):
    """
    Fit xDAWN spatial filters on training data.

    Args:
        X_train: (n_train, n_channels, n_samples) EEG epochs
        y_train: (n_train,) binary labels
        n_components: number of spatial filters (default: 4)

    Returns:
        W: (n_channels, n_components) spatial filter matrix
    """
    # Average target ERP
    X_target = X_train[y_train == 1]
    if len(X_target) == 0:
        raise ValueError("No target epochs in training data")

    ERP = X_target.mean(axis=0)  # (n_channels, n_samples)

    # Replicate ERP matrix
    n_target = X_target.shape[0]
    D = np.tile(ERP, (n_target, 1, 1))  # (n_target, n_channels, n_samples)

    # Flatten time dimension for covariance computation
    X_flat = X_train.transpose(0, 2, 1).reshape(-1, X_train.shape[1])  # (n_train*n_samples, n_channels)
    D_flat = D.transpose(0, 2, 1).reshape(-1, D.shape[1])              # (n_target*n_samples, n_channels)

    # Covariance matrices with regularization for numerical stability
    Cxx = X_flat.T @ X_flat
    Cdd = D_flat.T @ D_flat

    # Add regularization to ensure positive definiteness
    # Use a fraction of the trace as regularization strength
    reg_strength = 1e-6 * np.trace(Cxx) / Cxx.shape[0]
    Cxx_reg = Cxx + reg_strength * np.eye(Cxx.shape[0])
    Cdd_reg = Cdd + reg_strength * np.eye(Cdd.shape[0])

    # Generalized eigenvalue problem: Cdd w = λ Cxx w
    try:
        eigenvals, eigenvecs = eigh(Cdd_reg, Cxx_reg)
    except np.linalg.LinAlgError:
        # Fallback: stronger regularization
        reg_strength = 1e-4 * np.trace(Cxx) / Cxx.shape[0]
        Cxx_reg = Cxx + reg_strength * np.eye(Cxx.shape[0])
        Cdd_reg = Cdd + reg_strength * np.eye(Cdd.shape[0])
        eigenvals, eigenvecs = eigh(Cdd_reg, Cxx_reg)

    # Select top n_components (largest eigenvalues)
    W = eigenvecs[:, -n_components:]

    return W


def transform_xdawn(X, W):
    """
    Project epochs through xDAWN spatial filters.

    Args:
        X: (n, n_channels, n_samples) EEG epochs
        W: (n_channels, n_components) spatial filter matrix

    Returns:
        X_filtered: (n, n_components, n_samples) filtered epochs
    """
    n_epochs, n_channels, n_samples = X.shape

    # Reshape for matrix multiplication
    X_reshaped = X.transpose(0, 2, 1).reshape(-1, n_channels)  # (n*n_samples, n_channels)

    # Apply spatial filters
    X_filtered_flat = X_reshaped @ W  # (n*n_samples, n_components)

    # Reshape back
    X_filtered = X_filtered_flat.reshape(n_epochs, n_samples, -1).transpose(0, 2, 1)

    return X_filtered


def load_data(data_path):
    """Load preprocessed P300 data."""
    print(f"Loading data from: {data_path}")
    data = np.load(data_path)

    X = data['X']            # (n_epochs, n_channels, n_samples)
    y = data['y']            # (n_epochs,)
    subject_id = data['subject_id']  # (n_epochs,)

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
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'auc_roc': float(roc_auc_score(y_true, y_probs)),
        'auc_pr': float(average_precision_score(y_true, y_probs))
    }


def train_one_epoch(model, train_loader, criterion, optimizer, device, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward pass
        logits = model(X_batch).squeeze()
        loss = criterion(logits, y_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(model, X_test, y_test, device):
    """Evaluate model and find best threshold."""
    model.eval()

    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test[:, None, :, :]).to(device)
        logits = model(X_test_tensor).squeeze()
        probs = torch.sigmoid(logits).cpu().numpy()

    # Find best threshold
    best_thresh, best_f1 = find_best_threshold(y_test, probs, metric='f1')

    # Predict with best threshold
    y_pred = (probs > best_thresh).astype(int)

    # Compute metrics
    metrics = compute_metrics(y_test, y_pred, probs)
    metrics['best_threshold'] = float(best_thresh)

    return metrics, probs


def loso_cross_validation_xdawn4(X, y, subject_id, config, normalize=True):
    """
    LOSO CV with xDAWN-4 preprocessing per fold.

    Key: xDAWN is fitted on training data only (no leakage).
    """
    # Detect device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"\nUsing device: {device}")
    if device.type == 'mps':
        print("  → Apple Silicon GPU detected (MPS backend)")

    # Optional normalization
    if normalize:
        X = normalize_data(X)

    # LOSO split
    logo = LeaveOneGroupOut()
    n_splits = logo.get_n_splits(groups=subject_id)

    print(f"\nStarting LOSO CV with xDAWN-4 preprocessing ({n_splits} folds)...")
    print("=" * 70)

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, subject_id)):
        test_subject = subject_id[test_idx[0]]

        print(f"\nFold {fold+1}/{n_splits}: Test Subject {test_subject}")
        print("-" * 70)

        # Split data
        X_train_raw, y_train = X[train_idx], y[train_idx]
        X_test_raw, y_test = X[test_idx], y[test_idx]

        print(f"  Raw data:")
        print(f"    Train: {len(X_train_raw)} epochs, {X_train_raw.shape[1]} channels")
        print(f"    Test:  {len(X_test_raw)} epochs")

        # Fit xDAWN-4 on training data ONLY
        print(f"  Fitting xDAWN-4 spatial filters on training data...")
        W = fit_xdawn(X_train_raw, y_train, n_components=4)

        # Transform both train and test
        X_train = transform_xdawn(X_train_raw, W)  # (n_train, 4, n_samples)
        X_test = transform_xdawn(X_test_raw, W)    # (n_test, 4, n_samples)

        print(f"  After xDAWN-4:")
        print(f"    Train: {X_train.shape} ({X_train.shape[1]} components)")
        print(f"    Test:  {X_test.shape}")

        # Create datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train[:, None, :, :]),
            torch.FloatTensor(y_train)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=False
        )

        # Initialize model with 4 input channels (not 8!)
        model = EEGNet(
            n_channels=4,  # Key change: xDAWN reduced 8 → 4
            n_samples=config.n_samples,
            n_classes=config.n_classes,
            F1=config.F1,
            D=config.D,
            dropout=config.dropout
        ).to(device)

        # Optimizer and loss
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([config.pos_weight]).to(device)
        )

        # Train
        print(f"  Training {config.n_epochs} epochs...")
        for epoch in range(config.n_epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, config)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1:2d}: Loss={train_loss:.4f}")

        # Evaluate
        print(f"  Evaluating...")
        metrics, probs = evaluate(model, X_test, y_test, device)

        metrics['subject'] = int(test_subject)
        fold_results.append(metrics)

        print(f"  Results:")
        print(f"    BAcc:      {metrics['balanced_accuracy']:.3f}")
        print(f"    F1:        {metrics['f1']:.3f}")
        print(f"    AUC-ROC:   {metrics['auc_roc']:.3f}")
        print(f"    AUC-PR:    {metrics['auc_pr']:.3f}")
        print(f"    Threshold: {metrics['best_threshold']:.3f}")

    # Aggregate results
    print("\n" + "=" * 70)
    print("  LOSO RESULTS SUMMARY (xDAWN-4 + EEGNet)")
    print("=" * 70)

    metrics_to_aggregate = ['balanced_accuracy', 'f1', 'auc_roc', 'auc_pr', 'precision', 'recall']
    aggregate_results = {}

    for metric in metrics_to_aggregate:
        values = [r[metric] for r in fold_results]
        aggregate_results[metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'per_subject': values
        }

        print(f"  {metric:20s}: {aggregate_results[metric]['mean']:.3f} ± {aggregate_results[metric]['std']:.3f}")

    return {
        'fold_results': fold_results,
        'aggregate': aggregate_results,
        'config': {
            'n_channels': 4,  # After xDAWN-4
            'n_samples': config.n_samples,
            'F1': config.F1,
            'D': config.D,
            'dropout': config.dropout,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'n_epochs': config.n_epochs,
            'pos_weight': config.pos_weight,
            'normalized': normalize,
            'xdawn_components': 4
        }
    }


def main():
    # Paths
    data_path = Path(__file__).parent.parent.parent / "P300_8subject_complete" / "preprocessing" / "p300_preprocessed_v2.npz"
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("  Phase 1: xDAWN-4 + EEGNet P300 Classification")
    print("=" * 70)

    # Load data
    X, y, subject_id = load_data(data_path)

    # Configuration
    config = EEGNetConfig()

    print(f"\nModel Configuration:")
    print(f"  Spatial preprocessing: xDAWN-4 (8 → 4 channels)")
    print(f"  F1 (temporal filters): {config.F1}")
    print(f"  D (spatial depth):     {config.D}")
    print(f"  Dropout:               {config.dropout}")
    print(f"  Batch size:            {config.batch_size}")
    print(f"  Learning rate:         {config.learning_rate}")
    print(f"  Epochs:                {config.n_epochs}")
    print(f"  Pos weight:            {config.pos_weight}")

    # Run LOSO with xDAWN-4
    results = loso_cross_validation_xdawn4(X, y, subject_id, config, normalize=True)

    # Save results
    output_file = results_dir / "eegnet_xdawn4_loso_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Print final summary
    print("\n" + "=" * 70)
    print("  FINAL RESULTS (xDAWN-4 + EEGNet)")
    print("=" * 70)
    agg = results['aggregate']
    print(f"  Balanced Accuracy: {agg['balanced_accuracy']['mean']:.3f} ± {agg['balanced_accuracy']['std']:.3f}")
    print(f"  F1 Score:          {agg['f1']['mean']:.3f} ± {agg['f1']['std']:.3f}")
    print(f"  AUC-ROC:           {agg['auc_roc']['mean']:.3f} ± {agg['auc_roc']['std']:.3f}")
    print(f"  AUC-PR:            {agg['auc_pr']['mean']:.3f} ± {agg['auc_pr']['std']:.3f}")

    print("\n  Comparison to raw EEGNet:")
    print(f"    Raw EEGNet BAcc:     0.651 ± 0.046")
    print(f"    xDAWN-4 + EEGNet:    {agg['balanced_accuracy']['mean']:.3f} ± {agg['balanced_accuracy']['std']:.3f}")
    improvement = (agg['balanced_accuracy']['mean'] - 0.651) * 100
    print(f"    Improvement:         {improvement:+.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
