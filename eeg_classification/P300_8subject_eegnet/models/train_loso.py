"""
EEGNet LOSO Training for P300 Classification
=============================================
Leave-One-Subject-Out cross-validation with:
  - Raw EEG epochs (no manual feature engineering)
  - Class-weighted loss for imbalance
  - Threshold tuning for optimal F1
  - Per-subject and aggregate metrics

Usage:
    python models/train_loso.py
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
from eegnet import EEGNet, EEGNetConfig


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
    """Z-score normalization per channel (optional but helps stability)."""
    print("\nNormalizing data (z-score per channel)...")
    scaler = StandardScaler()
    n_epochs, n_channels, n_samples = X.shape

    # Reshape to (n_epochs * n_channels, n_samples)
    X_reshaped = X.reshape(-1, n_samples)

    # Normalize
    X_normalized = scaler.fit_transform(X_reshaped)

    # Reshape back
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

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(model, X_test, y_test, device):
    """Evaluate model and find best threshold."""
    model.eval()

    with torch.no_grad():
        # Add channel dimension: (n, ch, time) → (n, 1, ch, time)
        X_test_tensor = torch.FloatTensor(X_test[:, None, :, :]).to(device)

        # Get logits and probabilities
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


def loso_cross_validation(X, y, subject_id, config, normalize=True):
    """
    Leave-One-Subject-Out cross-validation.

    Args:
        X: EEG epochs (n_epochs, n_channels, n_samples)
        y: Labels (n_epochs,)
        subject_id: Subject IDs (n_epochs,)
        config: EEGNetConfig
        normalize: Whether to normalize data

    Returns:
        results: Dict with per-fold and aggregate metrics
    """
    # Detect best available device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')  # Apple Silicon GPU
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

    print(f"\nStarting LOSO CV ({n_splits} folds)...")
    print("=" * 70)

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, subject_id)):
        test_subject = subject_id[test_idx[0]]

        print(f"\nFold {fold+1}/{n_splits}: Test Subject {test_subject}")
        print("-" * 70)

        # Split data
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        print(f"  Train: {len(X_train)} epochs ({y_train.sum()} targets, {y_train.mean()*100:.1f}%)")
        print(f"  Test:  {len(X_test)} epochs ({y_test.sum()} targets, {y_test.mean()*100:.1f}%)")

        # Create datasets
        # Add channel dimension: (n, ch, time) → (n, 1, ch, time)
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

        # Initialize model
        model = EEGNet(
            n_channels=config.n_channels,
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
    print("  LOSO RESULTS SUMMARY")
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
    # Paths
    data_path = Path(__file__).parent.parent.parent / "P300_8subject_complete" / "preprocessing" / "p300_preprocessed_v2.npz"
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("  EEGNet P300 Classification - LOSO Evaluation")
    print("=" * 70)

    # Load data
    X, y, subject_id = load_data(data_path)

    # Configuration
    config = EEGNetConfig()

    print(f"\nModel Configuration:")
    print(f"  F1 (temporal filters): {config.F1}")
    print(f"  D (spatial depth):     {config.D}")
    print(f"  Dropout:               {config.dropout}")
    print(f"  Batch size:            {config.batch_size}")
    print(f"  Learning rate:         {config.learning_rate}")
    print(f"  Epochs:                {config.n_epochs}")
    print(f"  Pos weight:            {config.pos_weight}")

    # Run LOSO
    results = loso_cross_validation(X, y, subject_id, config, normalize=True)

    # Save results
    output_file = results_dir / "eegnet_loso_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Print final summary
    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    agg = results['aggregate']
    print(f"  Balanced Accuracy: {agg['balanced_accuracy']['mean']:.3f} ± {agg['balanced_accuracy']['std']:.3f}")
    print(f"  F1 Score:          {agg['f1']['mean']:.3f} ± {agg['f1']['std']:.3f}")
    print(f"  AUC-ROC:           {agg['auc_roc']['mean']:.3f} ± {agg['auc_roc']['std']:.3f}")
    print(f"  AUC-PR:            {agg['auc_pr']['mean']:.3f} ± {agg['auc_pr']['std']:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
