"""
EEGNet LOSO Training with Probability Saving for Ensemble
==========================================================
MODIFIED version of train_loso_optimized.py that saves per-fold probabilities
for ensemble learning.

This is the FINAL run to enable proper ensemble with probability averaging.

Usage:
    python models/train_loso_save_probs.py --config [baseline|optimized]
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
import argparse
import warnings
warnings.filterwarnings('ignore')

# Add models to path
sys.path.insert(0, str(Path(__file__).parent))
from eegnet import EEGNet


class ModelConfig:
    """Configuration for EEGNet variants."""
    def __init__(self, config_name='baseline'):
        self.n_channels = 8
        self.n_samples = 250
        self.n_classes = 1
        self.D = 2
        self.normalized = True

        if config_name == 'baseline':
            # Original baseline config
            self.F1 = 8
            self.dropout = 0.25
            self.batch_size = 128
            self.learning_rate = 0.001
            self.n_epochs = 25
            self.pos_weight = 4.0
            self.use_scheduler = False

        elif config_name == 'optimized':
            # Optimized config
            self.F1 = 16
            self.dropout = 0.2
            self.batch_size = 256
            self.learning_rate = 0.002
            self.n_epochs = 50
            self.pos_weight = 5.0
            self.use_scheduler = True
            self.scheduler_patience = 10
            self.scheduler_factor = 0.5


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
    """Evaluate model and return predictions + probabilities."""
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
    """LOSO cross-validation with probability saving."""

    # Detect device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("\nUsing device: mps (Apple Silicon GPU)")
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

        # Convert to PyTorch
        X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1)
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1)
        y_test_tensor = torch.FloatTensor(y_test)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

        # Initialize model
        model = EEGNet(
            n_channels=config.n_channels,
            n_samples=config.n_samples,
            n_classes=config.n_classes,
            F1=config.F1,
            D=config.D,
            dropout=config.dropout
        ).to(device)

        # Loss and optimizer
        pos_weight = torch.tensor([config.pos_weight]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        # Optional LR scheduler
        if config.use_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=config.scheduler_patience,
                factor=config.scheduler_factor
            )

        # Training loop
        print(f"  Training EEGNet ({config.n_epochs} epochs)...")
        for epoch in range(config.n_epochs):
            loss = train_epoch(model, train_loader, criterion, optimizer, device)

            if config.use_scheduler:
                scheduler.step(loss)

            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch + 1}/{config.n_epochs}: Loss = {loss:.4f}")

        # Evaluate - THIS IS KEY: save probabilities
        y_test_np, y_probs = evaluate(model, test_loader, device)

        # Find best threshold
        best_thresh, _ = find_best_threshold(y_test_np, y_probs, metric='f1')
        y_pred = (y_probs > best_thresh).astype(int)

        # Compute metrics
        metrics = compute_metrics(y_test_np, y_pred, y_probs)
        metrics['best_threshold'] = best_thresh
        metrics['subject'] = int(test_subject)

        # **CRITICAL: Save probabilities for ensemble**
        metrics['probabilities'] = y_probs.tolist()
        metrics['true_labels'] = y_test_np.tolist()

        print(f"  BAcc: {metrics['balanced_accuracy']:.3f}, F1: {metrics['f1']:.3f}")

        fold_results.append(metrics)

    # Aggregate
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

    print("\n" + "=" * 70)
    print(f"MEAN BAcc: {aggregate['balanced_accuracy']['mean']:.3f} ± {aggregate['balanced_accuracy']['std']:.3f}")

    return {
        'fold_results': fold_results,
        'aggregate': aggregate,
        'config': {
            'F1': config.F1,
            'dropout': config.dropout,
            'n_epochs': config.n_epochs,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'pos_weight': config.pos_weight
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='baseline',
                       choices=['baseline', 'optimized'],
                       help='Model configuration')
    args = parser.parse_args()

    print("=" * 70)
    print(f"  EEGNet LOSO with Probability Saving ({args.config.upper()})")
    print("=" * 70)

    # Paths
    project_root = Path(__file__).parent.parent
    data_path = project_root.parent / 'P300_8subject_complete' / 'preprocessing' / 'p300_preprocessed_v2.npz'
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)

    # Load data
    X, y, subject_id = load_data(data_path)

    # Config
    config = ModelConfig(args.config)

    print(f"\nConfig: {args.config}")
    print(f"  F1={config.F1}, dropout={config.dropout}, epochs={config.n_epochs}")

    # Run LOSO
    results = loso_cross_validation(X, y, subject_id, config, normalize=True)

    # Save with probabilities
    results_path = results_dir / f'eegnet_{args.config}_with_probs.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print("Probabilities included for ensemble learning!")


if __name__ == '__main__':
    main()
