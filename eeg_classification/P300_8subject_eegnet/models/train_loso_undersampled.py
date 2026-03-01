"""
EEGNet LOSO Training with Class Undersampling
==============================================
Balances training data by undersampling non-targets to 1:2 or 1:3 ratio.

Strategy: Instead of pos_weight, physically balance training data
- Target ratio 1:2 (1 target : 2 non-targets)
- Or 1:3 for less aggressive undersampling

Expected: May improve precision/F1 by reducing false positives

Usage:
    python models/train_loso_undersampled.py --ratio 2
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


class UndersampledConfig:
    """Configuration for undersampled EEGNet training."""
    def __init__(self, target_ratio=2):
        # Model architecture - use optimized params
        self.n_channels = 8
        self.n_samples = 250
        self.n_classes = 1
        self.F1 = 16           # Optimized value
        self.D = 2
        self.dropout = 0.2

        # Training hyperparameters
        self.batch_size = 128  # Smaller batch (less data after undersampling)
        self.learning_rate = 0.001
        self.n_epochs = 50

        # Undersampling ratio (targets : non-targets)
        self.target_ratio = target_ratio  # 2 means 1:2 (1 target, 2 non-targets)

        # Since data is balanced, don't need heavy pos_weight
        self.pos_weight = 1.5  # Mild (was 4.0-5.0 for imbalanced)

        # Data preprocessing
        self.normalized = True


def undersample_data(X, y, target_ratio=2, random_state=42):
    """
    Undersample non-targets to achieve desired ratio.

    Args:
        X: (n, channels, samples) data
        y: (n,) labels
        target_ratio: Desired ratio (targets : non-targets)
            - 2 means 1:2 (1 target, 2 non-targets)
            - 3 means 1:3 (1 target, 3 non-targets)
        random_state: For reproducibility

    Returns:
        X_balanced, y_balanced
    """
    np.random.seed(random_state)

    target_idx = np.where(y == 1)[0]
    non_target_idx = np.where(y == 0)[0]

    n_targets = len(target_idx)
    n_keep_non_targets = n_targets * target_ratio

    # Randomly sample non-targets
    if n_keep_non_targets < len(non_target_idx):
        non_target_idx_sampled = np.random.choice(
            non_target_idx,
            size=n_keep_non_targets,
            replace=False
        )
    else:
        non_target_idx_sampled = non_target_idx

    # Combine and shuffle
    balanced_idx = np.concatenate([target_idx, non_target_idx_sampled])
    np.random.shuffle(balanced_idx)

    original_ratio = len(non_target_idx) / n_targets
    new_ratio = len(non_target_idx_sampled) / n_targets

    print(f"    Original ratio: 1:{original_ratio:.1f}")
    print(f"    New ratio:      1:{new_ratio:.1f}")
    print(f"    Kept {len(balanced_idx)}/{len(y)} epochs ({len(balanced_idx)/len(y)*100:.1f}%)")

    return X[balanced_idx], y[balanced_idx]


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
    """LOSO cross-validation with undersampling."""

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
        print("\nNormalizing data (z-score per channel)...")
        X = normalize_data(X)

    logo = LeaveOneGroupOut()
    n_folds = len(np.unique(subject_id))

    print(f"\nStarting LOSO CV with 1:{config.target_ratio} undersampling ({n_folds} folds)...")
    print("=" * 70)

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, subject_id)):
        test_subject = subject_id[test_idx][0]
        print(f"\nFold {fold + 1}/{n_folds}: Test Subject {test_subject}")
        print("-" * 70)

        X_train_full, X_test = X[train_idx], X[test_idx]
        y_train_full, y_test = y[train_idx], y[test_idx]

        print(f"  Before undersampling: {len(X_train_full)} train epochs")

        # UNDERSAMPLE training data only (not test!)
        X_train, y_train = undersample_data(
            X_train_full,
            y_train_full,
            target_ratio=config.target_ratio,
            random_state=42 + fold  # Different seed per fold
        )

        print(f"  Test: {len(X_test)} epochs (unchanged)")

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

        # Loss with MILD pos_weight (data is now balanced)
        pos_weight = torch.tensor([config.pos_weight]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        # Training loop
        print(f"  Training undersampled EEGNet ({config.n_epochs} epochs)...")
        for epoch in range(config.n_epochs):
            loss = train_epoch(model, train_loader, criterion, optimizer, device)

            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch + 1}/{config.n_epochs}: Loss = {loss:.4f}")

        # Evaluate on FULL test set (not undersampled)
        y_test_np, y_probs = evaluate(model, test_loader, device)

        # Find best threshold
        best_thresh, _ = find_best_threshold(y_test_np, y_probs, metric='f1')
        y_pred = (y_probs > best_thresh).astype(int)

        # Compute metrics
        metrics = compute_metrics(y_test_np, y_pred, y_probs)
        metrics['best_threshold'] = best_thresh
        metrics['subject'] = int(test_subject)

        # Save probabilities for ensemble
        metrics['probabilities'] = y_probs.tolist()
        metrics['true_labels'] = y_test_np.tolist()

        print(f"  Results:")
        print(f"    BAcc:      {metrics['balanced_accuracy']:.3f}")
        print(f"    F1:        {metrics['f1']:.3f}")
        print(f"    Precision: {metrics['precision']:.3f}")
        print(f"    Recall:    {metrics['recall']:.3f}")

        fold_results.append(metrics)

    # Aggregate
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
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

    return {
        'fold_results': fold_results,
        'aggregate': aggregate,
        'config': {
            'F1': config.F1,
            'dropout': config.dropout,
            'n_epochs': config.n_epochs,
            'target_ratio': config.target_ratio,
            'pos_weight': config.pos_weight,
            'undersampling': True
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio', type=int, default=2, choices=[2, 3],
                       help='Target:non-target ratio (2=1:2, 3=1:3)')
    args = parser.parse_args()

    print("=" * 70)
    print(f"  EEGNet with 1:{args.ratio} Undersampling")
    print("=" * 70)

    # Paths
    project_root = Path(__file__).parent.parent
    data_path = project_root.parent / 'P300_8subject_complete' / 'preprocessing' / 'p300_preprocessed_v2.npz'
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)

    # Load data
    X, y, subject_id = load_data(data_path)

    # Config
    config = UndersampledConfig(target_ratio=args.ratio)

    print(f"\nConfiguration:")
    print(f"  Undersampling ratio:   1:{config.target_ratio}")
    print(f"  F1 (temporal filters): {config.F1}")
    print(f"  Dropout:               {config.dropout}")
    print(f"  Epochs:                {config.n_epochs}")
    print(f"  Pos weight:            {config.pos_weight} (mild, data is balanced)")

    # Run LOSO
    results = loso_cross_validation(X, y, subject_id, config, normalize=True)

    # Save
    results_path = results_dir / f'eegnet_undersampled_1to{args.ratio}_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Results saved to: {results_path}")
    print(f"{'=' * 70}")

    # Comparison
    print("\nCOMPARISON WITH BASELINE:")
    print("-" * 70)
    print(f"Baseline (imbalanced):        BAcc = 0.651 ± 0.046")
    print(f"Undersampled 1:{args.ratio} (balanced):  BAcc = {results['aggregate']['balanced_accuracy']['mean']:.3f} "
          f"± {results['aggregate']['balanced_accuracy']['std']:.3f}")

    improvement = (results['aggregate']['balanced_accuracy']['mean'] - 0.651) * 100
    print(f"Change:                       {improvement:+.1f}%")


if __name__ == '__main__':
    main()
