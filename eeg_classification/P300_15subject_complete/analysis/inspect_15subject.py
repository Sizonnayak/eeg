"""
Data Inspection for P300 15-Subject Dataset
============================================
Different structure from 8-subject dataset:
- SBJ01-15 folders (15 subjects)
- Each has S01-S07 sub-sessions
- Train/Test splits pre-defined
- trainData.mat: (8, 350, 1600) = (channels, trials/epochs, time_samples)
- trainEvents.txt: which row/col flashed
- trainTargets.txt: 0=non-target, 1=target
- trainLabels.txt: ground truth letters (for spelling reconstruction)

Goal: Inspect all subjects, check data quality, determine preprocessing needs
"""

import os
import numpy as np
import scipy.io as sio
from pathlib import Path

DATA_DIR = "/Users/siznayak/Documents/others/MTech/Dataset/p300_15subject"

def inspect_subject_session(sbj, session):
    """Inspect one subject's one session."""
    train_path = Path(DATA_DIR) / sbj / session / "Train"
    test_path = Path(DATA_DIR) / sbj / session / "Test"

    result = {
        "subject": sbj,
        "session": session,
        "train": {},
        "test": {}
    }

    # Load training data
    if train_path.exists():
        mat = sio.loadmat(train_path / "trainData.mat")
        X_train = mat["trainData"]  # (8, n_trials, time_samples)

        with open(train_path / "trainTargets.txt") as f:
            targets = np.array([int(line.strip()) for line in f])

        with open(train_path / "trainEvents.txt") as f:
            events = np.array([int(line.strip()) for line in f])

        result["train"] = {
            "shape": X_train.shape,
            "n_channels": X_train.shape[0],
            "n_trials": X_train.shape[1],
            "n_timepoints": X_train.shape[2],
            "n_targets": targets.sum(),
            "n_nontargets": (targets == 0).sum(),
            "target_rate": targets.mean(),
            "min_val": X_train.min(),
            "max_val": X_train.max(),
            "mean": X_train.mean(),
            "std": X_train.std(),
        }

    # Load test data
    if test_path.exists():
        mat_test = sio.loadmat(test_path / "testData.mat")
        X_test = mat_test["testData"]

        with open(test_path / "testTargets.txt") as f:
            targets_test = np.array([int(line.strip()) for line in f])

        result["test"] = {
            "shape": X_test.shape,
            "n_trials": X_test.shape[1],
            "n_targets": targets_test.sum(),
            "target_rate": targets_test.mean(),
        }

    return result


def inspect_all():
    """Inspect all 15 subjects."""
    print("=" * 80)
    print("  P300 15-Subject Dataset Inspection")
    print("=" * 80)

    all_results = []

    for sbj_num in range(1, 16):
        sbj = f"SBJ{sbj_num:02d}"
        sbj_path = Path(DATA_DIR) / sbj

        if not sbj_path.exists():
            continue

        # Count sessions
        sessions = sorted([s.name for s in sbj_path.iterdir() if s.is_dir() and s.name.startswith('S')])

        print(f"\n{sbj} — {len(sessions)} sessions")
        print("-" * 80)

        for session in sessions:
            result = inspect_subject_session(sbj, session)
            all_results.append(result)

            tr = result["train"]
            te = result["test"]

            print(f"  {session}:")
            print(f"    Train: shape={tr['shape']}, "
                  f"trials={tr['n_trials']}, "
                  f"target={tr['n_targets']}/{tr['n_trials']} ({tr['target_rate']:.1%})")
            print(f"           range=[{tr['min_val']:.2f}, {tr['max_val']:.2f}] µV, "
                  f"mean={tr['mean']:.2f}, std={tr['std']:.2f}")

            if te:
                print(f"    Test:  trials={te['n_trials']}, "
                      f"target={te['n_targets']}/{te['n_trials']} ({te['target_rate']:.1%})")

    # Summary statistics
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)

    total_train_trials = sum(r["train"]["n_trials"] for r in all_results)
    total_train_targets = sum(r["train"]["n_targets"] for r in all_results)
    total_test_trials = sum(r["test"]["n_trials"] for r in all_results if r["test"])
    total_test_targets = sum(r["test"]["n_targets"] for r in all_results if r["test"])

    print(f"Total subjects: 15")
    print(f"Total sessions: {len(all_results)}")
    print(f"Total train trials: {total_train_trials}")
    print(f"Total train targets: {total_train_targets} ({total_train_targets/total_train_trials:.1%})")
    print(f"Total test trials: {total_test_trials}")
    print(f"Total test targets: {total_test_targets} ({total_test_targets/total_test_trials:.1%})")

    # Data format
    first_result = all_results[0]["train"]
    print(f"\nData format:")
    print(f"  Shape: {first_result['shape']} = (channels, trials, timepoints)")
    print(f"  Channels: {first_result['n_channels']}")
    print(f"  Timepoints: {first_result['n_timepoints']}")
    print(f"  Note: This is CHANNELS-FIRST format (different from 8-subject dataset)")

    return all_results


if __name__ == "__main__":
    results = inspect_all()
    print("\nDone!")
