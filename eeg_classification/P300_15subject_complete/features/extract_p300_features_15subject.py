"""
P300 Feature Extraction for 15-Subject Dataset
===============================================
Same approach as 8-subject dataset:
  - Crop P300 window (200-500ms post-stimulus)
  - Bin averaging (15 bins × 5 samples each at 250 Hz)
  - Flatten to 120-dim features

Input:  preprocessing/p300_preprocessed_15subject.npz
        X: (35700, 8, 1600) — filtered + CAR epochs

Output: features/p300_features_15subject.npz
        X_epochs: (35700, 8, 1600) — for Riemannian methods
        X_features: (35700, 120) — binned features
        y, subject_id, session_id

Challenge: Data has 1600 timepoints but we need to identify P300 window
  - Assuming: -200ms to +6200ms at 250 Hz = 1600 samples
  - P300 window: 200-500ms post-stimulus
  - At -200ms start → P300 is at samples [100:175] (same as 8-subject!)

  BUT if epoch window is different, need to calculate correctly.
  Let me verify with you if needed.

For now, using same indices as 8-subject: samples 100:175
"""

import os
import numpy as np

# Paths
PREPROCESSED_PATH = "/Users/siznayak/Documents/others/MTech/EEG_Classification/P300_15subject_complete/preprocessing/p300_preprocessed_15subject.npz"
SAVE_PATH = "/Users/siznayak/Documents/others/MTech/EEG_Classification/P300_15subject_complete/features/p300_features_15subject.npz"

# Feature constants
FS = 250
PRE_MS = 200  # Assuming same as 8-subject: epoch starts at -200ms
P300_START_MS = 200
P300_END_MS = 500

# Calculate indices
# If epoch starts at -200ms:
# t=0ms (stimulus) is at sample 200ms * 250Hz / 1000 = 50
# t=200ms is at sample (200+200)ms * 250/1000 = 100
# t=500ms is at sample (500+200)ms * 250/1000 = 175

P300_START_IDX = int((P300_START_MS + PRE_MS) * FS / 1000)  # 100
P300_END_IDX   = int((P300_END_MS + PRE_MS) * FS / 1000)    # 175

N_BINS = 15
BIN_SIZE = (P300_END_IDX - P300_START_IDX) // N_BINS  # 5

CH_NAMES = ["C3", "Cz", "C4", "CPz", "P3", "Pz", "P4", "POz"]


def extract_features_from_epochs(X):
    """
    Crop P300 window → 15 bins → flatten.

    Args:
        X: (n, 8, 1600)
    Returns:
        (n, 120)
    """
    n = X.shape[0]

    # Crop P300 window
    X_p300 = X[:, :, P300_START_IDX:P300_END_IDX]  # (n, 8, 75)

    # Bin averaging: reshape to (n, 8, 15, 5) and average over last axis
    X_binned = X_p300.reshape(n, len(CH_NAMES), N_BINS, BIN_SIZE).mean(axis=3)  # (n, 8, 15)

    # Flatten
    return X_binned.reshape(n, -1).astype(np.float32)  # (n, 120)


def build_feature_names():
    """Build feature names for interpretability."""
    names = []
    for ch in CH_NAMES:
        for b in range(N_BINS):
            t_start = P300_START_MS + b * BIN_SIZE * 1000 / FS
            t_end = t_start + BIN_SIZE * 1000 / FS
            names.append(f"{ch}_{int(t_start)}-{int(t_end)}ms")
    return np.array(names)


def main():
    print("=" * 80)
    print("  P300 Feature Extraction — 15-Subject Dataset")
    print(f"  Recipe: {P300_START_MS}-{P300_END_MS}ms → 8×{P300_END_IDX-P300_START_IDX} "
          f"→ bin({N_BINS}) → 8×15 → flatten → 120-dim")
    print("=" * 80)

    if not os.path.exists(PREPROCESSED_PATH):
        raise FileNotFoundError(f"Preprocessed file not found: {PREPROCESSED_PATH}")

    print(f"\nLoading: {PREPROCESSED_PATH}")
    data = np.load(PREPROCESSED_PATH, allow_pickle=True)

    X = data["X"]  # (35700, 8, 1600)
    y = data["y"]
    subject_id = data["subject_id"]
    session_id = data["session_id"]

    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Target rate: {y.mean()*100:.1f}%")
    print(f"  Subjects: {np.unique(subject_id)}")
    print(f"  Sessions: {np.unique(session_id)}")

    # Per-subject counts
    print(f"\nPer-subject epoch counts:")
    for sid in np.unique(subject_id):
        mask = subject_id == sid
        n = mask.sum()
        t = y[mask].sum()
        print(f"  SBJ{sid:02d}: {n:5d} epochs  ({t:4d} target, {n-t:5d} non-target)")

    # Extract features
    print(f"\nExtracting 120-dim binned features...")
    print(f"  P300 window: samples [{P300_START_IDX}:{P300_END_IDX}] "
          f"({P300_START_MS}-{P300_END_MS}ms)")

    X_features = extract_features_from_epochs(X)
    feature_names = build_feature_names()

    print(f"  X_features shape: {X_features.shape}")
    print(f"  Feature range: [{X_features.min():.3f}, {X_features.max():.3f}]")

    # Save
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    np.savez_compressed(
        SAVE_PATH,
        X_epochs=X,  # (35700, 8, 1600) — for Riemannian methods
        X_features=X_features,  # (35700, 120)
        y=y,
        subject_id=subject_id,
        session_id=session_id,
        feature_names=feature_names,
    )

    print(f"\nSaved → {SAVE_PATH}")
    print(f"Keys: X_epochs (n,8,1600), X_features (n,120), y, subject_id, session_id\n")


if __name__ == "__main__":
    main()
