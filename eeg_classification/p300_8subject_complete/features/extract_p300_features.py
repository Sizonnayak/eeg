"""
P300 Feature Extraction
========================
Feature recipe (fixed across all subjects):
  1. Crop P300 window 200-500 ms  → epoch[:, 100:175]  shape (8, 75)
  2. Temporal compression via jumping mean into 15 bins
     75 / 15 = 5 samples/bin      → shape (8, 15)
  3. Flatten                       → 120-dim feature vector

Input:
    p300_preprocessed.npz  (from preprocessing/p300_preprocess_for_ml.py)
        X          : (33473, 8, 250)
        y          : (33473,)
        subject_id : (33473,)

Output:
    p300_features.npz
        X_features  : (33473, 120)
        y           : (33473,)
        subject_id  : (33473,)
        feature_names: (120,)  string labels  e.g. "Fz_bin00"

Usage:
    python features/extract_p300_features.py
"""

import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PREPROCESSED_PATH = "/Users/siznayak/Documents/others/MTech/EEG_Classification/preprocessing/p300_preprocessed.npz"
SAVE_PATH         = "/Users/siznayak/Documents/others/MTech/EEG_Classification/features/p300_features.npz"

# ---------------------------------------------------------------------------
# Feature constants
# ---------------------------------------------------------------------------
FS          = 250
PRE_MS      = 200       # pre-stimulus duration stored in epochs

# P300 window: 200-500 ms post-stimulus
P300_START_MS  = 200
P300_END_MS    = 500
# Convert to sample indices within the epoch (-200 to +800 ms, 250 samples)
# sample index = (time_ms + PRE_MS) * FS / 1000
P300_START_IDX = int((P300_START_MS + PRE_MS) * FS / 1000)   # (200+200)*250/1000 = 100
P300_END_IDX   = int((P300_END_MS   + PRE_MS) * FS / 1000)   # (500+200)*250/1000 = 175
# → P300 window = epoch[:, 100:175]  → 75 samples

N_BINS   = 15        # temporal compression bins
BIN_SIZE = (P300_END_IDX - P300_START_IDX) // N_BINS   # 75 // 15 = 5 samples per bin

CH_NAMES = ["Fz", "Cz", "P3", "Pz", "P4", "PO7", "PO8", "Oz"]

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features_single(epoch):
    """
    Extract 120-dim feature vector from one epoch.

    Args:
        epoch : np.ndarray shape (8, 250)  — channels × time-points

    Returns:
        feat  : np.ndarray shape (120,)
    """
    # Step 1: Crop P300 window  (8, 75)
    epoch_p300 = epoch[:, P300_START_IDX:P300_END_IDX]

    # Step 2: Temporal compression  →  (8, 15, 5).mean(axis=2)  →  (8, 15)
    epoch_binned = epoch_p300.reshape(len(CH_NAMES), N_BINS, BIN_SIZE).mean(axis=2)

    # Step 3: Flatten  →  (120,)
    feat = epoch_binned.reshape(-1)
    return feat.astype(np.float32)


def build_feature_names():
    """Return list of 120 human-readable feature names: 'Fz_bin00', ..., 'Oz_bin14'."""
    names = []
    for ch in CH_NAMES:
        for b in range(N_BINS):
            # Approximate centre time of this bin in ms
            t_start = P300_START_MS + b * BIN_SIZE * 1000 / FS
            t_end   = t_start + BIN_SIZE * 1000 / FS
            names.append(f"{ch}_{int(t_start)}-{int(t_end)}ms")
    return np.array(names)


def extract_all(X):
    """
    Vectorised extraction over all epochs.

    Args:
        X : (n_epochs, 8, 250)

    Returns:
        X_features : (n_epochs, 120)
    """
    # Step 1: Crop (n_epochs, 8, 75)
    X_p300 = X[:, :, P300_START_IDX:P300_END_IDX]

    # Step 2: Bin  (n_epochs, 8, 15)
    X_binned = X_p300.reshape(X.shape[0], len(CH_NAMES), N_BINS, BIN_SIZE).mean(axis=3)

    # Step 3: Flatten  (n_epochs, 120)
    X_features = X_binned.reshape(X.shape[0], -1)

    return X_features.astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  P300 Feature Extraction")
    print("  Recipe: 200-500ms → 8×75 → bin(15) → 8×15 → flatten → 120-dim")
    print("=" * 70)

    # Load preprocessed data
    if not os.path.exists(PREPROCESSED_PATH):
        raise FileNotFoundError(
            f"Preprocessed file not found: {PREPROCESSED_PATH}\n"
            "Run preprocessing/p300_preprocess_for_ml.py first."
        )

    print(f"\n  Loading: {PREPROCESSED_PATH}")
    data = np.load(PREPROCESSED_PATH, allow_pickle=True)
    X          = data["X"]           # (n_epochs, 8, 250)
    y          = data["y"]           # (n_epochs,)
    subject_id = data["subject_id"]  # (n_epochs,)

    print(f"  X shape (raw epochs) : {X.shape}   (n_epochs, channels, time_points)")
    print(f"  y shape              : {y.shape}")
    print(f"  Subjects             : {np.unique(subject_id)}")

    # Show per-subject epoch counts
    print(f"\n  Per-subject epoch counts:")
    print(f"  {'Subject':<10} {'Total':>8} {'Target':>8} {'Non-target':>12}")
    print(f"  {'-'*40}")
    for sid in np.unique(subject_id):
        mask    = subject_id == sid
        n_total = mask.sum()
        n_tgt   = y[mask].sum()
        n_non   = (y[mask] == 0).sum()
        print(f"  S{sid:02d}       {n_total:>8}   {n_tgt:>8}   {n_non:>10}")
    print(f"  {'-'*40}")
    print(f"  {'Total':<10} {len(y):>8}   {y.sum():>8}   {(y==0).sum():>10}")

    # Extract features
    print(f"\n  Extracting features...")
    print(f"  P300 window  : {P300_START_MS}-{P300_END_MS} ms → indices [{P300_START_IDX}:{P300_END_IDX}] → 75 samples")
    print(f"  Bins         : {N_BINS} bins × {BIN_SIZE} samples each")
    print(f"  Feature size : 8 channels × {N_BINS} bins = {len(CH_NAMES) * N_BINS} dims")

    X_features = extract_all(X)
    feature_names = build_feature_names()

    print(f"\n  X_features shape : {X_features.shape}   (n_epochs, 120)")
    print(f"  Feature names [0] : {feature_names[0]}")
    print(f"  Feature names[-1] : {feature_names[-1]}")
    print(f"  Min / Max values  : {X_features.min():.4f} / {X_features.max():.4f}")

    # Save
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    np.savez_compressed(
        SAVE_PATH,
        X_features    = X_features,
        y             = y,
        subject_id    = subject_id,
        feature_names = feature_names,
    )

    print(f"\n  Saved → {SAVE_PATH}")
    print(f"  Ready for LDA / SVC classification.\n")


if __name__ == "__main__":
    main()
