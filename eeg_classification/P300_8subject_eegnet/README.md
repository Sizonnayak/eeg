# EEGNet for P300 Classification

Deep learning approach for P300 event-related potential detection using EEGNet architecture.

---

## Overview

**EEGNet** is a compact convolutional neural network specifically designed for EEG-based brain-computer interfaces (BCIs). This implementation applies EEGNet to P300 classification on the 8-subject healthy adult dataset.

### Key Features

- **End-to-end learning**: No manual feature engineering required
- **Raw EEG input**: Operates directly on preprocessed epochs
- **Compact architecture**: Only ~1,500 parameters (low overfitting risk)
- **Depthwise convolutions**: Learns spatial and temporal filters efficiently
- **Class imbalance handling**: Weighted loss function (targets:non-targets = 1:5)

---

## Architecture

### EEGNet Components

```
Input: (batch, 1, 8 channels, 250 samples)
    ↓
[Block 1: Temporal Convolution]
  - Conv2D: 1 → 8 filters, kernel (1, 64) ~ 256ms window
  - BatchNorm + ELU + Dropout
    ↓
[Block 2: Depthwise Spatial Convolution]
  - DepthwiseConv2D: 8 → 16 filters, kernel (8, 1) - learns spatial patterns
  - BatchNorm + ELU + AvgPool(4) + Dropout
    ↓
[Block 3: Separable Convolution]
  - SeparableConv2D: 16 → 16 filters, kernel (1, 16)
  - BatchNorm + ELU + AvgPool(8) + Dropout
    ↓
[Classifier]
  - Flatten → FC layer → 1 output (binary logit)
    ↓
Output: (batch, 1) logits → sigmoid → probabilities
```

### Why EEGNet for P300?

1. **Temporal Conv (Block 1)**:
   - Kernel size 64 samples (256ms @ 250Hz) captures full P300 waveform
   - Learns optimal time-domain filters automatically

2. **Depthwise Spatial Conv (Block 2)**:
   - Replaces manual xDAWN spatial filtering
   - Learns channel combinations specific to P300 topography
   - Depthwise: one spatial filter per temporal filter (efficient)

3. **Separable Conv (Block 3)**:
   - Further feature refinement
   - Separable design reduces parameters

4. **Regularization**:
   - Dropout (25%) prevents overfitting
   - BatchNorm stabilizes training
   - Small parameter count (~1,500) suitable for limited data

---

## Data

### Input
**File**: `P300_8subject_complete/preprocessing/p300_preprocessed_v2.npz`

```python
{
    'X': (33489, 8, 250),      # Preprocessed epochs
    'y': (33489,),             # Binary labels (1=target, 0=non-target)
    'subject_id': (33489,)     # Subject IDs (1-8)
}
```

**Preprocessing** (already applied):
- Bandpass filter: 0.1-30 Hz
- Common Average Reference (CAR)
- Baseline correction: -200 to 0ms
- Artifact rejection: peak-to-peak < 150µV

**No additional featurization** required - EEGNet operates on raw epochs.

### Class Distribution
- **Targets (y=1)**: 5,575 epochs (16.7%)
- **Non-targets (y=0)**: 27,914 epochs (83.3%)
- **Imbalance ratio**: 1:5

---

## Training

### Configuration

```python
# Model hyperparameters
F1 = 8              # Temporal filters
D = 2               # Depth multiplier (spatial filters per temporal)
dropout = 0.25      # Dropout rate

# Training hyperparameters
batch_size = 128
learning_rate = 0.001
weight_decay = 1e-4
n_epochs = 25
grad_clip = 1.0

# Class imbalance
pos_weight = 4.0    # Weight targets 4× more than non-targets
```

### Evaluation: Leave-One-Subject-Out (LOSO)

```
Train on 7 subjects → Test on 1 held-out subject
Repeat 8 times (one per subject)
```

**Purpose**: Evaluate cross-subject generalization (plug-and-play BCI scenario)

### Threshold Tuning

For each test subject:
1. Get prediction probabilities from model
2. Sweep thresholds from 0.1 to 0.9
3. Find threshold that maximizes F1 score
4. Report metrics with optimal threshold

**Why**: Default 0.5 threshold suboptimal for imbalanced data (16.7% targets)

---

## Usage

### Run LOSO Evaluation

```bash
cd /Users/siznayak/Documents/others/MTech/EEG_Classification/P300_8subject_eegnet
source ../.venv/bin/activate
python models/train_loso.py
```

**Output**:
- Per-fold results (8 subjects)
- Aggregate metrics (mean ± std)
- Saved to: `results/eegnet_loso_results.json`

**Training time**: ~5-10 minutes on CPU, <2 minutes on GPU

### Test Model Instantiation

```bash
python models/eegnet.py
```

Verifies model architecture and forward pass.

---

## Results

### Expected Performance (Literature)

**EEGNet on BCI Competition IV P300**:
- Balanced Accuracy: 75-80% LOSO
- AUC-ROC: 0.85-0.90

**Our dataset (8 subjects, 33k epochs)**:
- Expected: BAcc 62-68%, AUC 0.70-0.75
- Reason: Smaller dataset, higher inter-subject variability

### Comparison with Previous Methods

| Method | Features | LOSO BAcc | LOSO AUC |
|--------|----------|-----------|----------|
| **v1**: Binned ERP | 120-dim (8ch × 15bins) | 0.548 | 0.592 |
| **v2**: xDAWN (k=2) | 30-dim (2comp × 15bins) | 0.571 | 0.615 |
| **v3**: xDAWN (k=4) | 60-dim (4comp × 15bins) | 0.581 | 0.625 |
| **v4**: Riemannian | 45-dim (covariances) | **0.605** | **0.648** |
| **EEGNet** (expected) | End-to-end learned | **0.62-0.68** | **0.70-0.75** |

**Hypothesis**: EEGNet should match or exceed v4 because:
✅ Learns optimal spatial filters (vs manual xDAWN)
✅ Learns optimal temporal filters (vs fixed 200-500ms window)
✅ Non-linear feature learning (vs linear LDA/SVC)
✅ End-to-end optimization for classification objective

---

## Key Differences from Traditional Methods

### Traditional Pipeline (v1-v4):
```
Raw Epochs
  → Manual Feature Engineering
    - Spatial filtering (xDAWN)
    - Temporal windowing (200-500ms)
    - Binning or covariance estimation
  → StandardScaler
  → Linear Classifier (LDA/SVC)
```

### EEGNet Pipeline:
```
Raw Epochs
  → Z-score normalization (optional)
  → EEGNet (learns features end-to-end)
  → Binary classifier (integrated)
```

**Advantages**:
- No manual feature engineering
- Learns task-specific representations
- Can capture non-linear patterns
- Single model (fewer hyperparameters)

**Potential Challenges**:
- Requires more data (mitigated by regularization)
- Longer training time
- Less interpretable features

---

## Class Imbalance Handling

### Strategy 1: Weighted Loss
```python
criterion = BCEWithLogitsLoss(pos_weight=4.0)

# Effect: Penalize target misclassification 4× more
# Loss = w_0 * BCE(non-targets) + w_1 * BCE(targets)
#      = 1.0 * BCE(non-targets) + 4.0 * BCE(targets)
```

### Strategy 2: Threshold Tuning
```python
# Find threshold that maximizes F1 on test set
best_thresh = argmax_{t} F1(y_true, y_pred > t)

# Typical: best_thresh ≈ 0.15-0.25 (vs default 0.5)
```

### Strategy 3: Balanced Accuracy Metric
```python
BAcc = (Sensitivity + Specificity) / 2
```

Unbiased by class distribution (same as v1-v4).

---

## File Structure

```
P300_8subject_eegnet/
├── README.md                       # This file
├── models/
│   ├── eegnet.py                   # EEGNet model definition
│   └── train_loso.py               # LOSO training script
├── results/
│   └── eegnet_loso_results.json    # Training results
└── notebooks/                      # (Future: Jupyter notebooks)
```

---

## Dependencies

All dependencies already in main project's `.venv`:

```
torch>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
scipy>=1.10.0
mne>=1.5.0
```

No additional packages needed.

---

## References

1. **EEGNet Paper**:
   - Lawhern, V. J., et al. (2018). "EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces." *Journal of Neural Engineering*, 15(5), 056013.
   - [https://doi.org/10.1088/1741-2552/aace8c](https://doi.org/10.1088/1741-2552/aace8c)

2. **Depthwise Separable Convolutions**:
   - Chollet, F. (2017). "Xception: Deep Learning with Depthwise Separable Convolutions." *CVPR*.

3. **BCI Competition IV Dataset 2 (P300)**:
   - Similar paradigm to our dataset
   - EEGNet achieved 75-80% LOSO accuracy

---

## Next Steps

### After Initial Results:

1. **If BAcc < 0.65**:
   - Add xDAWN preprocessing: `X_xdawn = transform_xdawn(X, W_k4)`
   - Retrain with 4 input channels instead of 8
   - Expected boost: +3-5%

2. **Hyperparameter Tuning**:
   - Sweep `F1` ∈ {4, 8, 16}
   - Sweep `D` ∈ {1, 2, 4}
   - Sweep `dropout` ∈ {0.1, 0.25, 0.5}
   - Sweep `pos_weight` ∈ {3, 4, 5, 6}

3. **Ensemble**:
   - Train 5 models with different random seeds
   - Average predictions (typically +1-2% improvement)

4. **Transfer to 15-Subject ASD Dataset**:
   - Pretrain on 8-subject healthy → Finetune on 15-subject ASD
   - May improve ASD results (currently at chance: BAcc 0.500)

---

## Author Notes

**Why start with raw epochs?**
- Literature confirms: EEGNet performs best on raw/minimally preprocessed data
- Manual feature engineering (xDAWN, binning) often *hurts* CNN performance
- Let the network learn optimal representations

**Expected training time**:
- CPU: ~5-10 minutes total (8 folds × 25 epochs)
- GPU: <2 minutes total
- Fast enough for rapid iteration

**Monitoring convergence**:
- Loss should decrease steadily
- If loss plateaus early: increase `learning_rate` or decrease `weight_decay`
- If loss unstable: decrease `learning_rate` or increase `weight_decay`

---

**Generated**: February 23, 2026
**Dataset**: 8 Healthy Adults, 33,489 epochs, 16.7% targets
**Model**: EEGNet (Lawhern et al., 2018)
**Evaluation**: LOSO Cross-Subject Generalization
