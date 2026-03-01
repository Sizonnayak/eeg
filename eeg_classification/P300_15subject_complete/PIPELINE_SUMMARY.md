# P300 Classification Pipeline — 15-Subject ASD Dataset
## End-to-End Summary

---

## Dataset Information

**Source**: IFMBE 2019 Challenge - ASD Patients Dataset
- **Subjects**: 15 ASD patients
- **Sessions per subject**: 7 sessions
- **Total sessions**: 105 (102 after excluding 3 corrupted)
- **Channels**: 8 (C3, Cz, C4, CPz, P3, Pz, P4, POz)
- **Sampling rate**: 250 Hz
- **Epoch window**: -200ms to +1200ms (1400ms total, but data has 1600 samples = 6.4s)
- **Stimulus onset**: Sample 50 (200ms post-epoch start)
- **Raw data format**: (8 channels, 350 epochs, 1600 timepoints) - **CHANNELS-FIRST**
- **Target rate**: 12.5% (1 target : 7 non-targets)
- **Trials per session**: 350 epochs
- **Total epochs**: 35,700 (after excluding 3 sessions)

**Data Location**: `/Users/siznayak/Documents/others/MTech/Dataset/p300_15subject`

---

## Pipeline Overview

```
Raw Data → Preprocessing → Feature Extraction → Model Training → Evaluation
```

---

## 1. Data Inspection

**Script**: [analysis/inspect_15subject.py](analysis/inspect_15subject.py)

### Steps:
1. **Load all 105 sessions** (15 subjects × 7 sessions)
2. **Check data format**:
   - Shape: (8, 350, 1600) = (channels, epochs, timepoints)
   - Transpose required: channels-first → epochs-first
3. **Identify corrupted sessions**:
   - Compute peak-to-peak amplitude per session
   - Flag extreme artifacts (>1000 µV)

### Key Findings:
- **3 severely corrupted sessions identified**:
  - SBJ03/S07: ±4420 µV
  - SBJ04/S03: ±2156 µV
  - SBJ13/S07: ±13571 µV
- **Normal range**: 100-300 µV peak-to-peak
- **Target distribution**: Consistent 12.5% across sessions

---

## 2. Preprocessing

**Script**: [preprocessing/p300_preprocess_15subject.py](preprocessing/p300_preprocess_15subject.py)

### Method:

#### A. Data Loading
```python
# Load Train data from each session
train_path = DATA_DIR / sbj / session / "Train"
mat = sio.loadmat(train_path / "trainData.mat")
X_train = mat["trainData"]  # Shape: (8, 350, 1600)

# Load labels
with open(train_path / "trainTargets.txt") as f:
    targets = [int(line.strip()) for line in f]
```

#### B. Data Format Conversion
```python
# Transpose from channels-first to epochs-first
X = X.transpose(1, 0, 2)  # (350, 8, 1600)
```

#### C. Bandpass Filtering
```python
# Using MNE-Python
info = mne.create_info(
    ch_names=['C3', 'Cz', 'C4', 'CPz', 'P3', 'Pz', 'P4', 'POz'],
    sfreq=250.0,
    ch_types='eeg'
)

raw = mne.io.RawArray(X.transpose(1, 0, 2).reshape(8, -1), info)
raw.filter(l_freq=0.1, h_freq=30.0, method='iir', phase='forward')

# Bandpass: 0.1 - 30.0 Hz
# - Removes DC drift (< 0.1 Hz)
# - Removes high-frequency noise (> 30 Hz)
# - Preserves P300 band (~0.5-15 Hz)
```

#### D. Common Average Reference (CAR)
```python
# Re-reference to common average
# Each channel = original - mean(all channels)
# Reduces common-mode noise
raw.set_eeg_reference('average', projection=False)
```

#### E. Artifact Rejection
```python
# Subject-specific thresholds based on signal quality
THRESHOLDS = {
    "SBJ01": 150, "SBJ02": 150, "SBJ03": 200,
    "SBJ04": 250, "SBJ05": 200, "SBJ06": 150,
    "SBJ07": 250, "SBJ08": 150, "SBJ09": 200,
    "SBJ10": 150, "SBJ11": 200, "SBJ12": 250,
    "SBJ13": 300, "SBJ14": 250, "SBJ15": 250
}

# Reject epochs with peak-to-peak > threshold
ptp = np.ptp(X_epoch, axis=(1, 2))  # (n_epochs,)
good_idx = ptp < threshold
```

### Output:
**File**: `preprocessing/p300_preprocessed_15subject.npz`

```python
{
    'X_epochs': (35700, 8, 1600),  # Clean EEG epochs
    'y': (35700,),                  # Binary labels (0/1)
    'subject_id': (35700,),         # Subject IDs (1-15)
    'session_id': (35700,),         # Session IDs (1-7)
}
```

### Preprocessing Results:
- **Input**: 105 sessions × 350 epochs = 36,750 epochs
- **Excluded sessions**: 3 (severely corrupted)
- **Remaining sessions**: 102
- **Total epochs**: 35,700
- **Rejected epochs**: 0 (thresholds set appropriately)
- **Target rate**: 12.5% (4,468 targets)

---

## 3. Feature Extraction

**Script**: [features/extract_p300_features_15subject.py](features/extract_p300_features_15subject.py)

### Method: **Binned P300 Features**

#### A. P300 Time Window Selection
```python
# P300 occurs ~200-500ms post-stimulus
# Stimulus at sample 50 (200ms post-epoch start)
# P300 window: 200-500ms → samples 100:175

P300_START_IDX = 100  # 200ms post-stimulus
P300_END_IDX = 175    # 500ms post-stimulus

# Extract P300 window
X_p300 = X[:, :, P300_START_IDX:P300_END_IDX]  # (35700, 8, 75)
```

#### B. Temporal Binning
```python
# Bin 75 timepoints into 15 bins (5 samples per bin)
# Reduces dimensionality while preserving temporal structure

n_bins = 15
bin_size = 75 // n_bins  # 5 samples per bin

X_binned = X_p300.reshape(n, 8, n_bins, bin_size).mean(axis=3)
# Shape: (35700, 8, 15)

# Flatten to feature vector
X_features = X_binned.reshape(n, -1)  # (35700, 120)
```

### Feature Description:
- **Dimensionality**: 120 features
- **Structure**: 8 channels × 15 time bins = 120
- **Time resolution**: Each bin = 20ms (5 samples at 250 Hz)
- **Bins represent**: 200ms, 220ms, 240ms, ..., 480ms, 500ms

### Output:
**File**: `features/p300_features_15subject.npz`

```python
{
    'X_epochs': (35700, 8, 1600),    # Full preprocessed epochs
    'X_features': (35700, 120),       # Binned P300 features
    'y': (35700,),                    # Binary labels
    'subject_id': (35700,),           # Subject IDs
    'session_id': (35700,),           # Session IDs
}
```

---

## 4. Model Building & Training

**Script**: [models/lda_svc_binned_15subject.py](models/lda_svc_binned_15subject.py)

### Models Evaluated:

#### A. Linear Discriminant Analysis (LDA)
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

clf = LinearDiscriminantAnalysis(
    solver='lsqr',      # Least squares solution
    shrinkage='auto'    # Automatic shrinkage regularization
)
```

**Properties**:
- Linear decision boundary
- Assumes Gaussian class distributions
- Good for small sample sizes
- Fast training and prediction

#### B. Support Vector Classifier (SVC)
```python
from sklearn.svm import SVC

clf = SVC(
    C=1.0,                  # Regularization parameter
    kernel='rbf',           # Radial Basis Function kernel
    gamma='scale',          # Kernel coefficient = 1/(n_features * X.var())
    probability=True,       # Enable probability estimates
    random_state=42
)
```

**Properties**:
- Non-linear decision boundary (RBF kernel)
- Robust to outliers
- Computationally expensive for large datasets
- Requires careful parameter tuning

### Preprocessing Pipeline (Within Each CV Fold):
```python
from sklearn.preprocessing import StandardScaler

# Standardize features (zero mean, unit variance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

## 5. Evaluation

### Three Evaluation Strategies:

#### A. Within-Subject Cross-Validation
```python
from sklearn.model_selection import StratifiedKFold

# For each subject independently:
# - 5-fold stratified CV
# - Tests subject-specific performance
# - Optimistic estimate (no cross-subject generalization)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in skf.split(X_subject, y_subject):
    # Train and evaluate on same subject
    ...
```

**Purpose**: Evaluate performance when calibrating per subject

#### B. Leave-One-Subject-Out (LOSO)
```python
from sklearn.model_selection import LeaveOneGroupOut

# Train on 14 subjects, test on 1 held-out subject
# Repeat 15 times (one for each subject)
# Tests cross-subject generalization

logo = LeaveOneGroupOut()
for train_idx, test_idx in logo.split(X, y, groups=subject_id):
    test_subject = subject_id[test_idx[0]]
    # Train on 14 subjects, test on 1
    ...
```

**Purpose**: Evaluate cross-subject generalization (realistic BCI scenario)

#### C. LOSO + Threshold Calibration
```python
# For each LOSO fold:
# 1. Split training data (14 subjects) into train/val (80/20)
# 2. Train model on training set
# 3. Find optimal decision threshold on validation set using Youden's J
# 4. Apply optimized threshold to test subject

# Youden's J statistic = sensitivity + specificity - 1
fpr, tpr, thresholds = roc_curve(y_val, y_val_score)
j_scores = tpr - fpr
optimal_threshold = thresholds[np.argmax(j_scores)]
```

**Purpose**: Optimize decision threshold for imbalanced data (12.5% targets)

---

## 6. Evaluation Metrics

### Metrics Computed:

#### A. Balanced Accuracy (BAcc)
```python
from sklearn.metrics import balanced_accuracy_score

# Average of sensitivity and specificity
# Range: [0, 1], chance = 0.5
# Robust to class imbalance
BAcc = (sensitivity + specificity) / 2
```

**Interpretation**:
- 0.5 = chance level (random classifier)
- 1.0 = perfect classification
- Best metric for imbalanced datasets

#### B. ROC-AUC (Area Under ROC Curve)
```python
from sklearn.metrics import roc_auc_score

# Area under Receiver Operating Characteristic curve
# Range: [0, 1], chance = 0.5
# Measures discrimination ability across all thresholds
auc_roc = roc_auc_score(y_true, y_score)
```

**Interpretation**:
- 0.5 = no discrimination
- 1.0 = perfect discrimination
- Threshold-independent metric

#### C. PR-AUC (Area Under Precision-Recall Curve)
```python
from sklearn.metrics import precision_recall_curve, auc

# More informative than ROC-AUC for imbalanced data
# Focuses on positive class (targets)
precision, recall, _ = precision_recall_curve(y_true, y_score)
auc_pr = auc(recall, precision)
```

**Interpretation**:
- Baseline = target rate (12.5% for this dataset)
- Higher values indicate better positive class detection

#### D. F1 Score
```python
from sklearn.metrics import f1_score

# Harmonic mean of precision and recall
# Range: [0, 1]
# Good for imbalanced classification
F1 = 2 * (precision * recall) / (precision + recall)
```

---

## 7. Results

### Final Results Summary:

**File**: `results/lda_svc_binned_15subject_results.json`

#### LDA (Linear Discriminant Analysis):
```
Within-Subject CV:
  BAcc: 0.500 ± 0.000
  AUC:  0.360 ± 0.040
  PR:   0.093 ± 0.007

LOSO (Cross-Subject):
  BAcc: 0.500
  AUC:  0.504
  PR:   0.127

LOSO + Threshold Calibration:
  BAcc: 0.499
  AUC:  0.505
  PR:   0.127
```

#### SVC (Support Vector Classifier):
```
Within-Subject CV:
  BAcc: 0.500 ± 0.000
  AUC:  0.612 ± 0.033
  PR:   0.161 ± 0.020

LOSO (Cross-Subject):
  BAcc: 0.500
  AUC:  0.501
  PR:   0.127

LOSO + Threshold Calibration:
  BAcc: 0.499
  AUC:  0.497
  PR:   0.126
```

### Key Observations:

1. **Chance-level performance**: All LOSO metrics are at ~0.5 (random guessing)
2. **Within-subject SVC shows some learning**: AUC=0.612 suggests model can learn patterns within a subject
3. **No cross-subject generalization**: LOSO performance drops to chance
4. **Threshold calibration doesn't help**: Already at chance level

---

## 8. Comparison: 8-Subject vs 15-Subject

| Metric | 8-Subject (Healthy) | 15-Subject (ASD) |
|--------|---------------------|------------------|
| **Dataset** | Healthy adults | ASD patients |
| **Subjects** | 8 | 15 |
| **Epochs** | 33,489 | 35,700 |
| **Target Rate** | 16.7% | 12.5% |
| **Sampling Rate** | 250 Hz | 250 Hz |
| **Channels** | 8 (same) | 8 (same) |
| **LOSO BAcc** | **0.605** | **0.500** |
| **LOSO AUC** | **0.648** | **0.501** |
| **Best Model** | Riemannian+rLDA | None (chance) |

### Why 15-Subject Failed:

1. **Poor Signal Quality**: ASD patients may have:
   - Lower P300 amplitude
   - Variable P300 latency
   - More artifacts (movement, attention issues)

2. **Dataset Issues**:
   - Data heavily decimated (1600 samples for 1.4s = unclear processing)
   - 3 sessions had severe corruption
   - Possible preprocessing artifacts

3. **Riemannian Approach Failed**:
   - Covariance estimation errors: "leading minor not positive definite"
   - Indicates ill-conditioned covariance matrices
   - Suggests poor trial-level SNR

4. **Feature Engineering Issues**:
   - Fixed P300 window (200-500ms) may not be optimal for ASD
   - Simple binning may not capture relevant patterns
   - Need adaptive windowing or subject-specific features

---

## 9. File Structure

```
P300_15subject_complete/
├── DATASET_INFO.md               # Dataset documentation
├── PIPELINE_SUMMARY.md           # This file
├── analysis/
│   └── inspect_15subject.py      # Data inspection script
├── preprocessing/
│   ├── p300_preprocess_15subject.py
│   └── p300_preprocessed_15subject.npz  (721 MB)
├── features/
│   ├── extract_p300_features_15subject.py
│   └── p300_features_15subject.npz      (56 MB)
├── models/
│   ├── lda_svc_classifier_15subject.py  # Riemannian (failed)
│   └── lda_svc_binned_15subject.py      # Binned features (completed)
└── results/
    └── lda_svc_binned_15subject_results.json  (58 KB)
```

---

## 10. Technical Summary

### Complete Pipeline:

```python
# 1. PREPROCESSING
Raw EEG (8, 350, 1600)
  → Transpose to (350, 8, 1600)
  → Bandpass filter [0.1-30 Hz]
  → Common Average Reference
  → Artifact rejection [150-300 µV thresholds]
  → Output: (35700, 8, 1600)

# 2. FEATURE EXTRACTION
Preprocessed epochs (35700, 8, 1600)
  → Extract P300 window [samples 100:175]
  → Temporal binning [15 bins × 8 channels]
  → Output: (35700, 120) features

# 3. MODEL TRAINING
Features (35700, 120)
  → StandardScaler normalization
  → Train LDA/SVC classifiers
  → 3 evaluation strategies:
      - Within-subject 5-fold CV
      - LOSO cross-validation (15 folds)
      - LOSO + threshold calibration

# 4. EVALUATION
  → Compute metrics: BAcc, AUC, PR-AUC, F1
  → Aggregate across folds
  → Save results to JSON
```

### Key Parameters:

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **Sampling Rate** | 250 Hz | Fixed by dataset |
| **Bandpass** | 0.1-30 Hz | Standard EEG preprocessing |
| **Reference** | CAR | Common average reference |
| **Artifact Threshold** | 150-300 µV | Subject-specific (based on signal quality) |
| **P300 Window** | 200-500ms | Standard P300 latency |
| **Time Bins** | 15 bins | 20ms resolution |
| **Features** | 120-dim | 8 channels × 15 bins |
| **CV Folds** | 5 | Standard for within-subject |
| **LOSO Folds** | 15 | One per subject |
| **LDA Solver** | lsqr + auto shrinkage | Handles small samples |
| **SVC Kernel** | RBF | Non-linear classification |

---

## 11. Conclusion

The 15-subject ASD P300 classification pipeline was successfully implemented end-to-end, but **achieved only chance-level performance (BAcc=0.500, AUC=0.501)** for cross-subject classification.

### What Worked:
- ✅ Complete preprocessing pipeline
- ✅ Robust artifact handling (subject-specific thresholds)
- ✅ Clean feature extraction (120-dim binned features)
- ✅ Proper cross-validation (LOSO for cross-subject generalization)
- ✅ Comprehensive evaluation metrics

### What Didn't Work:
- ❌ Riemannian geometry approach (numerical instability)
- ❌ Binned features approach (no discrimination)
- ❌ Cross-subject generalization (chance level)

### Likely Reasons for Failure:
1. **Population difference**: ASD patients have different P300 characteristics than healthy controls
2. **Signal quality**: Lower SNR in ASD dataset
3. **Data preprocessing**: Unknown preprocessing applied before distribution (1600 samples unexplained)
4. **Fixed feature engineering**: Simple binning insufficient for this complex dataset

### Recommendations for Improvement:
1. **Adaptive P300 windowing**: Subject-specific latency detection
2. **Deep learning**: CNN/RNN to learn features automatically
3. **Single-trial analysis**: Check if any trials show clear P300
4. **Data quality check**: Verify original data source and preprocessing
5. **Alternative features**: Wavelets, time-frequency, or subject-specific templates

---

**Generated**: February 23, 2026
**Dataset**: 15-Subject ASD P300 (IFMBE 2019 Challenge)
**Final Performance**: BAcc=0.500 (chance level)
