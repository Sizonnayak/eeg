# P300 EEG Dataset - Subject-Specific Preprocessing Summary

## Overview

This document summarizes the customized preprocessing pipeline applied to each subject in the P300 dataset based on their individual data quality characteristics.

**Date:** February 3, 2026
**Dataset:** P300 8-Subject Dataset
**Total Subjects:** 8
**Preprocessing Script:** `p300_eda_subject_specific.py`

---

## P300 Window Definition

**Consistent P300 Window: 200-500 ms**
- Based on visual P300 ERP literature (Polich, 2007; Krusienski et al., 2006)
- Used for both visualization (shaded region in plots) and peak detection
- Captures early P300a (200-300 ms) and classic P300b (300-500 ms) components
- Optimal window for visual oddball paradigms in BCI applications

## Preprocessing Strategy

Each subject received customized filtering based on data quality assessment from the initial inspection:

### Clean Subjects (S01, S06, S07, S08)
- **Quality:** High quality, no artifacts
- **Filtering:** Conservative (0.5-20 Hz bandpass)
- **Rationale:** Preserve more frequency content, focus on P300 range
- **Artifact Threshold:** 150 µV
- **Bad Channels:** None

### Noisy Subject (S02)
- **Quality:** High variance in P3 channel
- **Filtering:** Standard (1.0-30 Hz bandpass)
- **Rationale:** Keep broader bandwidth to handle variability
- **Artifact Threshold:** 200 µV (more lenient)
- **Bad Channels:** None

### Artifact Subjects (S03, S04, S05)
- **Quality:** Saturated channels detected
- **Filtering:** Restrictive (1.0-20 Hz bandpass)
- **Rationale:** Focus on P300 range, reduce artifact impact
- **Artifact Threshold:** 100 µV (stricter rejection)
- **Bad Channels:** Interpolated using spherical splines
  - **S03:** P4, PO8 (severe saturation: max 3837 µV)
  - **S04:** PO8 (saturation: max 1780 µV)
  - **S05:** Oz (saturation: max 892 µV)

---

## Subject-by-Subject Details

### Subject 01 (S01_clean)
- **Quality:** Clean ✅
- **Filtering:** 0.5-20 Hz, no notch
- **Bad Channels:** None
- **Target Epochs:** 3,500
- **Non-Target Epochs:** 700
- **Rejected Epochs:** 0
- **Best P300 Channel:** (see config file)
- **Outputs:**
  - `S01_butterfly_erp.png` - ERP waveforms
  - `S01_topo_maps.png` - Topographic maps
  - `S01_config.txt` - Configuration and results

### Subject 02 (S02_noisy)
- **Quality:** Noisy ⚠️
- **Filtering:** 1.0-30 Hz, no notch
- **Bad Channels:** None (high variance in P3, but not interpolated)
- **Target Epochs:** 3,498
- **Non-Target Epochs:** 700
- **Rejected Epochs:** 0
- **Note:** Kept broader bandwidth due to high signal variance
- **Outputs:**
  - `S02_butterfly_erp.png`
  - `S02_topo_maps.png`
  - `S02_config.txt`

### Subject 03 (S03_artifacts)
- **Quality:** Severe Artifacts ❌
- **Filtering:** 1.0-20 Hz, no notch
- **Bad Channels:** P4, PO8 (interpolated)
- **Target Epochs:** 3,497
- **Non-Target Epochs:** 700
- **Rejected Epochs:** 0
- **Artifact Details:** P4 reached 3837 µV (extreme saturation)
- **Outputs:**
  - `S03_butterfly_erp.png`
  - `S03_topo_maps.png`
  - `S03_config.txt`

### Subject 04 (S04_artifacts)
- **Quality:** Artifacts ⚠️
- **Filtering:** 1.0-20 Hz, no notch
- **Bad Channels:** PO8 (interpolated)
- **Target Epochs:** 3,488
- **Non-Target Epochs:** 695
- **Rejected Epochs:** 14
- **Artifact Details:** PO8 reached 1780 µV
- **Outputs:**
  - `S04_butterfly_erp.png`
  - `S04_topo_maps.png`
  - `S04_config.txt`

### Subject 05 (S05_artifacts)
- **Quality:** Artifacts ⚠️
- **Filtering:** 1.0-20 Hz, no notch
- **Bad Channels:** Oz (interpolated)
- **Target Epochs:** 3,439
- **Non-Target Epochs:** 682
- **Rejected Epochs:** 77 (highest rejection rate)
- **Artifact Details:** Oz reached 892 µV
- **Note:** Most aggressive artifact rejection needed
- **Outputs:**
  - `S05_butterfly_erp.png`
  - `S05_topo_maps.png`
  - `S05_config.txt`

### Subject 06 (S06_clean)
- **Quality:** Clean ✅
- **Filtering:** 0.5-20 Hz, no notch
- **Bad Channels:** None
- **Target Epochs:** 3,497
- **Non-Target Epochs:** 700
- **Rejected Epochs:** 0
- **Outputs:**
  - `S06_butterfly_erp.png`
  - `S06_topo_maps.png`
  - `S06_config.txt`

### Subject 07 (S07_clean)
- **Quality:** Clean ✅
- **Filtering:** 0.5-20 Hz, no notch
- **Bad Channels:** None
- **Target Epochs:** 3,500
- **Non-Target Epochs:** 698
- **Rejected Epochs:** 0
- **Outputs:**
  - `S07_butterfly_erp.png`
  - `S07_topo_maps.png`
  - `S07_config.txt`

### Subject 08 (S08_clean)
- **Quality:** Clean ✅
- **Filtering:** 0.5-20 Hz, no notch
- **Bad Channels:** None
- **Target Epochs:** 3,482
- **Non-Target Epochs:** 697
- **Rejected Epochs:** 19
- **Outputs:**
  - `S08_butterfly_erp.png`
  - `S08_topo_maps.png`
  - `S08_config.txt`

---

## Summary Statistics

| Subject | Quality | Bad Channels | Filter (Hz) | Target Epochs | Non-Target | Rejected |
|---------|---------|--------------|-------------|---------------|------------|----------|
| S01 | Clean ✅ | None | 0.5-20 | 3,500 | 700 | 0 |
| S02 | Noisy ⚠️ | None | 1.0-30 | 3,498 | 700 | 0 |
| S03 | Severe ❌ | P4, PO8 | 1.0-20 | 3,497 | 700 | 0 |
| S04 | Artifacts ⚠️ | PO8 | 1.0-20 | 3,488 | 695 | 14 |
| S05 | Artifacts ⚠️ | Oz | 1.0-20 | 3,439 | 682 | 77 |
| S06 | Clean ✅ | None | 0.5-20 | 3,497 | 700 | 0 |
| S07 | Clean ✅ | None | 0.5-20 | 3,500 | 698 | 0 |
| S08 | Clean ✅ | None | 0.5-20 | 3,482 | 697 | 19 |

**Total Epochs After Preprocessing:**
- **Target:** 27,901 (avg 3,488 per subject)
- **Non-Target:** 5,572 (avg 697 per subject)
- **Total Rejected:** 110 epochs across all subjects

---

## Key Findings

### 1. Data Quality Distribution
- **Clean subjects:** 4/8 (50%) - S01, S06, S07, S08
- **Noisy subjects:** 1/8 (12.5%) - S02
- **Artifact subjects:** 3/8 (37.5%) - S03, S04, S05

### 2. Artifact Rejection
- Subject S05 had the highest rejection rate (77 epochs = 1.8%)
- Most subjects (5/8) had zero rejected epochs after interpolation
- Total data loss: ~0.3% across entire dataset

### 3. Bad Channel Interpolation
- 4 channels interpolated across 3 subjects
- All interpolations used spherical splines with MNE's standard_1020 montage
- Interpolation successful in recovering data quality

### 4. Filtering Rationale
- **No notch filter applied** to any subject (inspection showed no 50 Hz line noise)
- **Conservative filtering (0.5-20 Hz)** for clean subjects preserves low-frequency components
- **Restrictive filtering (1.0-20 Hz)** for artifact subjects reduces noise
- **Standard filtering (1.0-30 Hz)** for noisy subject maintains bandwidth

---

## File Organization

```
/Users/siznayak/Documents/others/MTech/EEG_Classification/analysis/plots/
├── S01_clean/
│   ├── S01_butterfly_erp.png
│   ├── S01_topo_maps.png
│   └── S01_config.txt
├── S02_noisy/
│   ├── S02_butterfly_erp.png
│   ├── S02_topo_maps.png
│   └── S02_config.txt
├── S03_artifacts/
│   ├── S03_butterfly_erp.png
│   ├── S03_topo_maps.png
│   └── S03_config.txt
├── S04_artifacts/
│   ├── S04_butterfly_erp.png
│   ├── S04_topo_maps.png
│   └── S04_config.txt
├── S05_artifacts/
│   ├── S05_butterfly_erp.png
│   ├── S05_topo_maps.png
│   └── S05_config.txt
├── S06_clean/
│   ├── S06_butterfly_erp.png
│   ├── S06_topo_maps.png
│   └── S06_config.txt
├── S07_clean/
│   ├── S07_butterfly_erp.png
│   ├── S07_topo_maps.png
│   └── S07_config.txt
├── S08_clean/
│   ├── S08_butterfly_erp.png
│   ├── S08_topo_maps.png
│   └── S08_config.txt
└── PREPROCESSING_SUMMARY.md (this file)
```

---

## Scripts Used

### Main Preprocessing Script
**Location:** `/Users/siznayak/Documents/others/MTech/EEG_Classification/analysis/eda/p300_eda_subject_specific.py`

**Key Features:**
- Subject-specific configuration dictionary
- Custom filtering per subject
- Automatic bad channel interpolation
- Artifact-based epoch rejection
- Baseline correction (pre-stimulus mean subtraction)
- P300 peak statistics computation

**To run:**
```bash
cd /Users/siznayak/Documents/others/MTech/EEG_Classification/analysis/eda
source ../../.venv/bin/activate
python p300_eda_subject_specific.py
```

### Data Inspection Script
**Location:** `/Users/siznayak/Documents/others/MTech/EEG_Classification/analysis/inspect/data_inspect.py`

Used for initial quality assessment to determine preprocessing strategy.

---

## Recommendations for Machine Learning

### 1. Subject Selection
- **High-quality training:** Use S01, S06, S07, S08 (clean subjects)
- **Validation:** Include S02 to test robustness to noise
- **Testing artifacts:** Include S03, S04, S05 to evaluate artifact handling

### 2. Cross-Validation Strategy
- Consider subject-wise leave-one-out cross-validation
- Account for subject variability in model evaluation
- Consider excluding S03 if artifact interpolation is insufficient

### 3. Feature Extraction
- Focus on 200-600 ms post-stimulus window (P300 range)
- Prioritize posterior channels: Pz, PO7, PO8, Oz
- Consider channel-specific features given interpolation in some subjects

### 4. Data Augmentation
- Avoid augmentation on interpolated channels
- Consider temporal jittering (±50 ms) for clean subjects only

---

## Next Steps

1. ✅ Data inspection complete
2. ✅ Subject-specific preprocessing complete
3. ⏭️ Feature extraction (time-domain, frequency-domain, spatial features)
4. ⏭️ Model training (CNN, LSTM, or classical ML)
5. ⏭️ Cross-validation and performance evaluation

---

## Contact & References

**Dataset:** P300 8-Subject BCI Dataset
**Preprocessing:** MNE-Python (filtering, interpolation)
**Visualization:** Matplotlib, MNE topographic plotting

For questions about this preprocessing pipeline, refer to the individual subject config files in each folder.
