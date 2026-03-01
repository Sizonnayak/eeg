# P300 EEG Classification — Complete Project

**Dataset:** p300_8subject (BCI Competition-style dataset)  
**Subjects:** 8  
**Epochs:** 33,489 total (5,575 target, 27,914 non-target)  
**Channels:** 8 EEG (Fz, Cz, P3, Pz, P4, PO7, PO8, Oz)  
**Sampling rate:** 250 Hz  

---

## Folder Structure

```
P300_8subject_complete/
├── raw_data/              # Original .mat files (8 subjects)
├── preprocessing/         # v1 and v2 preprocessing scripts + outputs
├── features/              # Feature extraction scripts + .npz files
├── models/                # v1, v2, v3, v4 classifier scripts
├── results/               # JSON results + plots for all versions
├── analysis/              # Data inspection scripts + reports
├── docs/                  # Presentation, papers, documentation
├── run_pipeline.py        # Master script to run entire pipeline
└── README.md              # This file
```

---

## Quick Start

### 1. Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the full pipeline (v4)
```bash
python run_pipeline.py --version v4
```

Skip preprocessing/features if already done:
```bash
python run_pipeline.py --version v4 --skip-preproc --skip-features
```

---

## Pipeline Versions

| Version | Method | LOSO BAcc | LOSO AUC | Description |
|---------|--------|-----------|----------|-------------|
| **v1** | rLDA 120d raw bins | 0.519 | 0.636 | Baseline: binned ERP features |
| **v2** | rLDA + xDAWN(2) | 0.505 | 0.606 | CAR + 0.1-30Hz + xDAWN (failed) |
| **v3** | xDAWN sweep k=0,2,4,6 | 0.509 | 0.608 | Proved xDAWN doesn't help LOSO |
| **v4** | Riemannian EA+TS | 0.517 | 0.648 | XdawnCov → EA → TangentSpace |
| **v4+cal** | v4 + Youden's J | **0.605** | 0.648 | **Best** with threshold calibration |

---

## Key Files

- `run_pipeline.py` — Master script with `--version` flag
- `preprocessing/p300_preprocess_v2.py` — 0.1-30Hz + CAR + bad channel interpolation
- `features/extract_p300_features_v2.py` — 200-500ms → 15 bins → 120-dim
- `models/lda_svc_classifier_v4.py` — Riemannian pipeline + Youden's J calibration
- `results/classification_results_v4.json` — Final results
- `docs/P300_EEG_Classification.pptx` — 15-slide presentation

---

## Results (v4 + calibration)

**Within-Subject (5-fold CV):**
- BAcc = 0.600, AUC = 0.746, PR-AUC = 0.443

**Cross-Subject (LOSO, calibrated):**
- BAcc = 0.605 ± 0.050
- AUC = 0.648 ± 0.061
- PR-AUC = 0.296 (1.78× above random baseline of 0.167)

**Per-subject LOSO (calibrated):**
```
S01: BAcc=0.510, AUC=0.539, thresh=0.178
S02: BAcc=0.567, AUC=0.593, thresh=0.165
S03: BAcc=0.581, AUC=0.611, thresh=0.174
S04: BAcc=0.616, AUC=0.667, thresh=0.181
S05: BAcc=0.628, AUC=0.684, thresh=0.165
S06: BAcc=0.621, AUC=0.668, thresh=0.154
S07: BAcc=0.690, AUC=0.750, thresh=0.183  ← best
S08: BAcc=0.627, AUC=0.672, thresh=0.156
```

---

## Citation

If you use this code, please cite:
```
P300 EEG Classification Pipeline
M.Tech Project, Department of AI & Data Science
2025-2026
```

Dataset: BCI Competition-style 8-subject P300 speller data

---

## Contact

For questions or issues, contact: [your email/github]
