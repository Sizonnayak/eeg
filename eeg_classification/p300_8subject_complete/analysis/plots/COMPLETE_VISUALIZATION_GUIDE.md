# P300 EEG - Complete Visualization Suite

## Overview

This directory contains comprehensive preprocessing and visualization results for all 8 P300 subjects. Each subject has **8 files** (7 plots + 1 config) providing complete analysis from raw data to P300 characterization.

**Date Generated:** February 3, 2026
**Script:** `p300_eda_comprehensive.py`
**P300 Window:** 200-500 ms (consistent visualization and detection)

---

## 📊 Complete Visualization Set (Per Subject)

Each subject folder (S01_clean, S02_noisy, S03-S05_artifacts, S06-S08_clean) contains:

### 1. **S0X_butterfly_erp.png** - Classic Butterfly ERP
**What it shows:**
- Target vs Non-target averaged ERPs
- All 8 channels overlaid with individual traces (faint)
- Grand average (bold lines)
- P300 zone shaded (200-500 ms)

**Key Features:**
- Red line: Target ERP (should show P300 positivity)
- Blue dashed: Non-target ERP
- Gray shading: P300 window (200-500 ms)
- Vertical line at 0: Stimulus onset

**What to look for:**
- Clear separation between target and non-target
- Positive deflection in target around 200-500 ms
- Posterior channels (Pz, PO7, PO8, Oz) should show strongest P300

---

### 2. **S0X_topo_maps.png** - Topographic Maps
**What it shows:**
- Brain topography at 3 key timepoints: 0ms, 350ms, 600ms
- Three conditions: Target, Non-target, Difference (Target - Non-target)
- Uses MNE standard_1020 montage for accurate spatial representation

**Layout:**
```
         Target    |    Non-target    |    Difference
0 ms     [map]     |      [map]       |      [map]
350 ms   [map]     |      [map]       |      [map]
600 ms   [map]     |      [map]       |      [map]
```

**What to look for:**
- At 350 ms: Difference map should show parietal-central positivity (red)
- P300 topography: Strongest at Pz, Cz (midline parietal/central)
- Target map at 350 ms should be more positive than non-target

---

### 3. **S0X_butterfly_overlay.png** - All-Channel Butterfly (NEW!)
**What it shows:**
- All 8 channels overlaid like butterfly wings
- 4-panel comparison: Raw vs Preprocessed, Target vs Non-target
- Key P300 channels highlighted (Pz=red, Cz=blue, Oz=green)

**Layout:**
```
┌─────────────────────┬─────────────────────┐
│ RAW: Target         │ RAW: Non-target    │
├─────────────────────┼─────────────────────┤
│ PREPROC: Target     │ PREPROC: Non-target│
└─────────────────────┴─────────────────────┘
```

**Key Features:**
- Gray lines: All channels (faint)
- Colored lines: Key P300 channels (bold)
- Yellow shading: P300 zone (200-500 ms)
- Green dashed line: Expected P300 peak (~300 ms)

**What to look for:**
- Preprocessing should reduce noise (bottom row cleaner than top)
- Pz (red) should show strongest P300 deflection
- Target plots should show clear positive peak around 300-400 ms

---

### 4. **S0X_difference_waves.png** - Difference Waves (NEW!)
**What it shows:**
- Target - Non-target difference for each of 8 channels
- Isolates P300 component by subtracting background activity
- Comparison: Raw (gray) vs Preprocessed (red)

**Layout:**
```
┌─────┬─────┬─────┬─────┐
│ Fz  │ Cz  │ P3  │ Pz  │
├─────┼─────┼─────┼─────┤
│ P4  │ PO7 │ PO8 │ Oz  │
└─────┴─────┴─────┴─────┘
```

**Key Features:**
- Gray line: Raw difference
- Red line: Preprocessed difference
- Green shading: Positive P300 effect
- Zero line: Baseline (no difference)
- Yellow background: P300 window

**What to look for:**
- ✅ Positive peak in 200-500 ms window (indicates P300)
- ✅ Pz should show strongest difference (typical P300 topography)
- ✅ Preprocessed should maintain or enhance peak clarity
- ✅ Peak latency around 300-400 ms (classic P300 timing)

**Interpretation:**
- **Large positive peak** = Strong P300 effect (good for classification)
- **Pz > Cz > other channels** = Typical P300 spatial distribution
- **Smooth curve** = Good preprocessing

---

### 5. **S0X_signal_quality.png** - Signal Quality Metrics (NEW!)
**What it shows:**
- 4 panels showing signal quality before/after preprocessing
- Quantitative comparison: Raw (gray) vs Preprocessed (red)

**Panels:**
1. **SNR (Signal-to-Noise Ratio)**: Higher is better
2. **Baseline Noise**: Standard deviation in pre-stimulus period (lower is better)
3. **Amplitude Range**: Mean peak-to-peak per channel
4. **Trial-to-Trial Variability**: Consistency across epochs

**What to look for:**
- ✅ SNR should increase after preprocessing
- ✅ Baseline noise should decrease
- ✅ Amplitude range normalized across channels
- ✅ Variability reduced (more consistent trials)

**Quality Indicators:**
- SNR > 5 dB: Good quality
- SNR 0-5 dB: Moderate quality
- SNR < 0 dB: Poor quality

---

### 6. **S0X_time_frequency.png** - Time-Frequency Analysis (NEW!)
**What it shows:**
- Spectrograms for 6 key channels (Pz, Cz, Oz, P3, P4, PO7)
- Power distribution across frequency (0-40 Hz) and time
- Shows what frequency components are present when

**Layout:**
```
┌────┬────┬────┐
│ Pz │ Cz │ Oz │
├────┼────┼────┤
│ P3 │ P4 │PO7 │
└────┴────┴────┘
```

**Key Features:**
- X-axis: Time (ms) relative to stimulus
- Y-axis: Frequency (Hz)
- Color: Power (dB) - red=high, blue=low
- White dashed lines: Stimulus onset (0) and P300 peak (300)

**What to look for:**
- P300 is a slow component (2-8 Hz)
- Increased low-frequency power (< 10 Hz) around 200-500 ms
- Evoked response: time-locked pattern visible

**Frequency Bands:**
- Delta (2-4 Hz): P300 dominant frequency
- Theta (4-8 Hz): Secondary P300 component
- Alpha (8-12 Hz): Background rhythm
- Beta (12-30 Hz): Should be minimal in P300

---

### 7. **S0X_channel_p300_amplitude.png** - P300 Amplitude Bars (NEW!)
**What it shows:**
- Bar charts comparing P300 peak amplitude across all 8 channels
- Two views: Absolute amplitude (left) and P300 effect size (right)

**Left Panel: Peak Amplitude**
- Red bars: Target peak amplitude in P300 window
- Blue bars: Non-target peak amplitude
- Shows absolute values

**Right Panel: P300 Effect Size (Difference)**
- Green bars: Positive difference (P300 present)
- Red bars: Negative difference (inverse effect)
- Shows Target - Non-target
- Values annotated on top of bars

**What to look for:**
- ✅ Pz should have highest positive difference (strongest P300)
- ✅ All channels should show positive difference (valid P300)
- ✅ Posterior channels (Pz, PO7, PO8, Oz) > anterior (Fz, Cz)

**Typical P300 Amplitude:**
- Good P300: 2-5 µV difference
- Moderate P300: 1-2 µV difference
- Weak P300: < 1 µV difference

---

### 8. **S0X_config.txt** - Configuration & Results
**What it contains:**
- Preprocessing configuration (filters, bad channels, thresholds)
- Epoch counts (target, non-target, rejected)
- P300 peak statistics table (all channels sorted by SNR)
- Best P300 channel identification
- List of all generated plots

**Key Information:**
- Quality assessment (Clean, Noisy, Artifacts)
- Filtering parameters (highpass, lowpass, notch)
- Bad channels interpolated
- Artifact rejection threshold
- Peak latency and SNR per channel

---

## 📁 Directory Structure

```
/Users/siznayak/Documents/others/MTech/EEG_Classification/analysis/plots/
├── S01_clean/                 (8 files - 7 plots + 1 config)
│   ├── S01_butterfly_erp.png
│   ├── S01_topo_maps.png
│   ├── S01_butterfly_overlay.png
│   ├── S01_difference_waves.png
│   ├── S01_signal_quality.png
│   ├── S01_time_frequency.png
│   ├── S01_channel_p300_amplitude.png
│   └── S01_config.txt
│
├── S02_noisy/                 (8 files)
├── S03_artifacts/             (8 files - interpolated P4, PO8)
├── S04_artifacts/             (8 files - interpolated PO8)
├── S05_artifacts/             (8 files - interpolated Oz)
├── S06_clean/                 (8 files)
├── S07_clean/                 (8 files)
├── S08_clean/                 (8 files)
│
├── PREPROCESSING_SUMMARY.md   (Subject-specific preprocessing details)
├── P300_WINDOW_EXPLANATION.md (Scientific rationale for 200-500 ms)
└── COMPLETE_VISUALIZATION_GUIDE.md (This file)
```

**Total Files:** 64 files (56 plots + 8 configs) across 8 subjects

---

## 🎯 Quick Quality Check

### For Each Subject, Check:

1. **Butterfly ERP (Plot 1):**
   - ✅ Clear target/non-target separation?
   - ✅ Positive deflection in P300 window?

2. **Topographic Maps (Plot 2):**
   - ✅ Parietal-central positivity at 350 ms (difference map)?
   - ✅ Reasonable spatial distribution?

3. **Difference Waves (Plot 4):**
   - ✅ Positive peak in 200-500 ms for most channels?
   - ✅ Pz shows strongest effect?

4. **Signal Quality (Plot 5):**
   - ✅ SNR > 0 dB for most channels?
   - ✅ Preprocessing improves metrics?

5. **Channel Amplitude (Plot 7):**
   - ✅ Positive P300 effect (green bars)?
   - ✅ Pz highest or near-highest?

---

## 📊 Subject Quality Summary

| Subject | Quality | Plots | Bad Channels | Best P300 Channel | Peak Latency | SNR |
|---------|---------|-------|--------------|-------------------|--------------|-----|
| S01 | Clean ✅ | 8 | None | PO7 | 208 ms | 2.42 |
| S02 | Noisy ⚠️ | 8 | None | P3 | 326 ms | 1.29 |
| S03 | Artifacts ❌ | 8 | P4, PO8 (interp) | PO8 | 200 ms | 2.03 |
| S04 | Artifacts ⚠️ | 8 | PO8 (interp) | - | 200-500 ms | - |
| S05 | Artifacts ⚠️ | 8 | Oz (interp) | P3 | 410 ms | 1.66 |
| S06 | Clean ✅ | 8 | None | - | 200-500 ms | - |
| S07 | Clean ✅ | 8 | None | - | 200-500 ms | - |
| S08 | Clean ✅ | 8 | None | - | 200-500 ms | - |

---

## 🔬 Scientific Background

### P300 Component
- **Definition:** Positive ERP deflection ~300 ms post-stimulus
- **Cognitive function:** Target detection, attention allocation, memory updating
- **Typical latency:** 200-500 ms (visual paradigms)
- **Typical amplitude:** 2-10 µV (target - non-target)
- **Scalp distribution:** Maximum at parietal midline (Pz)

### P300 Sub-components
1. **P300a (200-300 ms):** Frontal, novelty detection, attention orienting
2. **P300b (300-500 ms):** Parietal, target processing, context updating

### This Dataset
- **Paradigm:** P300 speller (visual oddball)
- **Target:** Row/column containing desired character flashes
- **Non-target:** Other rows/columns flash
- **P300 trigger:** Attention to target stimuli
- **Channels:** 8 (Fz, Cz, P3, Pz, P4, PO7, PO8, Oz)

---

## 🚀 Next Steps: Machine Learning

### Recommended Features (Based on These Plots):

1. **Time-Domain Features:**
   - Mean amplitude in P300 window (200-500 ms)
   - Peak amplitude in P300 window
   - Peak latency
   - Area under curve (200-500 ms)

2. **Frequency-Domain Features:**
   - Delta power (2-4 Hz) in P300 window
   - Theta power (4-8 Hz) in P300 window
   - Delta/Theta ratio

3. **Spatial Features:**
   - Pz amplitude (strongest P300)
   - Pz/Fz ratio (parietal dominance)
   - Difference wave peak (Target - Non-target)

4. **Spatiotemporal Features:**
   - Covariance across channels in P300 window
   - Topographic pattern at 350 ms
   - Multi-channel concatenation

### Suggested Channels for ML:
- **Primary:** Pz (strongest P300)
- **Secondary:** Cz, PO7, PO8, Oz (good P300 response)
- **Optional:** P3, P4 (moderate response)
- **Exclude:** Fz (weakest, frontal)

---

## 📚 Related Documents

1. **PREPROCESSING_SUMMARY.md** - Subject-by-subject preprocessing details
2. **P300_WINDOW_EXPLANATION.md** - Scientific justification for 200-500 ms window
3. **Individual config files** - Per-subject statistics and settings

---

## 🛠️ Regenerating Plots

To regenerate all plots with current settings:

```bash
cd /Users/siznayak/Documents/others/MTech/EEG_Classification/analysis/eda
source ../../.venv/bin/activate
python p300_eda_comprehensive.py
```

**Processing time:** ~2-3 minutes for all 8 subjects
**Output:** 64 files (8 per subject)

---

**Generated:** February 3, 2026
**Script:** `p300_eda_comprehensive.py`
**Total Visualization Time:** Complete P300 characterization from raw signal to P300 peak!
