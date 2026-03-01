# EEG Inspector – Dynamic Usage Guide

Flexible, format‑agnostic inspector for EEG datasets. Works with MATLAB `.mat`, MNE‑backed formats (EDF/BDF/BrainVision/EEGLAB/FIF/GDF), array files (`.npy/.npz`), plain tables (`.csv/.txt`), and PyTorch checkpoints (`.pth`).

The script prints a per‑file summary (structure, channels, QC, events) and a cross‑file consistency section.

## Quick Start

From the directory containing `data_inspect.py` (recommended):

```
python EEG_Classification/analysis/inspect/data_inspect.py --data-dir /path/to/dir --pattern "*.mat"
```

Or from anywhere using a relative/absolute path to the script:

```
python path/to/analysis/inspect/data_inspect.py --data-dir /path/to/edf --pattern "*.edf" --mne-events auto
```

NPZ with explicit sampling rate:

```
python EEG_Classification/analysis/inspect/data_inspect.py --data-dir /path/to/npz --pattern "*.npz" --fs 256
```

PTH (PyTorch), fall back to `--fs` if not embedded:

```
python EEG_Classification/analysis/inspect/data_inspect.py --data-dir /path/to/pth --pattern "*.pth" --fs 500
```

Save a combined text report and JSON summary:

```
python EEG_Classification/analysis/inspect/data_inspect.py --data-dir /data --pattern "*.edf" \
  --mne-events auto \
  --save-report /data/inspect_report.txt \
  --save-json /data/inspect_report.json
```

Tip: quote patterns like `"*.edf"` to avoid shell expansion.

## Supported Formats

- MAT: `*.mat` via `scipy.io.loadmat`
- MNE‑backed: `*.edf`, `*.bdf`, `*.vhdr` (BrainVision), `*.set` (EEGLAB), `*.fif`, `*.gdf`
- Arrays: `*.npy`, `*.npz` (requires `--fs` if not embedded)
- Tables: `*.csv`, `*.txt` (requires `--fs`)
- PyTorch: `*.pth` (tries to locate arrays/metadata; else `--fs`)

Optional dependencies:
- MNE (`pip install mne`) for non‑MAT formats and event extraction
- PyTorch (`pip install torch`) for `.pth`

## CLI Options

- `--data-dir` (required): Folder containing files to inspect
- `--pattern` (default `"*.mat"`): Glob pattern for files
- `--units-scale` (default `1.0`): Multiply samples to convert to µV (e.g., volts→µV = `1e6`)
- `--eeg-key`: MAT key path to EEG matrix (e.g., `data.X`)
- `--event-key`: MAT key path to events table (e.g., `data.flash`)
- `--fs-key`: MAT key name/path to sampling rate (e.g., `Fs` or `data.fs`)
- `--ch-names-key`: MAT key path to channel names (e.g., `channelNames`)
- `--fs`: Sampling rate (Hz) for formats without embedded fs (`npy/npz/csv/txt/pth`)
- `--mne-events` (`auto|annotations|stim|none`, default `auto`): How to extract events with MNE
- `--max-files`: Limit number of files to scan (0 = all)

## What It Detects

- Sampling rate (Fs) from metadata or `--fs`
- EEG array (2D samples×channels), re‑orients if needed
- Channel names (or generates `Ch1..ChN`)
- Event table (MAT or MNE annotations/stim) if available
- Signal QC: NaN/Inf counts, flat/saturated channels, PSD with mains (50/60 Hz) detection, alpha‑band presence
- Event timing: inter‑event intervals, jitter, simultaneous events
- Cross‑file consistency: Fs set, channel set, durations

## Heuristics & Conventions

- EEG matrix selection: prefer the largest 2D array where samples > channels and `2 ≤ channels ≤ 512`
- Events (MAT): prefer the largest small 2D array (< 10% of EEG rows) or `--event-key`
- Mains detection: compares 50±2 vs 60±2 Hz PSD and reports the dominant ratio
- Units: run QC on data after applying `--units-scale` (assumes thresholds in µV)

## Examples

MAT with explicit keys:

```
python EEG_Classification/analysis/inspect/data_inspect.py --data-dir /data/mat --pattern "*.mat" \
  --eeg-key data.X --event-key data.flash --fs-key Fs --ch-names-key channelNames
```

BrainVision with events from annotations only:

```
python EEG_Classification/analysis/inspect/data_inspect.py --data-dir /data/vhdr --pattern "*.vhdr" --mne-events annotations
```

CSV where amplitudes are in volts (scale to µV) and fs=1000:

```
python EEG_Classification/analysis/inspect/data_inspect.py --data-dir /data/csv --pattern "*.csv" --fs 1000 --units-scale 1e6
```

Per-file text reports next to each input:

```
python EEG_Classification/analysis/inspect/data_inspect.py --data-dir /data --pattern "*.mat" --save-per-file
```

## Output Glossary

- `Flat channels`: variance < 1 µV² (adjust via `--units-scale` if needed)
- `Saturated`: |amplitude| > 500 µV (post‑scale)
- `Mains (X Hz) ratio`: PSD power at X Hz relative to 5–45 Hz band; `FLAG` suggests a notch is advisable
- `Alpha peak present`: alpha (8–13 Hz) power exceeds theta (4–7 Hz)
- `Timing jitter`: std of short inter‑event intervals (≤ 2 s)

## Troubleshooting

- "No files found": check `--data-dir` and quote your `--pattern`
- "Sampling rate is required": provide `--fs` for `npy/npz/csv/txt/pth` when fs isn’t embedded
- "MNE is not available": install `mne` to read non‑MAT formats
- "PyTorch is required": install `torch` to read `.pth`
- Strange flags on units: set `--units-scale` so data are in µV for QC thresholds

## Notes & Limits

- Event extraction for non‑MAT depends on MNE annotations/stim; complex schemas may need custom logic
- Very short recordings can make PSD‑based checks less stable
- The inspector is read‑only; it does not modify files
