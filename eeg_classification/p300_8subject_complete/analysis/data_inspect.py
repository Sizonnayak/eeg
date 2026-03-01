"""
EEG Dataset Inspection (Dynamic)
================================
Flexible pre-EDA inspector for EEG datasets.

Highlights
- Supports multiple formats: MAT (generic), EDF/BDF/BrainVision/FIF (via MNE)
- Auto-discovers sampling rate, channels, event structure, and basic QC
- CLI options for directory, file pattern, unit scaling, and key overrides
- 50/60 Hz mains detection for line-noise flagging

Usage examples
- python data_inspect.py --data-dir ./Dataset/p300_8subject --pattern "*.mat"
- python data_inspect.py --data-dir ./recordings --pattern "*.edf"
- python data_inspect.py --data-dir ./any --pattern "*" --eeg-key data.X --event-key data.flash
"""

import argparse
import os
import glob
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
from scipy.signal import welch
import scipy.io as sio

try:
    import mne  # optional, used for non-MAT formats
    MNE_AVAILABLE = True
except Exception:
    MNE_AVAILABLE = False

# Red-flag thresholds
SATURATION_UV = 500     # amplitude > ±500 µV → saturated
FLAT_CH_VAR = 1.0       # channel variance < 1 µV² → disconnected
JITTER_MS = 100         # timing jitter tolerance (ms)


def parse_args():
    p = argparse.ArgumentParser(description="Dynamic EEG dataset inspector")
    p.add_argument("--data-dir", required=True, help="Directory containing EEG files")
    p.add_argument("--pattern", default="*.mat", help="Glob pattern for files (e.g., *.mat, *.edf)")
    p.add_argument("--units-scale", type=float, default=1.0,
                   help="Multiply signals by this factor to convert to µV")
    p.add_argument("--eeg-key", default=None,
                   help="Key path to EEG matrix in MAT (e.g., data.X)")
    p.add_argument("--event-key", default=None,
                   help="Key path to events in MAT (e.g., data.flash)")
    p.add_argument("--fs-key", default=None,
                   help="Key name/path to sampling rate in MAT (e.g., Fs or data.fs)")
    p.add_argument("--ch-names-key", default=None,
                   help="Key path to channel names in MAT (e.g., channelNames)")
    p.add_argument("--fs", type=float, default=0.0,
                   help="Sampling rate (Hz) to use when the file format does not store it (e.g., npy/csv/pth)")
    p.add_argument("--mne-events", default="auto", choices=["auto", "annotations", "stim", "none"],
                   help="How to extract events with MNE-backed formats")
    p.add_argument("--max-files", type=int, default=0,
                   help="Limit number of files to inspect (0 = all)")
    p.add_argument("--save-report", default=None,
                   help="Path to write a single combined text report (console output)")
    p.add_argument("--save-per-file", action="store_true",
                   help="Also write a per-file summary next to each input file (name: <file>_inspect.txt)")
    p.add_argument("--save-json", default=None,
                   help="Path to write a compact JSON summary for programmatic use")
    return p.parse_args()


def discover_mat_keys(mat: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Return a flat dict of all top-level keys and their types/shapes.
    Handles nested structs by unpacking one level.
    """
    discovered = {}
    for key in mat.keys():
        if key.startswith("__"):
            continue
        val = mat[key]
        # Nested structured array (e.g. mat["data"] with sub-fields)
        if hasattr(val, "dtype") and val.dtype.names:
            for name in val.dtype.names:
                sub = val[name].item()
                if hasattr(sub, "shape"):
                    discovered[f"{key}.{name}"] = {"type": "array", "shape": sub.shape, "dtype": str(sub.dtype)}
                else:
                    discovered[f"{key}.{name}"] = {"type": "scalar", "value": sub}
        elif hasattr(val, "shape"):
            discovered[key] = {"type": "array", "shape": val.shape, "dtype": str(val.dtype)}
        else:
            discovered[key] = {"type": "scalar", "value": val}
    return discovered


def _get_from_path(mat: Dict[str, Any], path: str):
    cur = mat
    for part in path.split('.'):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif hasattr(cur, "dtype") and cur.dtype.names and part in cur.dtype.names:
            cur = cur[part].item()
        else:
            return None
    return cur


def extract_eeg_and_events(mat: Dict[str, Any],
                           eeg_key: Optional[str] = None,
                           event_key: Optional[str] = None,
                           fs_key: Optional[str] = None,
                           ch_names_key: Optional[str] = None
                           ) -> Tuple[np.ndarray, Optional[np.ndarray], int, List[str], Dict[str, Any]]:
    """
    Auto-detect the EEG data array and event table from a MAT file.
    Returns (X, events, fs, ch_names, metadata) or raises ValueError.
    """
    keys = discover_mat_keys(mat)

    # --- Sampling rate ---
    fs = None
    if fs_key:
        val = _get_from_path(mat, fs_key)
        if val is not None:
            fs = int(np.squeeze(val))
    if fs is None:
        for candidate in ["Fs", "fs", "sfreq", "sampling_rate", "SamplingRate"]:
            if candidate in mat:
                fs = int(np.squeeze(mat[candidate]))
                break
            for k in keys:
                if k.endswith(f".{candidate}"):
                    parent, child = k.split(".", 1)
                    fs = int(np.squeeze(mat[parent][candidate].item()))
                    break
            if fs:
                break
    if fs is None:
        raise ValueError("Could not find sampling rate (tried Fs, fs, sfreq, sampling_rate)")

    # --- EEG data
    X = None
    x_key = None
    if eeg_key:
        X = _get_from_path(mat, eeg_key)
        if X is None:
            raise ValueError(f"EEG key '{eeg_key}' not found in MAT file")
        x_key = eeg_key
    else:
        # largest plausible 2D array with samples>channels and 2<=channels<=512
        for k, info in keys.items():
            if info["type"] == "array" and len(info["shape"]) == 2:
                n0, n1 = info["shape"]
                if n0 < n1:
                    n0, n1 = n1, n0
                if not (2 <= n1 <= 512):
                    continue
                if X is None or (n0 * n1) > (X.shape[0] * X.shape[1]):
                    X = _get_from_path(mat, k) if "." in k else mat[k]
                    x_key = k
    if X is None:
        raise ValueError("Could not find a 2D EEG data array")
    # Ensure (n_samples, n_channels) — more samples than channels
    if X.shape[0] < X.shape[1]:
        X = X.T

    n_channels = X.shape[1]

    # --- Channel names: look for channelNames, ch_names, labels ---
    ch_names = None
    if ch_names_key:
        arr = _get_from_path(mat, ch_names_key)
        if arr is not None:
            ch_names = [str(c) for c in np.array(arr).flatten()]
    if ch_names is None:
        for candidate in ["channelNames", "ch_names", "labels", "channel_names", "channels"]:
            val = mat.get(candidate, None)
            if val is not None:
                ch_names = [str(c) for c in np.array(val).flatten()]
                break
            for k in keys:
                if k.endswith(f".{candidate}"):
                    parent, child = k.split(".", 1)
                    arr = mat[parent][child].item()
                    ch_names = [str(c) for c in np.array(arr).flatten()]
                    break
            if ch_names:
                break
    if ch_names is None:
        ch_names = [f"Ch{i+1}" for i in range(n_channels)]

    # --- Events: 2D array with far fewer rows than EEG (event table) ---
    # An event table has ~100s-1000s of rows vs ~100,000s for continuous EEG.
    # Pick the largest 2D array with row count < 10% of EEG length.
    max_event_rows = X.shape[0] * 0.1
    events = None
    if event_key:
        events = _get_from_path(mat, event_key)
    else:
        for k, info in keys.items():
            if k == x_key:
                continue
            if info["type"] != "array" or len(info["shape"]) != 2:
                continue
            if info["shape"][0] >= max_event_rows:
                continue
            candidate = _get_from_path(mat, k) if "." in k else mat[k]
            if events is None or candidate.shape[0] > events.shape[0]:
                events = candidate

    # --- Other scalar metadata ---
    metadata = {}
    for k, info in keys.items():
        if info["type"] == "scalar" and k not in ("Fs", "fs", "sfreq"):
            metadata[k] = info["value"]

    return X, events, fs, ch_names, metadata


def check_signal_quality(X, fs, ch_names, mains_hz: Optional[int] = None, units_scale: float = 1.0):
    """Run signal-quality checks on raw continuous EEG."""
    n_samples, n_ch = X.shape

    X = X * units_scale
    ch_min = X.min(axis=0)
    ch_max = X.max(axis=0)
    n_nan = int(np.isnan(X).sum())
    n_inf = int(np.isinf(X).sum())
    ch_var = X.var(axis=0)
    flat_channels = [ch_names[i] for i in range(n_ch) if ch_var[i] < FLAT_CH_VAR]
    saturated = [ch_names[i] for i in range(n_ch)
                 if ch_max[i] > SATURATION_UV or ch_min[i] < -SATURATION_UV]

    # PSD on first 60 s (or full if shorter)
    seg_len = min(n_samples, int(60 * fs))
    freqs, psd = welch(X[:seg_len, :].T, fs=fs, nperseg=min(seg_len, 1024))
    mean_psd = psd.mean(axis=0)

    # line noise check (auto-detect 50/60 if not provided)
    mask_50 = (freqs >= 48) & (freqs <= 52)
    mask_60 = (freqs >= 58) & (freqs <= 62)
    mask_ref = (freqs >= 5) & (freqs <= 45)
    power_50 = mean_psd[mask_50].mean() if mask_50.any() else 0.0
    power_60 = mean_psd[mask_60].mean() if mask_60.any() else 0.0
    if mains_hz is None:
        mains_hz = 50 if power_50 >= power_60 else 60
    mask_main = mask_50 if mains_hz == 50 else mask_60
    power_ref = mean_psd[mask_ref].mean() if mask_ref.any() else 1.0
    power_main = mean_psd[mask_main].mean() if mask_main.any() else 0.0
    notch_ratio = power_main / power_ref if power_ref > 0 else 0.0
    needs_notch = notch_ratio > 2.0

    # Alpha peak (8-13 Hz) vs theta (4-7 Hz)
    mask_alpha = (freqs >= 8) & (freqs <= 13)
    mask_theta = (freqs >= 4) & (freqs <= 7)
    power_alpha = mean_psd[mask_alpha].mean() if mask_alpha.any() else 0.0
    power_theta = mean_psd[mask_theta].mean() if mask_theta.any() else 1.0
    has_alpha = power_alpha > power_theta

    return {
        "ch_min": ch_min, "ch_max": ch_max, "ch_var": ch_var,
        "n_nan": n_nan, "n_inf": n_inf,
        "flat_channels": flat_channels,
        "saturated_channels": saturated,
        "mains_hz": mains_hz,
        "notch_ratio": round(notch_ratio, 2),
        "needs_notch": needs_notch,
        "has_alpha_peak": has_alpha,
    }


def check_event_timing(events, fs):
    """
    Check inter-event intervals and timing jitter.
    Assumes column 0 contains onset sample indices.
    """
    onsets = events[:, 0]
    intervals_samples = np.diff(onsets)
    intervals_ms = intervals_samples / fs * 1000

    # Jitter: std of non-zero short intervals only
    # (filters out inter-trial gaps and simultaneous events)
    short = intervals_ms[(intervals_ms > 0) & (intervals_ms < 2000)]
    jitter_ms = round(float(short.std()), 2) if len(short) > 0 else 0.0
    n_simultaneous = int(np.sum(intervals_samples == 0))

    return {
        "min_interval_ms": round(float(intervals_ms.min()), 2),
        "mean_interval_ms": round(float(intervals_ms.mean()), 2),
        "max_interval_ms": round(float(intervals_ms.max()), 2),
        "jitter_ms": jitter_ms,
        "n_simultaneous": n_simultaneous,
    }


def summarize_event_columns(events):
    """
    For each column in the event table, report unique values and counts.
    Detects binary/categorical columns automatically.
    """
    columns = []
    for col_idx in range(events.shape[1]):
        vals = events[:, col_idx]
        unique = np.unique(vals)
        columns.append({
            "col": col_idx,
            "n_unique": len(unique),
            "unique_values": unique if len(unique) <= 10 else unique[:5],
            "truncated": len(unique) > 10,
            "min": round(float(vals.min()), 2),
            "max": round(float(vals.max()), 2),
        })
    return columns


def _load_from_mat(filepath: str, args) -> Tuple[np.ndarray, Optional[np.ndarray], int, List[str], Dict[str, Any], Dict[str, Dict[str, Any]]]:
    mat = sio.loadmat(filepath, squeeze_me=True)
    keys_info = discover_mat_keys(mat)
    X, events, fs, ch_names, metadata = extract_eeg_and_events(
        mat,
        eeg_key=args.eeg_key,
        event_key=args.event_key,
        fs_key=args.fs_key,
        ch_names_key=args.ch_names_key,
    )
    return X, events, fs, ch_names, metadata, keys_info


def _read_raw_with_mne(filepath: str):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".edf" and hasattr(mne.io, "read_raw_edf"):
        return mne.io.read_raw_edf(filepath, preload=True, verbose="ERROR")
    if ext == ".bdf" and hasattr(mne.io, "read_raw_bdf"):
        return mne.io.read_raw_bdf(filepath, preload=True, verbose="ERROR")
    if ext == ".vhdr" and hasattr(mne.io, "read_raw_brainvision"):
        return mne.io.read_raw_brainvision(filepath, preload=True, verbose="ERROR")
    if ext == ".set" and hasattr(mne.io, "read_raw_eeglab"):
        return mne.io.read_raw_eeglab(filepath, preload=True, verbose="ERROR")
    if ext == ".fif" and hasattr(mne.io, "read_raw_fif"):
        return mne.io.read_raw_fif(filepath, preload=True, verbose="ERROR")
    if ext == ".gdf" and hasattr(mne.io, "read_raw_gdf"):
        return mne.io.read_raw_gdf(filepath, preload=True, verbose="ERROR")
    # Fallback to generic router if available (newer MNE versions)
    if hasattr(mne.io, "read_raw"):
        return mne.io.read_raw(filepath, preload=True, verbose="ERROR")
    raise RuntimeError(f"Unsupported MNE format for extension: {ext}")


def _extract_mne_events(raw, mode: str):
    events = None
    if mode == "none":
        return None
    if mode in ("annotations", "auto"):
        try:
            if getattr(raw, "annotations", None) and len(raw.annotations) > 0:
                ev, _ = mne.events_from_annotations(raw, verbose="ERROR")
                if ev is not None and len(ev) > 0:
                    events = ev
        except Exception:
            events = None
        if events is not None or mode == "annotations":
            return events
    if mode in ("stim", "auto"):
        try:
            ev = mne.find_events(raw, verbose="ERROR")
            if ev is not None and len(ev) > 0:
                events = ev
        except Exception:
            events = None
    return events


def _load_with_mne(filepath: str, mne_events_mode: str) -> Tuple[np.ndarray, Optional[np.ndarray], int, List[str]]:
    if not MNE_AVAILABLE:
        raise RuntimeError("MNE is not available to read non-MAT formats")
    raw = _read_raw_with_mne(filepath)
    fs = int(raw.info["sfreq"])
    ch_names = list(raw.ch_names)
    X = raw.get_data().T * 1e6  # convert to µV
    events = _extract_mne_events(raw, mne_events_mode)
    return X, events, fs, ch_names


def _find_best_2d_array(obj: Any) -> Optional[np.ndarray]:
    """Recursively search for the largest plausible 2D array (samples x channels)."""
    best = None
    def consider(arr):
        nonlocal best
        if not isinstance(arr, np.ndarray):
            return
        if arr.ndim != 2:
            return
        n0, n1 = arr.shape
        # Prefer samples > channels, with reasonable channel count
        if n0 < n1:
            arr = arr.T
            n0, n1 = arr.shape
        if not (2 <= n1 <= 512):
            return
        if best is None or (n0 * n1) > (best.shape[0] * best.shape[1]):
            best = arr

    def walk(x):
        if isinstance(x, np.ndarray):
            consider(x)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif hasattr(x, "items"):
            try:
                for _, v in x.items():
                    walk(v)
            except Exception:
                pass
        elif hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
            try:
                for v in x:
                    walk(v)
            except Exception:
                pass

    walk(obj)
    return best


def _load_from_numpy(filepath: str) -> np.ndarray:
    if filepath.lower().endswith(".npy"):
        return np.load(filepath)
    # npz: choose the best 2D array among items
    data = np.load(filepath)
    best = None
    for k in data.files:
        arr = data[k]
        if arr.ndim == 2:
            if best is None or arr.size > best.size:
                best = arr
    if best is None:
        raise ValueError("No 2D array found in npz file")
    return best


def _load_from_csv(filepath: str, delimiter: Optional[str] = None) -> np.ndarray:
    try:
        arr = np.loadtxt(filepath, delimiter=delimiter)
    except Exception:
        arr = np.genfromtxt(filepath, delimiter=delimiter)
    if arr.ndim != 2:
        raise ValueError("CSV/TXT must contain a 2D numeric table")
    return arr


def _load_from_pth(filepath: str) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[float], Optional[List[str]]]:
    try:
        import torch  # type: ignore
    except Exception as e:
        raise RuntimeError("PyTorch is required to load .pth files") from e
    obj = torch.load(filepath, map_location="cpu")
    # Convert tensors to numpy where possible
    def to_numpy(x):
        try:
            if hasattr(x, "detach"):
                x = x.detach()
            if hasattr(x, "cpu"):
                x = x.cpu()
            if hasattr(x, "numpy"):
                return x.numpy()
        except Exception:
            pass
        return x

    obj_np = obj
    if isinstance(obj, dict):
        obj_np = {k: to_numpy(v) for k, v in obj.items()}
    else:
        obj_np = to_numpy(obj)

    X = _find_best_2d_array(obj_np)
    if X is None:
        raise ValueError("Could not locate a 2D EEG array inside .pth contents")

    # Try to find optional extras
    fs = None
    ch_names = None
    events = None
    for key in ["fs", "Fs", "sfreq", "sampling_rate", "SamplingRate"]:
        if isinstance(obj_np, dict) and key in obj_np:
            try:
                fs = float(np.squeeze(obj_np[key]))
                break
            except Exception:
                pass
    for key in ["channelNames", "ch_names", "labels", "channel_names", "channels"]:
        if isinstance(obj_np, dict) and key in obj_np:
            try:
                ch_names = [str(c) for c in np.array(obj_np[key]).flatten()]
                break
            except Exception:
                pass
    for key in ["events", "flash", "event_table"]:
        if isinstance(obj_np, dict) and key in obj_np:
            try:
                cand = np.array(obj_np[key])
                if cand.ndim == 2 and cand.shape[0] < X.shape[0] * 0.1:
                    events = cand
                    break
            except Exception:
                pass
    return X, events, fs, ch_names


def inspect_subject(filepath, args):
    """Load one file (MAT/MNE-backed or array-like) and return discovered properties."""
    ext = os.path.splitext(filepath)[1].lower()
    events = None
    metadata = {"path": filepath, "ext": ext}
    keys_info = {}
    if ext == ".mat":
        X, events, fs, ch_names, meta2, keys_info = _load_from_mat(filepath, args)
        metadata.update(meta2)
    elif MNE_AVAILABLE and ext in {".edf", ".bdf", ".vhdr", ".set", ".fif", ".gdf"}:
        X, events, fs, ch_names = _load_with_mne(filepath, args.mne_events)
    elif ext in {".npy", ".npz"}:
        X = _load_from_numpy(filepath)
        # Require fs from args for array formats
        fs = int(args.fs) if args.fs else None
        if not fs:
            raise ValueError("Sampling rate (--fs) is required for npy/npz formats")
        ch_names = [f"Ch{i+1}" for i in range(X.shape[1] if X.ndim == 2 else 0)]
    elif ext in {".csv", ".txt"}:
        X = _load_from_csv(filepath, delimiter=None)
        fs = int(args.fs) if args.fs else None
        if not fs:
            raise ValueError("Sampling rate (--fs) is required for CSV/TXT formats")
        ch_names = [f"Ch{i+1}" for i in range(X.shape[1])]
    elif ext == ".pth":
        X, events, fs_opt, ch_names_opt = _load_from_pth(filepath)
        fs = int(fs_opt) if fs_opt else (int(args.fs) if args.fs else None)
        if not fs:
            raise ValueError("Sampling rate (--fs) is required for .pth when not stored inside")
        ch_names = ch_names_opt if ch_names_opt else [f"Ch{i+1}" for i in range(X.shape[1])]
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    # Ensure orientation is (n_samples, n_channels)
    if X.ndim != 2:
        raise ValueError("EEG data must be a 2D array (samples x channels)")
    if X.shape[0] < X.shape[1]:
        X = X.T
        # Update ch_names length if needed
        if ch_names and len(ch_names) != X.shape[1]:
            ch_names = [f"Ch{i+1}" for i in range(X.shape[1])]

    duration_s = X.shape[0] / fs
    sig_quality = check_signal_quality(X, fs, ch_names, mains_hz=None, units_scale=args.units_scale)

    timing = None
    event_cols = None
    if events is not None:
        timing = check_event_timing(events, fs)
        event_cols = summarize_event_columns(events)

    return {
        "metadata": metadata,
        "keys_info": keys_info,
        "fs": fs,
        "n_channels": len(ch_names),
        "ch_names": ch_names,
        "n_samples": X.shape[0],
        "duration_s": round(duration_s, 2),
        "n_events": events.shape[0] if events is not None else 0,
        "event_cols": event_cols,
        "signal": sig_quality,
        "timing": timing,
    }


def print_sep(char="=", width=60):
    print(char * width)


def flag(condition):
    return "OK" if condition else "FLAG"


def print_subject_report(props, idx):
    """Pretty-print one subject's full inspection."""
    sig = props["signal"]
    ch = props["ch_names"]

    print(f"\n  File {idx+1}  |  Metadata: {props['metadata']}")
    print_sep("-")

    # 1. Structure
    print(f"    Sampling rate      : {props['fs']} Hz")
    print(f"    Channels ({props['n_channels']})     : {ch}")
    print(f"    Recording length   : {props['n_samples']} samples  ({props['duration_s']} s)")

    # 2. Signal quality
    print(f"    NaN / Inf          : {sig['n_nan']} / {sig['n_inf']}  [{flag(sig['n_nan']==0 and sig['n_inf']==0)}]")
    print(f"    Flat channels      : {sig['flat_channels'] if sig['flat_channels'] else 'None'}  [{flag(not sig['flat_channels'])}]")
    print(f"    Saturated (>{SATURATION_UV}µV) : {sig['saturated_channels'] if sig['saturated_channels'] else 'None'}  [{flag(not sig['saturated_channels'])}]")
    for i, name in enumerate(ch):
        print(f"      {name:>4s}  range [{sig['ch_min'][i]:+8.1f}, {sig['ch_max'][i]:+8.1f}] µV  |  var {sig['ch_var'][i]:8.1f} µV²")

    # 3. Events
    if props["event_cols"]:
        print(f"    Total events       : {props['n_events']}")
        print(f"    Event table columns: {len(props['event_cols'])}")
        for col in props["event_cols"]:
            vals_str = str(col["unique_values"].tolist())
            if col["truncated"]:
                vals_str += " ..."
            print(f"      Col {col['col']}  |  {col['n_unique']} unique  |  range [{col['min']}, {col['max']}]  |  values: {vals_str}")
    else:
        print(f"    Events             : None found")

    # 4. PSD / power
    print(f"    Mains ({sig['mains_hz']} Hz) ratio : {sig['notch_ratio']}x  [{flag(not sig['needs_notch'])}]  {'→ notch filter needed' if sig['needs_notch'] else ''}")
    print(f"    Alpha peak present : {'Yes' if sig['has_alpha_peak'] else 'No'}")

    # 5. Timing
    if props["timing"]:
        tim = props["timing"]
        print(f"    Simultaneous events: {tim['n_simultaneous']}")
        print(f"    Inter-event mean   : {tim['mean_interval_ms']} ms  |  max: {tim['max_interval_ms']} ms")
        print(f"    Timing jitter (std): {tim['jitter_ms']} ms  [{flag(tim['jitter_ms'] < JITTER_MS)}]")


def subject_report_text(props, idx) -> str:
    from io import StringIO
    buf = StringIO()
    def w(s=""):
        buf.write(s + "\n")
    sig = props["signal"]
    ch = props["ch_names"]

    w(f"\n  File {idx+1}  |  Metadata: {props['metadata']}")
    w("-" * 60)
    w(f"    Sampling rate      : {props['fs']} Hz")
    w(f"    Channels ({props['n_channels']})     : {ch}")
    w(f"    Recording length   : {props['n_samples']} samples  ({props['duration_s']} s)")
    w(f"    NaN / Inf          : {sig['n_nan']} / {sig['n_inf']}  [{flag(sig['n_nan']==0 and sig['n_inf']==0)}]")
    w(f"    Flat channels      : {sig['flat_channels'] if sig['flat_channels'] else 'None'}  [{flag(not sig['flat_channels'])}]")
    w(f"    Saturated (>{SATURATION_UV}µV) : {sig['saturated_channels'] if sig['saturated_channels'] else 'None'}  [{flag(not sig['saturated_channels'])}]")
    for i, name in enumerate(ch):
        w(f"      {name:>4s}  range [{sig['ch_min'][i]:+8.1f}, {sig['ch_max'][i]:+8.1f}] µV  |  var {sig['ch_var'][i]:8.1f} µV²")
    if props["event_cols"]:
        w(f"    Total events       : {props['n_events']}")
        w(f"    Event table columns: {len(props['event_cols'])}")
        for col in props["event_cols"]:
            vals = col["unique_values"]
            vals_str = str(vals.tolist() if hasattr(vals, "tolist") else vals)
            if col["truncated"]:
                vals_str += " ..."
            w(f"      Col {col['col']}  |  {col['n_unique']} unique  |  range [{col['min']}, {col['max']}]  |  values: {vals_str}")
    else:
        w(f"    Events             : None found")
    w(f"    Mains ({sig['mains_hz']} Hz) ratio : {sig['notch_ratio']}x  [{flag(not sig['needs_notch'])}]  {'→ notch filter needed' if sig['needs_notch'] else ''}")
    w(f"    Alpha peak present : {'Yes' if sig['has_alpha_peak'] else 'No'}")
    if props["timing"]:
        tim = props["timing"]
        w(f"    Simultaneous events: {tim['n_simultaneous']}")
        w(f"    Inter-event mean   : {tim['mean_interval_ms']} ms  |  max: {tim['max_interval_ms']} ms")
        w(f"    Timing jitter (std): {tim['jitter_ms']} ms  [{flag(tim['jitter_ms'] < JITTER_MS)}]")
    return buf.getvalue()


def main():
    args = parse_args()
    files = sorted(glob.glob(os.path.join(args.data_dir, args.pattern)))
    if not files:
        print(f"No files found in {args.data_dir} matching {args.pattern}")
        return
    if args.max_files and len(files) > args.max_files:
        files = files[:args.max_files]

    print_sep()
    print("  EEG Dataset Inspection Report")
    print_sep()
    print(f"  Directory : {args.data_dir}")
    print(f"  Pattern   : {args.pattern}")
    print(f"  Files     : {len(files)}")
    print_sep()

    all_props = []
    combined_lines = []
    json_items = []
    for i, fp in enumerate(files):
        props = inspect_subject(fp, args)
        all_props.append(props)
        print_subject_report(props, i)
        if args.save_report or args.save_per_file:
            txt = subject_report_text(props, i)
            combined_lines.append(txt)
            if args.save_per_file:
                out_path = f"{fp}_inspect.txt"
                try:
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(txt)
                except Exception as e:
                    print(f"[WARN] Could not write per-file report: {out_path} ({e})")
        if args.save_json:
            # Compact per-file JSON record
            sig = props["signal"]
            tim = props.get("timing") or {}
            json_items.append({
                "path": props["metadata"].get("path", fp),
                "ext": props["metadata"].get("ext", os.path.splitext(fp)[1].lower()),
                "fs": props["fs"],
                "n_channels": props["n_channels"],
                "ch_names": props["ch_names"],
                "n_samples": props["n_samples"],
                "duration_s": props["duration_s"],
                "n_events": props["n_events"],
                "signal": {
                    "n_nan": sig["n_nan"],
                    "n_inf": sig["n_inf"],
                    "flat_channels": sig["flat_channels"],
                    "saturated_channels": sig["saturated_channels"],
                    "mains_hz": sig["mains_hz"],
                    "notch_ratio": sig["notch_ratio"],
                    "needs_notch": sig["needs_notch"],
                    "has_alpha_peak": sig["has_alpha_peak"],
                },
                "timing": tim,
                "metadata": {k: str(v) for k, v in props["metadata"].items() if k not in {"path", "ext"}},
            })

    # --- Cross-file consistency ---
    print("\n")
    print_sep()
    print("  Cross-File Consistency")
    print_sep()

    fs_vals = set(p["fs"] for p in all_props)
    ch_sets = set(tuple(p["ch_names"]) for p in all_props)
    dur_range = [p["duration_s"] for p in all_props]

    print(f"    Sampling rates     : {fs_vals}  [{flag(len(fs_vals)==1)}]")
    print(f"    Channel sets       : [{flag(len(ch_sets)==1)}]")
    print(f"    Duration range     : {min(dur_range)}–{max(dur_range)} s")

    any_nan = any(p["signal"]["n_nan"] > 0 for p in all_props)
    any_flat = any(len(p["signal"]["flat_channels"]) > 0 for p in all_props)
    any_sat = any(len(p["signal"]["saturated_channels"]) > 0 for p in all_props)
    any_notch = any(p["signal"]["needs_notch"] for p in all_props)

    cross_summary = []
    cross_summary.append(f"\n    NaN/Inf anywhere   : {'Yes' if any_nan else 'No'}")
    cross_summary.append(f"    Flat channels      : {'Yes' if any_flat else 'No'}")
    cross_summary.append(f"    Saturation         : {'Yes' if any_sat else 'No'}")
    cross_summary.append(f"    Notch filter needed: {'Yes' if any_notch else 'No'}")
    print("\n".join(cross_summary))

    # --- Discovered config ---
    print("\n")
    print_sep()
    print("  Discovered Configuration")
    print_sep()
    config_block = [
        f"    FS          = {all_props[0]['fs']}",
        f"    CH_NAMES    = {all_props[0]['ch_names']}",
        f"    N_FILES     = {len(all_props)}",
    ]
    for line in config_block:
        print(line)
    print_sep()

    # Save combined report if requested
    if args.save_report:
        try:
            with open(args.save_report, "w", encoding="utf-8") as f:
                # Header
                f.write("=" * 60 + "\n")
                f.write("  EEG Dataset Inspection Report\n")
                f.write("=" * 60 + "\n")
                f.write(f"  Directory : {args.data_dir}\n")
                f.write(f"  Pattern   : {args.pattern}\n")
                f.write(f"  Files     : {len(files)}\n")
                f.write("=" * 60 + "\n")
                # Per-file blocks
                for block in combined_lines:
                    f.write(block)
                # Cross-file
                f.write("\n" + "=" * 60 + "\n")
                f.write("  Cross-File Consistency\n")
                f.write("=" * 60 + "\n")
                f.write(f"    Sampling rates     : {fs_vals}  [{flag(len(fs_vals)==1)}]\n")
                f.write(f"    Channel sets       : [{flag(len(ch_sets)==1)}]\n")
                f.write(f"    Duration range     : {min(dur_range)}–{max(dur_range)} s\n\n")
                f.write("\n".join(cross_summary) + "\n\n")
                # Config
                f.write("=" * 60 + "\n")
                f.write("  Discovered Configuration\n")
                f.write("=" * 60 + "\n")
                for line in config_block:
                    f.write(line + "\n")
                f.write("=" * 60 + "\n")
            print(f"Saved combined report to: {args.save_report}")
        except Exception as e:
            print(f"[WARN] Could not write combined report: {args.save_report} ({e})")
    if args.save_json:
        try:
            import json
            payload = {
                "data_dir": args.data_dir,
                "pattern": args.pattern,
                "n_files": len(files),
                "fs_set": sorted(list(fs_vals)),
                "channel_sets_equal": len(ch_sets) == 1,
                "duration_range": [min(dur_range), max(dur_range)],
                "flags": {
                    "any_nan": any_nan,
                    "any_flat": any_flat,
                    "any_saturation": any_sat,
                    "any_notch_needed": any_notch,
                },
                "files": json_items,
            }
            with open(args.save_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"Saved JSON summary to: {args.save_json}")
        except Exception as e:
            print(f"[WARN] Could not write JSON summary: {args.save_json} ({e})")


if __name__ == "__main__":
    main()
