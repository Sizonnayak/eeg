"""
P300 EEG Data - Subject-Specific Preprocessing and EDA
========================================================
Customized filtering pipeline for each subject based on data quality:
  - Clean subjects (S01, S06, S07, S08): Conservative filtering (0.5-20 Hz)
  - Noisy subject (S02): Standard filtering (1-30 Hz)
  - Artifact subjects (S03, S04, S05): Aggressive filtering + artifact handling

Requirements: pip install numpy scipy matplotlib mne
Usage:        python p300_eda_subject_specific.py
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.signal import freqz, butter
import os
import warnings

warnings.filterwarnings("ignore")
mne.set_log_level("WARNING")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = "/Users/siznayak/Documents/others/MTech/Dataset/p300_8subject"
BASE_PLOT_DIR = "/Users/siznayak/Documents/others/MTech/EEG_Classification/analysis/plots"

FS = 250  # Sampling rate (Hz)
PRE_STIM_MS = 200   # Pre-stimulus window (ms)
POST_STIM_MS = 800  # Post-stimulus window (ms)
PRE_STIM_SAMPLES = int(PRE_STIM_MS * FS / 1000)       # 50
POST_STIM_SAMPLES = int(POST_STIM_MS * FS / 1000)     # 200
EPOCH_SAMPLES = PRE_STIM_SAMPLES + POST_STIM_SAMPLES  # 250
TIME_AXIS_MS = np.arange(EPOCH_SAMPLES) / FS * 1000 - PRE_STIM_MS
TIME_AXIS_S = TIME_AXIS_MS / 1000.0

# MNE channel names (standard_1020 compatible)
CH_NAMES_MNE = ["Fz", "Cz", "P3", "Pz", "P4", "PO7", "PO8", "Oz"]

# Colors
COLOR_TARGET = "#E74C3C"      # red
COLOR_NONTARGET = "#3498DB"   # blue

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})

# ---------------------------------------------------------------------------
# Subject-Specific Configuration
# ---------------------------------------------------------------------------
SUBJECT_CONFIG = {
    1: {
        "file": "P300S01.mat",
        "folder": "S01_clean",
        "quality": "Clean",
        "notch": False,  # No line noise detected
        "highpass": 0.5,  # Conservative
        "lowpass": 20,    # Focus on P300 range
        "bad_channels": [],
        "artifact_threshold": 150,  # µV
        "description": "High quality data, conservative filtering"
    },
    2: {
        "file": "P300S02.mat",
        "folder": "S02_noisy",
        "quality": "Noisy",
        "notch": False,
        "highpass": 1.0,   # Standard
        "lowpass": 30,     # Keep more bandwidth for noisy data
        "bad_channels": [],
        "artifact_threshold": 200,
        "description": "High variance in P3 channel, standard filtering"
    },
    3: {
        "file": "P300S03.mat",
        "folder": "S03_artifacts",
        "quality": "Severe Artifacts",
        "notch": False,
        "highpass": 1.0,
        "lowpass": 20,     # More restrictive
        "bad_channels": ["P4", "PO8"],  # Saturated channels
        "artifact_threshold": 100,  # Stricter rejection
        "description": "Severe artifacts in P4 and PO8, will interpolate bad channels"
    },
    4: {
        "file": "P300S04.mat",
        "folder": "S04_artifacts",
        "quality": "Artifacts",
        "notch": False,
        "highpass": 1.0,
        "lowpass": 20,
        "bad_channels": ["PO8"],  # Saturated channel
        "artifact_threshold": 100,
        "description": "Artifacts in PO8, will interpolate bad channel"
    },
    5: {
        "file": "P300S05.mat",
        "folder": "S05_artifacts",
        "quality": "Artifacts",
        "notch": False,
        "highpass": 1.0,
        "lowpass": 20,
        "bad_channels": ["Oz"],  # Saturated channel
        "artifact_threshold": 100,
        "description": "Artifacts in Oz, will interpolate bad channel"
    },
    6: {
        "file": "P300S06.mat",
        "folder": "S06_clean",
        "quality": "Clean",
        "notch": False,
        "highpass": 0.5,
        "lowpass": 20,
        "bad_channels": [],
        "artifact_threshold": 150,
        "description": "High quality data, conservative filtering"
    },
    7: {
        "file": "P300S07.mat",
        "folder": "S07_clean",
        "quality": "Clean",
        "notch": False,
        "highpass": 0.5,
        "lowpass": 20,
        "bad_channels": [],
        "artifact_threshold": 150,
        "description": "High quality data, conservative filtering"
    },
    8: {
        "file": "P300S08.mat",
        "folder": "S08_clean",
        "quality": "Clean",
        "notch": False,
        "highpass": 0.5,
        "lowpass": 20,
        "bad_channels": [],
        "artifact_threshold": 150,
        "description": "High quality data, conservative filtering"
    }
}

# ===========================================================================
# Data Loading & Filtering
# ===========================================================================

def load_subject(filepath):
    """Load a single MAT file and return structured dict."""
    mat = sio.loadmat(filepath, squeeze_me=True)
    data = mat["data"]
    return {
        "X": data["X"].item(),
        "flash": data["flash"].item(),
        "Fs": int(mat["Fs"]),
        "ch_names": [str(c) for c in mat["channelNames"]],
        "subject_id": int(mat["subject"]),
        "word": str(mat["Word"]),
    }


def filter_eeg_custom(X, fs, config):
    """
    Apply custom filtering based on subject configuration.
    Optionally interpolate bad channels before filtering.
    """
    info = mne.create_info(ch_names=CH_NAMES_MNE, sfreq=fs, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage)

    raw = mne.io.RawArray(X.T, info, verbose=False)

    # Mark bad channels if any
    if config["bad_channels"]:
        raw.info["bads"] = config["bad_channels"]
        print(f"    Marked bad channels: {config['bad_channels']}")
        # Interpolate bad channels
        raw.interpolate_bads(reset_bads=True, verbose=False)
        print(f"    Interpolated bad channels")

    # Apply notch filter if configured
    if config["notch"]:
        raw.notch_filter(freqs=50, verbose=False)
        print(f"    Applied notch filter at 50 Hz")

    # Apply bandpass filter
    raw.filter(l_freq=config["highpass"], h_freq=config["lowpass"], verbose=False)
    print(f"    Applied bandpass: {config['highpass']}-{config['lowpass']} Hz")

    return raw.get_data().T  # back to (n_samples, n_ch)


def butter_bandpass(lowcut, highcut, fs, order=5):
    """Butterworth bandpass filter coefficients."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


# ===========================================================================
# Epoch Extraction with Baseline Correction
# ===========================================================================

def extract_epochs(X, flash, artifact_threshold=150):
    """
    Extract epochs with baseline correction and artifact rejection.

    Args:
        X: Filtered EEG data (n_samples, n_ch)
        flash: Event table
        artifact_threshold: Maximum allowed amplitude in µV

    Returns:
        epochs_target, epochs_nontarget, n_rejected
    """
    n_samples = X.shape[0]
    epochs_target = []
    epochs_nontarget = []
    n_rejected = 0

    for i in range(flash.shape[0]):
        onset = int(flash[i, 0])
        start = onset - PRE_STIM_SAMPLES
        end = onset + POST_STIM_SAMPLES

        if start < 0 or end > n_samples:
            continue

        epoch = X[start:end, :].copy()

        # Artifact rejection: check if max amplitude exceeds threshold
        if np.abs(epoch).max() > artifact_threshold:
            n_rejected += 1
            continue

        # Baseline correction
        baseline = np.mean(epoch[:PRE_STIM_SAMPLES, :], axis=0)
        epoch = epoch - baseline

        is_target = (flash[i, 3] == 1)
        if is_target:
            epochs_target.append(epoch)
        else:
            epochs_nontarget.append(epoch)

    return np.array(epochs_target), np.array(epochs_nontarget), n_rejected


# ===========================================================================
# Plotting Functions (same as original)
# ===========================================================================

def plot_butterfly_erp(epochs_target, epochs_nontarget, subj_id, ch_names, ax):
    """Butterfly plot with P300 zone."""
    erp_target = epochs_target.mean(axis=0)
    erp_nontarget = epochs_nontarget.mean(axis=0)

    for ch in range(len(ch_names)):
        ax.plot(TIME_AXIS_S, erp_target[:, ch], color=COLOR_TARGET,
                alpha=0.35, linewidth=0.7)
        ax.plot(TIME_AXIS_S, erp_nontarget[:, ch], color=COLOR_NONTARGET,
                alpha=0.25, linewidth=0.7, linestyle="--")

    ax.plot(TIME_AXIS_S, erp_target.mean(axis=1), color=COLOR_TARGET,
            linewidth=2.5, label=f"Target (n={len(epochs_target)})")
    ax.plot(TIME_AXIS_S, erp_nontarget.mean(axis=1), color=COLOR_NONTARGET,
            linewidth=2.5, linestyle="--", label=f"Non-Target (n={len(epochs_nontarget)})")

    ax.axvspan(0.2, 0.5, color="grey", alpha=0.2, label="P300 zone (200-500 ms)")
    ax.axvline(0, color="gray", linewidth=1.0, linestyle=":")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(f"S{subj_id:02d} – Butterfly ERP")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_xlim(-0.2, 0.8)


def plot_topo_maps_mne(epochs_target, epochs_nontarget, subj_id, plot_dir):
    """Topographic maps at key timepoints."""
    info = mne.create_info(ch_names=CH_NAMES_MNE, sfreq=FS, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage)

    erp_target = epochs_target.mean(axis=0)
    erp_nontarget = epochs_nontarget.mean(axis=0)
    erp_diff = erp_target - erp_nontarget

    time_points_s = [0.0, 0.35, 0.6]
    time_labels = ["0 ms", "350 ms (P300)", "600 ms"]

    fig, axes = plt.subplots(3, 3, figsize=(12, 11))
    fig.suptitle(f"S{subj_id:02d} – Topographic Maps (Target / Non-Target / Difference)",
                 fontsize=12, fontweight="bold")

    evoked_tgt = mne.EvokedArray(erp_target.T * 1e-6, info, tmin=-0.2)
    evoked_nontgt = mne.EvokedArray(erp_nontarget.T * 1e-6, info, tmin=-0.2)
    evoked_diff = mne.EvokedArray(erp_diff.T * 1e-6, info, tmin=-0.2)

    for col, (evoked, col_title) in enumerate(
        [(evoked_tgt, "Target"), (evoked_nontgt, "Non-Target"), (evoked_diff, "Difference")]
    ):
        for row, (t_s, t_label) in enumerate(zip(time_points_s, time_labels)):
            ax = axes[row, col]
            mne.viz.plot_topomap(
                evoked.get_data()[:, evoked.time_as_index(t_s)[0]],
                info, axes=ax, show=False,
                cmap="RdBu_r", extrapolate="local"
            )
            ax.set_title(f"{col_title} @ {t_label}", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(plot_dir, f"S{subj_id:02d}_topo_maps.png"), dpi=150)
    plt.close(fig)


def compute_peak_stats(epochs_target, ch_names):
    """Compute P300 peak statistics per channel."""
    erp_target = epochs_target.mean(axis=0)

    results = []
    for ch_idx, ch_name in enumerate(ch_names):
        erp_ch = erp_target[:, ch_idx]
        post_stim_erp = erp_ch[PRE_STIM_SAMPLES:]

        # P300 search window: 200-500 ms (consistent with visualization)
        p300_start = int(200 * FS / 1000)  # 50 samples
        p300_end = int(500 * FS / 1000)    # 125 samples
        p300_window = post_stim_erp[p300_start:p300_end]

        peak_idx = np.argmax(p300_window)
        peak_latency_ms = 200 + peak_idx * 1000 / FS
        peak_amplitude = p300_window[peak_idx]

        baseline = erp_ch[:PRE_STIM_SAMPLES]
        baseline_std = np.std(baseline)
        snr = peak_amplitude / baseline_std if baseline_std > 0 else 0.0

        results.append({
            "Channel": ch_name,
            "Peak_Latency_ms": round(peak_latency_ms, 1),
            "Peak_Amp_uV": round(float(peak_amplitude), 2),
            "SNR": round(float(snr), 2),
        })

    return sorted(results, key=lambda x: -x["SNR"])


# ===========================================================================
# Main Processing Function
# ===========================================================================

def process_subject(subj_id, config):
    """Process one subject with custom configuration."""
    filepath = os.path.join(DATA_DIR, config["file"])
    plot_dir = os.path.join(BASE_PLOT_DIR, config["folder"])
    os.makedirs(plot_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  Processing Subject {subj_id} ({config['file']})")
    print(f"  Quality: {config['quality']}")
    print(f"  Description: {config['description']}")
    print(f"{'=' * 70}")

    # Load data
    subj_data = load_subject(filepath)
    X_raw = subj_data["X"]
    flash = subj_data["flash"]

    # Apply custom filtering
    print(f"  Filtering pipeline:")
    X_filtered = filter_eeg_custom(X_raw, FS, config)

    # Extract epochs with artifact rejection
    print(f"  Extracting epochs (artifact threshold: {config['artifact_threshold']} µV)...")
    epochs_target, epochs_nontarget, n_rejected = extract_epochs(
        X_filtered, flash, config["artifact_threshold"]
    )

    print(f"  Epochs: {len(epochs_target)} target, {len(epochs_nontarget)} non-target")
    print(f"  Rejected: {n_rejected} epochs due to artifacts")

    # Generate plots
    print(f"  Generating plots...")

    # 1. Butterfly ERP
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 5))
    plot_butterfly_erp(epochs_target, epochs_nontarget, subj_id, CH_NAMES_MNE, ax1)
    fig1.tight_layout()
    fig1.savefig(os.path.join(plot_dir, f"S{subj_id:02d}_butterfly_erp.png"), dpi=150)
    plt.close(fig1)

    # 2. Topographic maps
    plot_topo_maps_mne(epochs_target, epochs_nontarget, subj_id, plot_dir)

    # 3. Compute peak stats
    stats = compute_peak_stats(epochs_target, CH_NAMES_MNE)

    # Save configuration and results
    config_path = os.path.join(plot_dir, f"S{subj_id:02d}_config.txt")
    with open(config_path, "w") as f:
        f.write(f"Subject {subj_id} Preprocessing Configuration\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Quality Assessment: {config['quality']}\n")
        f.write(f"Description: {config['description']}\n\n")
        f.write("Filtering Parameters:\n")
        f.write(f"  Notch filter (50 Hz): {'Yes' if config['notch'] else 'No'}\n")
        f.write(f"  Highpass: {config['highpass']} Hz\n")
        f.write(f"  Lowpass: {config['lowpass']} Hz\n\n")
        f.write(f"Bad channels (interpolated): {config['bad_channels'] if config['bad_channels'] else 'None'}\n")
        f.write(f"Artifact threshold: {config['artifact_threshold']} µV\n\n")
        f.write("Results:\n")
        f.write(f"  Target epochs: {len(epochs_target)}\n")
        f.write(f"  Non-target epochs: {len(epochs_nontarget)}\n")
        f.write(f"  Rejected epochs: {n_rejected}\n\n")
        f.write("P300 Peak Stats:\n")
        f.write(f"  {'Channel':<8} {'Peak Latency (ms)':<20} {'Peak Amp (µV)':<16} {'SNR':<8}\n")
        f.write(f"  {'-'*8} {'-'*20} {'-'*16} {'-'*8}\n")
        for row in stats:
            f.write(f"  {row['Channel']:<8} {row['Peak_Latency_ms']:<20.1f} {row['Peak_Amp_uV']:<16.2f} {row['SNR']:<8.2f}\n")
        f.write(f"\nBest P300: {stats[0]['Channel']} | {stats[0]['Peak_Latency_ms']}ms | SNR {stats[0]['SNR']:.2f}\n")

    print(f"  Subject {subj_id} complete. Saved to {plot_dir}")
    return stats


def main():
    """Process all subjects with custom configurations."""
    print("=" * 70)
    print(" P300 EEG – Subject-Specific Preprocessing and EDA")
    print("=" * 70)

    all_stats = {}
    for subj_id, config in SUBJECT_CONFIG.items():
        stats = process_subject(subj_id, config)
        all_stats[subj_id] = stats

    # Summary
    print("\n" + "=" * 70)
    print(" Processing Complete!")
    print("=" * 70)
    print(f"\nResults saved in subject-specific folders under:")
    print(f"  {BASE_PLOT_DIR}/")
    print("\nSubject-specific folders:")
    for subj_id, config in SUBJECT_CONFIG.items():
        print(f"  S{subj_id:02d}: {config['folder']} ({config['quality']})")
    print("\nDone!")


if __name__ == "__main__":
    main()
