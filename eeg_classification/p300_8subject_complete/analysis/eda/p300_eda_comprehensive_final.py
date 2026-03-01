"""
P300 EEG Data - Comprehensive Subject-Specific Preprocessing and EDA
=====================================================================
Complete visualization suite combining:
1. Current plots: Butterfly ERP, Topographic maps, Config files
2. Additional plots:
   - All-channel butterfly overlay
   - Difference waves (Target - Non-target) per channel
   - Signal quality metrics (SNR, baseline statistics)
   - Time-frequency analysis
   - Channel-wise P300 amplitude comparison

Requirements: pip install numpy scipy matplotlib mne
Usage:        python p300_eda_comprehensive.py
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import mne
from scipy.signal import freqz, butter, spectrogram
import os
import warnings

warnings.filterwarnings("ignore")
mne.set_log_level("WARNING")
sns.set_style('whitegrid')

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
COLOR_RAW = "#95A5A6"         # gray

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})

# Subject-Specific Configuration (same as before)
SUBJECT_CONFIG = {
    1: {"file": "P300S01.mat", "folder": "S01_clean", "quality": "Clean", "notch": False, "highpass": 0.5, "lowpass": 20, "bad_channels": [], "artifact_threshold": 150, "description": "High quality data, conservative filtering"},
    2: {"file": "P300S02.mat", "folder": "S02_noisy", "quality": "Noisy", "notch": False, "highpass": 1.0, "lowpass": 30, "bad_channels": [], "artifact_threshold": 200, "description": "High variance in P3 channel, standard filtering"},
    3: {"file": "P300S03.mat", "folder": "S03_artifacts", "quality": "Severe Artifacts", "notch": False, "highpass": 1.0, "lowpass": 20, "bad_channels": ["P4", "PO8"], "artifact_threshold": 100, "description": "Severe artifacts in P4 and PO8, will interpolate bad channels"},
    4: {"file": "P300S04.mat", "folder": "S04_artifacts", "quality": "Artifacts", "notch": False, "highpass": 1.0, "lowpass": 20, "bad_channels": ["PO8"], "artifact_threshold": 100, "description": "Artifacts in PO8, will interpolate bad channel"},
    5: {"file": "P300S05.mat", "folder": "S05_artifacts", "quality": "Artifacts", "notch": False, "highpass": 1.0, "lowpass": 20, "bad_channels": ["Oz"], "artifact_threshold": 100, "description": "Artifacts in Oz, will interpolate bad channel"},
    6: {"file": "P300S06.mat", "folder": "S06_clean", "quality": "Clean", "notch": False, "highpass": 0.5, "lowpass": 20, "bad_channels": [], "artifact_threshold": 150, "description": "High quality data, conservative filtering"},
    7: {"file": "P300S07.mat", "folder": "S07_clean", "quality": "Clean", "notch": False, "highpass": 0.5, "lowpass": 20, "bad_channels": [], "artifact_threshold": 150, "description": "High quality data, conservative filtering"},
    8: {"file": "P300S08.mat", "folder": "S08_clean", "quality": "Clean", "notch": False, "highpass": 0.5, "lowpass": 20, "bad_channels": [], "artifact_threshold": 150, "description": "High quality data, conservative filtering"}
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
    """Apply custom filtering based on subject configuration."""
    info = mne.create_info(ch_names=CH_NAMES_MNE, sfreq=fs, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage)

    raw = mne.io.RawArray(X.T, info, verbose=False)

    if config["bad_channels"]:
        raw.info["bads"] = config["bad_channels"]
        print(f"    Marked bad channels: {config['bad_channels']}")
        raw.interpolate_bads(reset_bads=True, verbose=False)
        print(f"    Interpolated bad channels")

    if config["notch"]:
        raw.notch_filter(freqs=50, verbose=False)
        print(f"    Applied notch filter at 50 Hz")

    raw.filter(l_freq=config["highpass"], h_freq=config["lowpass"], verbose=False)
    print(f"    Applied bandpass: {config['highpass']}-{config['lowpass']} Hz")

    return raw.get_data().T


def extract_epochs(X, flash, artifact_threshold=150):
    """Extract epochs with baseline correction and artifact rejection."""
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

        if np.abs(epoch).max() > artifact_threshold:
            n_rejected += 1
            continue

        baseline = np.mean(epoch[:PRE_STIM_SAMPLES, :], axis=0)
        epoch = epoch - baseline

        is_target = (flash[i, 3] == 1)
        if is_target:
            epochs_target.append(epoch)
        else:
            epochs_nontarget.append(epoch)

    return np.array(epochs_target), np.array(epochs_nontarget), n_rejected


def extract_raw_epochs(X, flash, artifact_threshold=150):
    """Extract RAW epochs without filtering but with artifact rejection."""
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

        if np.abs(epoch).max() > artifact_threshold:
            n_rejected += 1
            continue

        baseline = np.mean(epoch[:PRE_STIM_SAMPLES, :], axis=0)
        epoch = epoch - baseline

        is_target = (flash[i, 3] == 1)
        if is_target:
            epochs_target.append(epoch)
        else:
            epochs_nontarget.append(epoch)

    return np.array(epochs_target), np.array(epochs_nontarget), n_rejected


# ===========================================================================
# NEW PLOT 1: All-Channel Butterfly Overlay
# ===========================================================================

def plot_butterfly_overlay(epochs_target_raw, epochs_nontarget_raw, epochs_target, epochs_nontarget, subj_id, plot_dir):
    """Butterfly plot showing all channels overlaid (like butterfly wings)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"S{subj_id:02d} – All-Channel Butterfly Overlay", fontsize=14, fontweight="bold")

    datasets = [
        (epochs_target_raw, "RAW: Target", axes[0, 0]),
        (epochs_nontarget_raw, "RAW: Non-target", axes[0, 1]),
        (epochs_target, "PREPROCESSED: Target", axes[1, 0]),
        (epochs_nontarget, "PREPROCESSED: Non-target", axes[1, 1]),
    ]

    for epochs, title, ax in datasets:
        erp = epochs.mean(axis=0)  # (250, 8)

        # Plot all channels in light gray
        for ch in range(erp.shape[1]):
            ax.plot(TIME_AXIS_S, erp[:, ch], color='gray', alpha=0.3, linewidth=0.5)

        # Highlight key P300 channels
        key_channels = {'Pz': 3, 'Cz': 1, 'Oz': 7}  # indices
        colors = {'Pz': 'red', 'Cz': 'blue', 'Oz': 'green'}

        for ch_name, ch_idx in key_channels.items():
            ax.plot(TIME_AXIS_S, erp[:, ch_idx], color=colors[ch_name],
                   linewidth=2, label=ch_name, alpha=0.8)

        # P300 zone
        ax.axvspan(0.2, 0.5, color='yellow', alpha=0.2, label='P300 zone')
        ax.axvline(0, color='black', linestyle=':', linewidth=1)
        ax.axvline(0.3, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Expected P300')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (µV)')
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.set_xlim(-0.2, 0.8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"S{subj_id:02d}_butterfly_overlay.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# NEW PLOT 2: Difference Waves (Target - Non-target)
# ===========================================================================

def plot_difference_waves(epochs_target_raw, epochs_nontarget_raw, epochs_target, epochs_nontarget, subj_id, plot_dir):
    """Plot difference waves for key channels."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"S{subj_id:02d} – Difference Waves (Target - Non-target)", fontsize=14, fontweight="bold")

    erp_target_raw = epochs_target_raw.mean(axis=0)
    erp_nontarget_raw = epochs_nontarget_raw.mean(axis=0)
    diff_raw = erp_target_raw - erp_nontarget_raw

    erp_target = epochs_target.mean(axis=0)
    erp_nontarget = epochs_nontarget.mean(axis=0)
    diff_preprocessed = erp_target - erp_nontarget

    axes = axes.flatten()

    for ch_idx, ch_name in enumerate(CH_NAMES_MNE):
        ax = axes[ch_idx]

        # Plot difference waves
        ax.plot(TIME_AXIS_S, diff_raw[:, ch_idx], color=COLOR_RAW,
               linewidth=1.5, label='Raw', alpha=0.7)
        ax.plot(TIME_AXIS_S, diff_preprocessed[:, ch_idx], color=COLOR_TARGET,
               linewidth=2, label='Preprocessed')

        # Shade positive areas (P300 effect)
        ax.fill_between(TIME_AXIS_S, 0, diff_preprocessed[:, ch_idx],
                       where=(diff_preprocessed[:, ch_idx] > 0),
                       color='green', alpha=0.2, label='Positive effect')

        # Zero line and P300 zone
        ax.axhline(0, color='black', linestyle=':', linewidth=1)
        ax.axvspan(0.2, 0.5, color='yellow', alpha=0.1)
        ax.axvline(0, color='black', linestyle=':', linewidth=0.8)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Difference (µV)')
        ax.set_title(f'{ch_name}')
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.2, 0.8)

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"S{subj_id:02d}_difference_waves.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# NEW PLOT 3: Signal Quality Metrics
# ===========================================================================

def compute_snr(epochs, baseline_period=(0, PRE_STIM_SAMPLES), signal_period=(PRE_STIM_SAMPLES, PRE_STIM_SAMPLES+75)):
    """Compute SNR per channel."""
    snr_list = []
    for ch in range(epochs.shape[2]):
        baseline = epochs[:, baseline_period[0]:baseline_period[1], ch].flatten()
        signal = epochs[:, signal_period[0]:signal_period[1], ch].flatten()

        baseline_std = np.std(baseline)
        signal_power = np.mean(signal ** 2)
        noise_power = baseline_std ** 2

        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0
        snr_list.append(snr)

    return np.array(snr_list)


def plot_signal_quality(epochs_target_raw, epochs_target, subj_id, plot_dir):
    """Plot signal quality metrics: SNR, baseline noise, amplitude ranges."""
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig)
    fig.suptitle(f"S{subj_id:02d} – Signal Quality Metrics", fontsize=14, fontweight="bold")

    # SNR comparison
    ax1 = fig.add_subplot(gs[0, :])
    snr_raw = compute_snr(epochs_target_raw)
    snr_preprocessed = compute_snr(epochs_target)

    x = np.arange(len(CH_NAMES_MNE))
    width = 0.35
    ax1.bar(x - width/2, snr_raw, width, label='Raw', color=COLOR_RAW, alpha=0.7)
    ax1.bar(x + width/2, snr_preprocessed, width, label='Preprocessed', color=COLOR_TARGET, alpha=0.7)
    ax1.set_xlabel('Channel')
    ax1.set_ylabel('SNR (dB)')
    ax1.set_title('Signal-to-Noise Ratio per Channel')
    ax1.set_xticks(x)
    ax1.set_xticklabels(CH_NAMES_MNE)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Baseline noise
    ax2 = fig.add_subplot(gs[1, 0])
    baseline_raw = np.std(epochs_target_raw[:, :PRE_STIM_SAMPLES, :], axis=(0, 1))
    baseline_preprocessed = np.std(epochs_target[:, :PRE_STIM_SAMPLES, :], axis=(0, 1))

    ax2.bar(x - width/2, baseline_raw, width, label='Raw', color=COLOR_RAW, alpha=0.7)
    ax2.bar(x + width/2, baseline_preprocessed, width, label='Preprocessed', color=COLOR_TARGET, alpha=0.7)
    ax2.set_xlabel('Channel')
    ax2.set_ylabel('Baseline Std Dev (µV)')
    ax2.set_title('Baseline Noise Level')
    ax2.set_xticks(x)
    ax2.set_xticklabels(CH_NAMES_MNE, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Amplitude range
    ax3 = fig.add_subplot(gs[1, 1])
    range_raw = np.ptp(epochs_target_raw, axis=1).mean(axis=0)
    range_preprocessed = np.ptp(epochs_target, axis=1).mean(axis=0)

    ax3.bar(x - width/2, range_raw, width, label='Raw', color=COLOR_RAW, alpha=0.7)
    ax3.bar(x + width/2, range_preprocessed, width, label='Preprocessed', color=COLOR_TARGET, alpha=0.7)
    ax3.set_xlabel('Channel')
    ax3.set_ylabel('Mean Peak-to-Peak (µV)')
    ax3.set_title('Average Amplitude Range')
    ax3.set_xticks(x)
    ax3.set_xticklabels(CH_NAMES_MNE, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Variance across trials
    ax4 = fig.add_subplot(gs[2, :])
    var_raw = np.var(epochs_target_raw, axis=0).mean(axis=0)
    var_preprocessed = np.var(epochs_target, axis=0).mean(axis=0)

    ax4.bar(x - width/2, var_raw, width, label='Raw', color=COLOR_RAW, alpha=0.7)
    ax4.bar(x + width/2, var_preprocessed, width, label='Preprocessed', color=COLOR_TARGET, alpha=0.7)
    ax4.set_xlabel('Channel')
    ax4.set_ylabel('Variance (µV²)')
    ax4.set_title('Trial-to-Trial Variability')
    ax4.set_xticks(x)
    ax4.set_xticklabels(CH_NAMES_MNE)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"S{subj_id:02d}_signal_quality.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# NEW PLOT 4: Time-Frequency Analysis
# ===========================================================================

def plot_time_frequency(epochs_target, epochs_nontarget, subj_id, plot_dir):
    """Time-frequency analysis for key channels."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"S{subj_id:02d} – Time-Frequency Analysis (Target epochs)", fontsize=14, fontweight="bold")

    key_channels = [(3, 'Pz'), (1, 'Cz'), (7, 'Oz'), (2, 'P3'), (4, 'P4'), (5, 'PO7')]

    for idx, (ch_idx, ch_name) in enumerate(key_channels):
        ax = axes.flatten()[idx]

        # Average across trials for this channel
        data = epochs_target[:, :, ch_idx].mean(axis=0)

        # Compute spectrogram
        f, t, Sxx = spectrogram(data, fs=FS, nperseg=32, noverlap=28)

        # Plot
        im = ax.pcolormesh(t * 1000 - PRE_STIM_MS, f, 10 * np.log10(Sxx + 1e-10),
                          shading='gouraud', cmap='jet')
        ax.set_ylim([0, 40])  # Focus on 0-40 Hz
        ax.axvline(0, color='white', linestyle='--', linewidth=1)
        ax.axvline(300, color='white', linestyle='--', linewidth=1)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(f'{ch_name}')
        plt.colorbar(im, ax=ax, label='Power (dB)')

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"S{subj_id:02d}_time_frequency.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# NEW PLOT 5: Channel P300 Amplitude Comparison
# ===========================================================================

def plot_channel_p300_amplitude(epochs_target, epochs_nontarget, subj_id, plot_dir):
    """Bar chart comparing P300 amplitude across channels."""
    erp_target = epochs_target.mean(axis=0)
    erp_nontarget = epochs_nontarget.mean(axis=0)

    # Find peak in P300 window (200-500 ms)
    p300_start = PRE_STIM_SAMPLES + int(200 * FS / 1000)
    p300_end = PRE_STIM_SAMPLES + int(500 * FS / 1000)

    peak_target = erp_target[p300_start:p300_end, :].max(axis=0)
    peak_nontarget = erp_nontarget[p300_start:p300_end, :].max(axis=0)
    difference = peak_target - peak_nontarget

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"S{subj_id:02d} – P300 Peak Amplitude per Channel", fontsize=14, fontweight="bold")

    x = np.arange(len(CH_NAMES_MNE))
    width = 0.35

    # Plot 1: Target vs Non-target
    ax1.bar(x - width/2, peak_target, width, label='Target', color=COLOR_TARGET, alpha=0.7)
    ax1.bar(x + width/2, peak_nontarget, width, label='Non-target', color=COLOR_NONTARGET, alpha=0.7)
    ax1.set_xlabel('Channel')
    ax1.set_ylabel('Peak Amplitude (µV)')
    ax1.set_title('Peak Amplitude in P300 Window (200-500 ms)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(CH_NAMES_MNE)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Difference (P300 effect)
    bars = ax2.bar(x, difference, color=['green' if d > 0 else 'red' for d in difference], alpha=0.7)
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Channel')
    ax2.set_ylabel('Difference (Target - Non-target) µV')
    ax2.set_title('P300 Effect Size per Channel')
    ax2.set_xticks(x)
    ax2.set_xticklabels(CH_NAMES_MNE)
    ax2.grid(True, alpha=0.3)

    # Annotate with values
    for i, (bar, val) in enumerate(zip(bars, difference)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"S{subj_id:02d}_channel_p300_amplitude.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# Existing Plots (keep from previous version)
# ===========================================================================

def plot_butterfly_erp(epochs_target, epochs_nontarget, subj_id, ch_names, ax):
    """Original butterfly plot."""
    erp_target = epochs_target.mean(axis=0)
    erp_nontarget = epochs_nontarget.mean(axis=0)

    for ch in range(len(ch_names)):
        ax.plot(TIME_AXIS_S, erp_target[:, ch], color=COLOR_TARGET, alpha=0.35, linewidth=0.7)
        ax.plot(TIME_AXIS_S, erp_nontarget[:, ch], color=COLOR_NONTARGET, alpha=0.25, linewidth=0.7, linestyle="--")

    ax.plot(TIME_AXIS_S, erp_target.mean(axis=1), color=COLOR_TARGET, linewidth=2.5, label=f"Target (n={len(epochs_target)})")
    ax.plot(TIME_AXIS_S, erp_nontarget.mean(axis=1), color=COLOR_NONTARGET, linewidth=2.5, linestyle="--", label=f"Non-Target (n={len(epochs_nontarget)})")

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

        p300_start = int(200 * FS / 1000)
        p300_end = int(500 * FS / 1000)
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


def plot_peak_stats_table(stats, subj_id, plot_dir):
    """Render P300 peak stats as a styled table (top 5 channels by SNR)."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.axis("off")
    stats_top5 = stats[:5]

    columns = ["Channel", "Peak Latency (ms)", "Peak Amp (µV)", "SNR"]
    cell_text = [[row["Channel"], f"{row['Peak_Latency_ms']:.1f}",
                  f"{row['Peak_Amp_uV']:.2f}", f"{row['SNR']:.2f}"]
                 for row in stats_top5]

    table = ax.table(cellText=cell_text, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.6)

    for j in range(len(columns)):
        table[(0, j)].set_facecolor("#34495E")
        table[(0, j)].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(cell_text) + 1):
        for j in range(len(columns)):
            table[(i, j)].set_facecolor("#ECF0F1" if i % 2 == 0 else "white")

    ax.set_title(f"S{subj_id:02d} – P300 Peak Stats (Top 5 by SNR)", fontsize=10, pad=20)

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"S{subj_id:02d}_peak_stats_table.png"), dpi=150)
    plt.close(fig)


# ===========================================================================
# Main Processing Function
# ===========================================================================

def process_subject(subj_id, config):
    """Process one subject with comprehensive plots."""
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
    X_raw_continuous = subj_data["X"]
    flash = subj_data["flash"]

    # Extract RAW epochs (before filtering)
    print(f"  Extracting raw epochs...")
    epochs_target_raw, epochs_nontarget_raw, n_rejected_raw = extract_raw_epochs(
        X_raw_continuous, flash, config["artifact_threshold"]
    )
    print(f"    Raw: {len(epochs_target_raw)} target, {len(epochs_nontarget_raw)} non-target")

    # Apply custom filtering
    print(f"  Filtering pipeline:")
    X_filtered = filter_eeg_custom(X_raw_continuous, FS, config)

    # Extract PREPROCESSED epochs
    print(f"  Extracting preprocessed epochs...")
    epochs_target, epochs_nontarget, n_rejected = extract_epochs(
        X_filtered, flash, config["artifact_threshold"]
    )
    print(f"    Preprocessed: {len(epochs_target)} target, {len(epochs_nontarget)} non-target")
    print(f"    Rejected: {n_rejected} epochs due to artifacts")

    # Generate ALL plots
    print(f"  Generating comprehensive plots...")

    # Original plots
    print(f"    -> Butterfly ERP...")
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 5))
    plot_butterfly_erp(epochs_target, epochs_nontarget, subj_id, CH_NAMES_MNE, ax1)
    fig1.tight_layout()
    fig1.savefig(os.path.join(plot_dir, f"S{subj_id:02d}_butterfly_erp.png"), dpi=150)
    plt.close(fig1)

    print(f"    -> Topographic maps...")
    plot_topo_maps_mne(epochs_target, epochs_nontarget, subj_id, plot_dir)

    # NEW comprehensive plots
    print(f"    -> All-channel butterfly overlay...")
    plot_butterfly_overlay(epochs_target_raw, epochs_nontarget_raw, epochs_target, epochs_nontarget, subj_id, plot_dir)

    print(f"    -> Difference waves...")
    plot_difference_waves(epochs_target_raw, epochs_nontarget_raw, epochs_target, epochs_nontarget, subj_id, plot_dir)

    print(f"    -> Signal quality metrics...")
    plot_signal_quality(epochs_target_raw, epochs_target, subj_id, plot_dir)

    print(f"    -> Time-frequency analysis...")
    plot_time_frequency(epochs_target, epochs_nontarget, subj_id, plot_dir)

    print(f"    -> Channel P300 amplitude...")
    plot_channel_p300_amplitude(epochs_target, epochs_nontarget, subj_id, plot_dir)

    # Compute peak stats
    stats = compute_peak_stats(epochs_target, CH_NAMES_MNE)

    print(f"    -> Peak stats table...")
    plot_peak_stats_table(stats, subj_id, plot_dir)

    # Save configuration
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
        f.write(f"\nTotal plots generated: 9\n")
        f.write("  1. S{:02d}_butterfly_erp.png - Classic ERP overlay\n".format(subj_id))
        f.write("  2. S{:02d}_topo_maps.png - Topographic maps\n".format(subj_id))
        f.write("  3. S{:02d}_butterfly_overlay.png - All-channel butterfly\n".format(subj_id))
        f.write("  4. S{:02d}_difference_waves.png - Target-Nontarget differences\n".format(subj_id))
        f.write("  5. S{:02d}_signal_quality.png - SNR and quality metrics\n".format(subj_id))
        f.write("  6. S{:02d}_time_frequency.png - Spectrograms\n".format(subj_id))
        f.write("  7. S{:02d}_channel_p300_amplitude.png - P300 amplitude bars\n".format(subj_id))
        f.write("  8. S{:02d}_peak_stats_table.png - P300 peak statistics table\n".format(subj_id))
        f.write("  9. S{:02d}_config.txt - This file\n".format(subj_id))

    print(f"  Subject {subj_id} complete. Saved to {plot_dir}")
    return stats


def main():
    """Process all subjects with comprehensive visualizations."""
    print("=" * 70)
    print(" P300 EEG – Comprehensive Subject-Specific Preprocessing and EDA")
    print(" Includes 9 files per subject:")
    print("   1. Butterfly ERP (classic)")
    print("   2. Topographic maps")
    print("   3. All-channel butterfly overlay")
    print("   4. Difference waves")
    print("   5. Signal quality metrics")
    print("   6. Time-frequency analysis")
    print("   7. Channel P300 amplitude")
    print("   8. Peak stats table")
    print("   9. Configuration file")
    print("=" * 70)

    all_stats = {}
    for subj_id, config in SUBJECT_CONFIG.items():
        stats = process_subject(subj_id, config)
        all_stats[subj_id] = stats

    print("\n" + "=" * 70)
    print(" Processing Complete!")
    print("=" * 70)
    print(f"\nResults saved in subject-specific folders under:")
    print(f"  {BASE_PLOT_DIR}/")
    print("\nEach subject folder contains 9 files (8 plots + 1 config)")
    print("\nDone!")


if __name__ == "__main__":
    main()
