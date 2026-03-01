"""
P300 EEG Data - Complete Exploratory Data Analysis
====================================================
Generates EDA plots and summary reports per subject:
  - Filtering pipeline (notch 50 Hz + bandpass 1-30 Hz)
  - Baseline correction (pre-stimulus mean subtraction)
  - Filtered vs Unfiltered comparison
  - Butterworth filter frequency response
  - MNE-based topographic maps (standard_1020 montage)
  - Per-subject data summary text report

Requirements: pip install numpy scipy matplotlib mne
Usage:        python p300_eda.py
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
PLOT_DIR = "/Users/siznayak/Documents/others/MTech/EEG_Classification/analysis/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

FS = 250  # Sampling rate (Hz)
PRE_STIM_MS = 200   # Pre-stimulus window (ms)
POST_STIM_MS = 800  # Post-stimulus window (ms)
PRE_STIM_SAMPLES = int(PRE_STIM_MS * FS / 1000)       # 50
POST_STIM_SAMPLES = int(POST_STIM_MS * FS / 1000)     # 200
EPOCH_SAMPLES = PRE_STIM_SAMPLES + POST_STIM_SAMPLES  # 250
TIME_AXIS_MS = np.arange(EPOCH_SAMPLES) / FS * 1000 - PRE_STIM_MS  # true per-sample positions
TIME_AXIS_S = TIME_AXIS_MS / 1000.0

SUBJECTS = [f"P300S{i:02d}.mat" for i in range(1, 9)]

# MNE channel names (standard_1020 compatible)
CH_NAMES_MNE = ["Fz", "Cz", "P3", "Pz", "P4", "PO7", "PO8", "Oz"]

# Filtering parameters (matching your Kaggle notebook)
NOTCH_FREQ = 50     # Hz
BAND_LOW = 1        # Hz
BAND_HIGH = 30      # Hz

# Colors
COLOR_TARGET = "#E74C3C"      # red
COLOR_NONTARGET = "#3498DB"   # blue

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})


# ===========================================================================
# Data Loading & Filtering
# ===========================================================================

def load_subject(filepath):
    """Load a single MAT file and return structured dict."""
    mat = sio.loadmat(filepath, squeeze_me=True)
    data = mat["data"]
    return {
        "X": data["X"].item(),          # (n_samples, 8) continuous EEG
        "flash": data["flash"].item(),   # (n_flashes, 4) [onset, ?, char_id, target_flag]
        "Fs": int(mat["Fs"]),
        "ch_names": [str(c) for c in mat["channelNames"]],
        "subject_id": int(mat["subject"]),
        "word": str(mat["Word"]),
    }


def filter_eeg(X, fs):
    """
    Apply notch filter (50 Hz) + bandpass (1-30 Hz) using MNE.
    Matches your Kaggle notebook pipeline:
        raw.notch_filter(freqs=50)
        raw.filter(l_freq=1, h_freq=30)
    """
    info = mne.create_info(ch_names=len(X[0]), sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(X.T, info, verbose=False)
    raw.notch_filter(freqs=NOTCH_FREQ, verbose=False)
    raw.filter(l_freq=BAND_LOW, h_freq=BAND_HIGH, verbose=False)
    return raw.get_data().T  # back to (n_samples, n_ch)


def butter_bandpass(lowcut, highcut, fs, order=5):
    """Butterworth bandpass filter coefficients (for frequency response plot)."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


# ===========================================================================
# Epoch Extraction with Baseline Correction
# ===========================================================================

def extract_epochs(X, flash):
    """
    Extract epochs from continuous data using flash onsets.
    flash[:, 0] = onset sample index
    flash[:, 3] = 1 (target) or 2 (non-target)

    Applies baseline correction: subtract mean of pre-stimulus window.

    Returns:
        epochs_target:    (n_target, epoch_samples, n_ch)
        epochs_nontarget: (n_nontarget, epoch_samples, n_ch)
    """
    n_samples = X.shape[0]
    epochs_target = []
    epochs_nontarget = []

    for i in range(flash.shape[0]):
        onset = int(flash[i, 0])
        start = onset - PRE_STIM_SAMPLES
        end = onset + POST_STIM_SAMPLES

        if start < 0 or end > n_samples:
            continue

        epoch = X[start:end, :].copy()  # (250, 8)

        # Baseline correction: subtract mean of pre-stimulus window
        baseline = np.mean(epoch[:PRE_STIM_SAMPLES, :], axis=0)
        epoch = epoch - baseline

        is_target = (flash[i, 3] == 1)
        if is_target:
            epochs_target.append(epoch)
        else:
            epochs_nontarget.append(epoch)

    return np.array(epochs_target), np.array(epochs_nontarget)


def extract_epochs_no_baseline(X, flash):
    """
    Extract epochs WITHOUT baseline correction (reproduces previous pipeline).
    Used for side-by-side comparison with the current baseline-corrected version.
    """
    n_samples = X.shape[0]
    epochs_target = []
    epochs_nontarget = []

    for i in range(flash.shape[0]):
        onset = int(flash[i, 0])
        start = onset - PRE_STIM_SAMPLES
        end = onset + POST_STIM_SAMPLES

        if start < 0 or end > n_samples:
            continue

        epoch = X[start:end, :].copy()

        is_target = (flash[i, 3] == 1)
        if is_target:
            epochs_target.append(epoch)
        else:
            epochs_nontarget.append(epoch)

    return np.array(epochs_target), np.array(epochs_nontarget)


# ===========================================================================
# Plot 1: Butterfly ERP (Target vs Non-Target) with P300 zone
# ===========================================================================

def plot_butterfly_erp(epochs_target, epochs_nontarget, subj_id, ch_names, ax):
    """
    Butterfly plot: average ERP across all channels.
    - Individual channels shown as faint lines
    - Grand mean bolded
    - P300 zone (200-400 ms) shaded gray
    """
    erp_target = epochs_target.mean(axis=0)      # (250, 8)
    erp_nontarget = epochs_nontarget.mean(axis=0)

    # Individual channels (faint)
    for ch in range(len(ch_names)):
        ax.plot(TIME_AXIS_S, erp_target[:, ch], color=COLOR_TARGET,
                alpha=0.35, linewidth=0.7)
        ax.plot(TIME_AXIS_S, erp_nontarget[:, ch], color=COLOR_NONTARGET,
                alpha=0.25, linewidth=0.7, linestyle="--")

    # Grand mean (bold)
    ax.plot(TIME_AXIS_S, erp_target.mean(axis=1), color=COLOR_TARGET,
            linewidth=2.5, label=f"Target (avg, n={len(epochs_target)})")
    ax.plot(TIME_AXIS_S, erp_nontarget.mean(axis=1), color=COLOR_NONTARGET,
            linewidth=2.5, linestyle="--", label=f"Non-Target (avg, n={len(epochs_nontarget)})")

    # P300 zone shading (0.2 - 0.4 s)
    ax.axvspan(0.2, 0.4, color="grey", alpha=0.2, label="P300 zone (200-400 ms)")

    # Stimulus onset line
    ax.axvline(0, color="gray", linewidth=1.0, linestyle=":")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(f"S{subj_id:02d} – Butterfly ERP (Baseline Corrected)")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_xlim(-0.2, 0.8)


# ===========================================================================
# Plot 1b: Filtered vs Unfiltered Single-Trial (Pz channel)
# ===========================================================================

def plot_filtered_vs_unfiltered(X_raw, X_filtered, flash, subj_id, ax):
    """
    Show first trial's Pz signal: raw (gray) vs filtered (blue).
    Matches your Kaggle notebook's single-trial visualization.
    """
    # Use first flash onset as reference
    onset = int(flash[0, 0])
    start = onset - PRE_STIM_SAMPLES
    end = onset + POST_STIM_SAMPLES

    pz_idx = 3  # Pz channel
    time = np.arange(end - start) / FS

    ax.plot(time, X_raw[start:end, pz_idx], color="gray", linewidth=1.2,
            alpha=0.7, label="Raw (unfiltered)")
    ax.plot(time, X_filtered[start:end, pz_idx], color="steelblue", linewidth=1.8,
            label="Filtered (1-30 Hz + notch 50)")

    ax.axvspan(0.2, 0.4, color="grey", alpha=0.15, label="P300 zone")
    ax.axvline(PRE_STIM_SAMPLES / FS, color="black", linewidth=0.8, linestyle=":",
               label="Stimulus onset")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(f"S{subj_id:02d} – Raw vs Filtered (Pz, Trial 1)")
    ax.legend(fontsize=7, loc="upper right")


# ===========================================================================
# Plot 2: Topographic Maps using MNE standard_1020 montage
# ===========================================================================

def make_mne_info():
    """Create MNE Info with standard_1020 montage."""
    info = mne.create_info(ch_names=CH_NAMES_MNE, sfreq=FS, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage)
    return info


def plot_topo_maps_mne(epochs_target, epochs_nontarget, subj_id):
    """
    Topographic maps at 0 ms, 350 ms, 600 ms.
    Uses MNE's EvokedArray + plot_topomap with standard_1020 montage.
    Matches your Kaggle notebook approach.
    """
    info = make_mne_info()

    erp_target = epochs_target.mean(axis=0)      # (250, 8)
    erp_nontarget = epochs_nontarget.mean(axis=0)
    erp_diff = erp_target - erp_nontarget

    time_points_s = [0.0, 0.35, 0.6]
    time_labels = ["0 ms", "350 ms (P300)", "600 ms"]

    fig, axes = plt.subplots(3, 3, figsize=(12, 11))
    fig.suptitle(f"S{subj_id:02d} – Topographic Maps (Target / Non-Target / Difference)",
                 fontsize=12, fontweight="bold")

    # MNE expects data in µV (already in µV), shape (n_ch, n_times)
    # We create EvokedArray with tinfo starting at -0.2 s
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
    fig.savefig(os.path.join(PLOT_DIR, f"S{subj_id:02d}_02_topo_maps_mne.png"), dpi=150)
    plt.close(fig)


# ===========================================================================
# Plot 3: Target vs Non-Target Ratio Bar
# ===========================================================================

def plot_ratio_bar(epochs_target, epochs_nontarget, subj_id, ax):
    """Bar chart: target vs non-target epoch counts with percentages."""
    n_target = len(epochs_target)
    n_nontarget = len(epochs_nontarget)
    total = n_target + n_nontarget
    pct_target = n_target / total * 100

    ax.bar(["Target", "Non-Target"], [n_target, n_nontarget],
           color=[COLOR_TARGET, COLOR_NONTARGET], edgecolor="white", linewidth=1.5)

    ax.text(0, n_target + 15, f"{n_target}\n({pct_target:.1f}%)", ha="center", va="bottom",
            fontsize=8, fontweight="bold")
    ax.text(1, n_nontarget + 15, f"{n_nontarget}\n({100 - pct_target:.1f}%)", ha="center",
            va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylabel("Number of Epochs")
    ax.set_title(f"S{subj_id:02d} – Target Ratio")
    ax.set_ylim(0, max(n_target, n_nontarget) * 1.25)


# ===========================================================================
# Plot 4: Amplitude Distribution (Violin + Boxplot)
# ===========================================================================

def plot_amplitude_distribution(epochs_target, epochs_nontarget, subj_id, axes):
    """
    Left: Violin plot of mean amplitude per epoch (butterfly signal).
    Right: Per-channel boxplot side-by-side.
    """
    # Mean amplitude per epoch (single value per epoch)
    amp_target_epoch = epochs_target.mean(axis=(1, 2))
    amp_nontarget_epoch = epochs_nontarget.mean(axis=(1, 2))

    ax1 = axes[0]
    parts = ax1.violinplot(
        [amp_target_epoch, amp_nontarget_epoch],
        positions=[0.8, 2.2], widths=0.8,
        showmedians=True, showextrema=True
    )
    parts["bodies"][0].set_facecolor(COLOR_TARGET)
    parts["bodies"][0].set_alpha(0.5)
    parts["bodies"][1].set_facecolor(COLOR_NONTARGET)
    parts["bodies"][1].set_alpha(0.5)
    ax1.set_xticks([0.8, 2.2])
    ax1.set_xticklabels(["Target", "Non-Target"])
    ax1.set_ylabel("Mean Amplitude (µV)")
    ax1.set_title(f"S{subj_id:02d} – Epoch Amplitude Distribution")

    # Per-channel boxplot
    ax2 = axes[1]
    n_ch = epochs_target.shape[2]
    x_pos_target = np.arange(n_ch) - 0.2
    x_pos_nontarget = np.arange(n_ch) + 0.2
    width = 0.35

    bp1 = ax2.boxplot([epochs_target[:, :, ch].flatten() for ch in range(n_ch)],
                      positions=x_pos_target, widths=width, patch_artist=True,
                      showfliers=False)
    bp2 = ax2.boxplot([epochs_nontarget[:, :, ch].flatten() for ch in range(n_ch)],
                      positions=x_pos_nontarget, widths=width, patch_artist=True,
                      showfliers=False)
    for patch in bp1["boxes"]:
        patch.set_facecolor(COLOR_TARGET)
        patch.set_alpha(0.6)
    for patch in bp2["boxes"]:
        patch.set_facecolor(COLOR_NONTARGET)
        patch.set_alpha(0.6)

    ax2.set_xticks(np.arange(n_ch))
    ax2.set_xticklabels(CH_NAMES_MNE, rotation=45)
    ax2.set_ylabel("Amplitude (µV)")
    ax2.set_title(f"S{subj_id:02d} – Per-Channel Amplitude")
    ax2.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Target", "Non-Target"],
               loc="upper right", fontsize=7)


# ===========================================================================
# Plot 5: Epoch Quality (Artifact Check)
# ===========================================================================

def plot_epoch_quality(epochs_target, epochs_nontarget, subj_id, ax):
    """Histogram of max absolute amplitude per epoch with threshold lines."""
    all_epochs = np.concatenate([epochs_target, epochs_nontarget], axis=0)
    max_amp = np.abs(all_epochs).max(axis=(1, 2))

    ax.hist(max_amp, bins=50, color="steelblue", edgecolor="white", alpha=0.8)

    # Draw threshold lines
    for thresh, label, color in [(50, "+-50 uV", "orange"),
                                  (100, "+-100 uV", "red"),
                                  (150, "+-150 uV", "darkred")]:
        ax.axvline(thresh, color=color, linewidth=1.5, linestyle="--", label=label)

    ax.set_xlabel("Max |Amplitude| per Epoch (µV)")
    ax.set_ylabel("Count")
    ax.set_title(f"S{subj_id:02d} – Epoch Quality (Artifact Check)")
    ax.legend(fontsize=7, loc="upper right")

    # Annotate % exceeding each threshold (after axes limits are set)
    ylim = ax.get_ylim()
    for thresh, color, yf in [(50, "orange", 0.88), (100, "red", 0.70), (150, "darkred", 0.52)]:
        n_exceed = np.sum(max_amp > thresh)
        pct = n_exceed / len(max_amp) * 100
        ax.text(thresh + 1, ylim[1] * yf, f"{pct:.1f}%", fontsize=7, color=color, fontweight="bold")


# ===========================================================================
# Plot 6: Channel Variance Heatmap
# ===========================================================================

def plot_channel_variance_heatmap(epochs_target, epochs_nontarget, subj_id, ax):
    """8x8 covariance heatmap with annotated values."""
    all_epochs = np.concatenate([epochs_target, epochs_nontarget], axis=0)
    reshaped = all_epochs.reshape(-1, all_epochs.shape[2])
    cov_matrix = np.cov(reshaped, rowvar=False)

    im = ax.imshow(cov_matrix, cmap="YlOrRd", aspect="equal")
    ax.set_xticks(range(len(CH_NAMES_MNE)))
    ax.set_yticks(range(len(CH_NAMES_MNE)))
    ax.set_xticklabels(CH_NAMES_MNE, rotation=45)
    ax.set_yticklabels(CH_NAMES_MNE)
    ax.set_title(f"S{subj_id:02d} – Channel Covariance Heatmap")

    for i in range(len(CH_NAMES_MNE)):
        for j in range(len(CH_NAMES_MNE)):
            color = "white" if cov_matrix[i, j] > cov_matrix.max() * 0.6 else "black"
            ax.text(j, i, f"{cov_matrix[i, j]:.1f}", ha="center", va="center",
                    fontsize=7, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label="Covariance (µV²)")


# ===========================================================================
# Plot 7: P300 Peak Latency / Amplitude / SNR Table
# ===========================================================================

def compute_peak_stats(epochs_target, ch_names):
    """
    Per channel: Peak latency, Peak amplitude, SNR
    within the P300 search window (200-600 ms post-stimulus).
    SNR = peak amplitude / std of baseline (-200 to 0 ms).
    """
    erp_target = epochs_target.mean(axis=0)  # (250, 8)

    results = []
    for ch_idx, ch_name in enumerate(ch_names):
        erp_ch = erp_target[:, ch_idx]

        # Post-stim ERP (samples 50 to 250)
        post_stim_erp = erp_ch[PRE_STIM_SAMPLES:]

        # P300 search window: 200-600 ms into post-stim
        p300_start = int(200 * FS / 1000)  # 50 samples
        p300_end = int(600 * FS / 1000)    # 150 samples
        p300_window = post_stim_erp[p300_start:p300_end]

        peak_idx = np.argmax(p300_window)
        peak_latency_ms = 200 + peak_idx * 1000 / FS
        peak_amplitude = p300_window[peak_idx]

        # Baseline std
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


def plot_peak_stats_table(stats, subj_id, ax):
    """Render P300 peak stats as a styled table (top 5 channels by SNR)."""
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


# ===========================================================================
# Data Summary: Per-Subject Text Report
# ===========================================================================

def generate_subject_summary(subj_data, X_filtered, epochs_target, epochs_nontarget, stats, filepath):
    """
    Generate a detailed text summary of the subject's data and save as .txt.
    Covers: raw shape, channel info, flash structure, epoch counts,
    target ratio, amplitude ranges, artifact stats, and peak stats.
    """
    subj_id = subj_data["subject_id"]
    X_raw = subj_data["X"]
    flash = subj_data["flash"]
    ch_names_mat = subj_data["ch_names"]

    n_target = len(epochs_target)
    n_nontarget = len(epochs_nontarget)
    total_epochs = n_target + n_nontarget
    pct_target = n_target / total_epochs * 100

    # Amplitude ranges (filtered epochs)
    all_epochs = np.concatenate([epochs_target, epochs_nontarget], axis=0)
    max_amp_per_epoch = np.abs(all_epochs).max(axis=(1, 2))
    n_exceed_50 = int(np.sum(max_amp_per_epoch > 50))
    n_exceed_100 = int(np.sum(max_amp_per_epoch > 100))
    n_exceed_150 = int(np.sum(max_amp_per_epoch > 150))

    # Per-channel variance
    reshaped = all_epochs.reshape(-1, all_epochs.shape[2])
    cov_matrix = np.cov(reshaped, rowvar=False)

    # Build report
    lines = []
    lines.append("=" * 70)
    lines.append(f"  P300 EDA – Subject S{subj_id:02d} Data Summary")
    lines.append("=" * 70)

    lines.append("")
    lines.append("--- Raw Data ---")
    lines.append(f"  File:                {os.path.basename(filepath)}")
    lines.append(f"  Subject ID:          {subj_id}")
    lines.append(f"  Word (target):       {subj_data['word']}")
    lines.append(f"  Sampling rate (Fs):  {FS} Hz")
    lines.append(f"  Continuous EEG shape:{X_raw.shape[0]} samples x {X_raw.shape[1]} channels")
    lines.append(f"  Duration:            {X_raw.shape[0] / FS:.1f} seconds")
    lines.append(f"  Raw amplitude range: [{X_raw.min():.2f}, {X_raw.max():.2f}] µV")

    lines.append("")
    lines.append("--- Channels ---")
    lines.append(f"  MAT file names:      {ch_names_mat}")
    lines.append(f"  MNE montage names:   {CH_NAMES_MNE}")
    lines.append(f"  Number of channels:  {X_raw.shape[1]}")

    lines.append("")
    lines.append("--- Flash Events ---")
    lines.append(f"  Total flashes:       {flash.shape[0]}")
    lines.append(f"  Flash columns:       [onset_sample, row_or_col, char_id, target_flag]")
    lines.append(f"  Target flag values:  1 = target, 2 = non-target")
    lines.append(f"  Target flashes:      {int(np.sum(flash[:, 3] == 1))}")
    lines.append(f"  Non-target flashes:  {int(np.sum(flash[:, 3] == 2))}")
    lines.append(f"  First flash onset:   sample {int(flash[0, 0])}")
    lines.append(f"  Last flash onset:    sample {int(flash[-1, 0])}")

    lines.append("")
    lines.append("--- Filtering Pipeline ---")
    lines.append(f"  Notch filter:        {NOTCH_FREQ} Hz")
    lines.append(f"  Bandpass:            {BAND_LOW}–{BAND_HIGH} Hz")
    lines.append(f"  Filtered amp range:  [{X_filtered.min():.2f}, {X_filtered.max():.2f}] µV")

    lines.append("")
    lines.append("--- Epoch Extraction ---")
    lines.append(f"  Window:              {-PRE_STIM_MS} to +{POST_STIM_MS} ms")
    lines.append(f"  Samples per epoch:   {EPOCH_SAMPLES}")
    lines.append(f"  Baseline correction: mean of pre-stimulus ({-PRE_STIM_MS} to 0 ms) subtracted")
    lines.append(f"  Target epochs:       {n_target}")
    lines.append(f"  Non-target epochs:   {n_nontarget}")
    lines.append(f"  Total epochs:        {total_epochs}")
    lines.append(f"  Target ratio:        {pct_target:.1f}%")
    lines.append(f"  Non-target ratio:   {100 - pct_target:.1f}%")

    lines.append("")
    lines.append("--- Epoch Quality (Artifact Check) ---")
    lines.append(f"  Epochs > ±50 µV:     {n_exceed_50} ({n_exceed_50/total_epochs*100:.1f}%)")
    lines.append(f"  Epochs > ±100 µV:    {n_exceed_100} ({n_exceed_100/total_epochs*100:.1f}%)")
    lines.append(f"  Epochs > ±150 µV:    {n_exceed_150} ({n_exceed_150/total_epochs*100:.1f}%)")
    lines.append(f"  Max |amplitude|:     {max_amp_per_epoch.max():.2f} µV")
    lines.append(f"  Median max |amp|:    {np.median(max_amp_per_epoch):.2f} µV")

    lines.append("")
    lines.append("--- Channel Variance (diagonal of covariance matrix) ---")
    for idx, ch in enumerate(CH_NAMES_MNE):
        lines.append(f"  {ch:>5}:  {cov_matrix[idx, idx]:>10.2f} µV²")

    lines.append("")
    lines.append("--- P300 Peak Stats (all channels, sorted by SNR) ---")
    lines.append(f"  {'Channel':<8} {'Peak Latency (ms)':<20} {'Peak Amp (µV)':<16} {'SNR':<8}")
    lines.append(f"  {'-'*8} {'-'*20} {'-'*16} {'-'*8}")
    for row in stats:
        lines.append(f"  {row['Channel']:<8} {row['Peak_Latency_ms']:<20.1f} {row['Peak_Amp_uV']:<16.2f} {row['SNR']:<8.2f}")

    lines.append("")
    lines.append("--- Best P300 Summary ---")
    lines.append(f"  Best P300 = {stats[0]['Channel']} | "
                 f"{stats[0]['Peak_Latency_ms']}ms | "
                 f"{stats[0]['Peak_Amp_uV']:.1f} µV | "
                 f"SNR {stats[0]['SNR']:.2f}")

    lines.append("")
    lines.append("=" * 70)

    report = "\n".join(lines)

    # Print to console
    print(report)

    # Save to file
    out_path = os.path.join(PLOT_DIR, f"S{subj_id:02d}_data_summary.txt")
    with open(out_path, "w") as f:
        f.write(report)

    return report


# ===========================================================================
# Bonus: Butterworth Filter Frequency Response (orders 1-6)
# ===========================================================================

def plot_filter_frequency_response():
    """
    Butterworth bandpass filter frequency response for orders 1-6.
    Matches your Kaggle notebook's filter order exploration.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    for order in [1, 2, 3, 4, 5, 6]:
        b, a = butter_bandpass(BAND_LOW, BAND_HIGH, FS, order=order)
        w, h = freqz(b, a, worN=2000)
        ax.plot((FS * 0.5 / np.pi) * w, abs(h), label=f"Order {order}")

    ax.plot([0, 0.5 * FS], [np.sqrt(0.5), np.sqrt(0.5)],
            "k--", linewidth=1, label="-3 dB")

    ax.set_title("Butterworth Bandpass Filter (1-30 Hz) – Frequency Response")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Gain")
    ax.set_xlim([0, 50])
    ax.legend(fontsize=8)
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "00_filter_frequency_response.png"), dpi=150)
    plt.close(fig)


# ===========================================================================
# Main: Generate All Plots Per Subject
# ===========================================================================

def process_subject(filepath):
    """Process one subject: filter, extract epochs, generate all plots."""
    print(f"\n{'=' * 60}")
    subj_data = load_subject(filepath)
    subj_id = subj_data["subject_id"]
    X_raw = subj_data["X"]
    flash = subj_data["flash"]
    print(f"  Processing Subject {subj_id} ({os.path.basename(filepath)})")

    # --- Filtering ---
    print(f"  Filtering (notch 50 Hz + bandpass 1-30 Hz)...")
    X_filtered = filter_eeg(X_raw, FS)

    # --- Epoch Extraction (on filtered data, with baseline correction) ---
    print(f"  Extracting epochs (baseline corrected)...")
    epochs_target, epochs_nontarget = extract_epochs(X_filtered, flash)
    print(f"  Epochs: {len(epochs_target)} target, {len(epochs_nontarget)} non-target")

    # ---------------------------------------------------------------
    # Figure 1: Combined Butterfly ERP comparison (2x2 layout)
    #   Top-left:  Previous butterfly (no filter, no baseline)
    #   Top-right: Previous butterfly zoomed into P300 zone
    #   Bot-left:  Current butterfly (filtered + baseline corrected)
    #   Bot-right: Single-trial Pz raw vs filtered
    # ---------------------------------------------------------------
    print(f"  -> Plot 1: Combined Butterfly ERP (previous vs current)...")

    # Previous epochs: no filtering, no baseline correction
    epochs_target_prev, epochs_nontarget_prev = extract_epochs_no_baseline(X_raw, flash)

    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 9))
    fig1.suptitle(f"S{subj_id:02d} – Butterfly ERP: Previous (no filter) vs Current (filtered + baseline)",
                     fontsize=12, fontweight="bold")

    # Top-left: Previous butterfly (raw, no baseline)
    plot_butterfly_erp(epochs_target_prev, epochs_nontarget_prev, subj_id, CH_NAMES_MNE, axes1[0, 0])
    axes1[0, 0].set_title(f"S{subj_id:02d} – Previous (No Filter, No Baseline)")

    # Top-right: Previous butterfly zoomed into P300 zone (0.1 – 0.5 s)
    plot_butterfly_erp(epochs_target_prev, epochs_nontarget_prev, subj_id, CH_NAMES_MNE, axes1[0, 1])
    axes1[0, 1].set_xlim(0.1, 0.5)
    axes1[0, 1].set_title(f"S{subj_id:02d} – Previous (Zoomed P300 Zone)")

    # Bottom-left: Current butterfly (filtered + baseline corrected)
    plot_butterfly_erp(epochs_target, epochs_nontarget, subj_id, CH_NAMES_MNE, axes1[1, 0])
    axes1[1, 0].set_title(f"S{subj_id:02d} – Current (Filtered + Baseline Corrected)")

    # Bottom-right: Single-trial Pz raw vs filtered
    plot_filtered_vs_unfiltered(X_raw, X_filtered, flash, subj_id, axes1[1, 1])

    fig1.tight_layout(rect=[0, 0, 1, 0.94])
    fig1.savefig(os.path.join(PLOT_DIR, f"S{subj_id:02d}_01_butterfly_combined.png"), dpi=150)
    plt.close(fig1)

    # ---------------------------------------------------------------
    # Figure 1 standalone: Butterfly ERP (filtered + baseline) — the original 7 essential plots
    # ---------------------------------------------------------------
    print(f"  -> Plot 1 standalone: Butterfly ERP...")
    fig1s, ax1s = plt.subplots(1, 1, figsize=(10, 5))
    plot_butterfly_erp(epochs_target, epochs_nontarget, subj_id, CH_NAMES_MNE, ax1s)
    fig1s.tight_layout()
    fig1s.savefig(os.path.join(PLOT_DIR, f"S{subj_id:02d}_01_butterfly_erp.png"), dpi=150)
    plt.close(fig1s)

    # ---------------------------------------------------------------
    # Figure 2: Topographic Maps (MNE standard_1020)
    # ---------------------------------------------------------------
    print(f"  -> Plot 2: Topographic Maps (MNE)...")
    plot_topo_maps_mne(epochs_target, epochs_nontarget, subj_id)

    # ---------------------------------------------------------------
    # Figure 3: Target Ratio Bar
    # ---------------------------------------------------------------
    print(f"  -> Plot 3: Target Ratio Bar...")
    fig3, ax3 = plt.subplots(1, 1, figsize=(6, 5))
    plot_ratio_bar(epochs_target, epochs_nontarget, subj_id, ax3)
    fig3.tight_layout()
    fig3.savefig(os.path.join(PLOT_DIR, f"S{subj_id:02d}_03_target_ratio.png"), dpi=150)
    plt.close(fig3)

    # ---------------------------------------------------------------
    # Figure 4: Amplitude Distribution
    # ---------------------------------------------------------------
    print(f"  -> Plot 4: Amplitude Distribution...")
    fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5))
    plot_amplitude_distribution(epochs_target, epochs_nontarget, subj_id, axes4)
    fig4.tight_layout()
    fig4.savefig(os.path.join(PLOT_DIR, f"S{subj_id:02d}_04_amplitude_dist.png"), dpi=150)
    plt.close(fig4)

    # ---------------------------------------------------------------
    # Figure 5: Epoch Quality
    # ---------------------------------------------------------------
    print(f"  -> Plot 5: Epoch Quality...")
    fig5, ax5 = plt.subplots(1, 1, figsize=(8, 5))
    plot_epoch_quality(epochs_target, epochs_nontarget, subj_id, ax5)
    fig5.tight_layout()
    fig5.savefig(os.path.join(PLOT_DIR, f"S{subj_id:02d}_05_epoch_quality.png"), dpi=150)
    plt.close(fig5)

    # ---------------------------------------------------------------
    # Figure 6: Channel Variance Heatmap
    # ---------------------------------------------------------------
    print(f"  -> Plot 6: Channel Variance Heatmap...")
    fig6, ax6 = plt.subplots(1, 1, figsize=(7, 6))
    plot_channel_variance_heatmap(epochs_target, epochs_nontarget, subj_id, ax6)
    fig6.tight_layout()
    fig6.savefig(os.path.join(PLOT_DIR, f"S{subj_id:02d}_06_channel_variance.png"), dpi=150)
    plt.close(fig6)

    # ---------------------------------------------------------------
    # Figure 7: P300 Peak Stats Table
    # ---------------------------------------------------------------
    print(f"  -> Plot 7: P300 Peak Stats Table...")
    stats = compute_peak_stats(epochs_target, CH_NAMES_MNE)
    fig7, ax7 = plt.subplots(1, 1, figsize=(8, 4))
    plot_peak_stats_table(stats, subj_id, ax7)
    fig7.tight_layout()
    fig7.savefig(os.path.join(PLOT_DIR, f"S{subj_id:02d}_07_peak_stats_table.png"), dpi=150)
    plt.close(fig7)

    # ---------------------------------------------------------------
    # Data Summary: save per-subject text report
    # ---------------------------------------------------------------
    print(f"  -> Generating data summary...")
    generate_subject_summary(subj_data, X_filtered, epochs_target, epochs_nontarget, stats, filepath)

    print(f"  Subject {subj_id} complete.")
    return stats


def main():
    print("=" * 60)
    print(" P300 EEG – Complete EDA Analysis (8 Subjects)")
    print(" Pipeline: Notch 50 Hz -> Bandpass 1-30 Hz -> Epochs -> Baseline")
    print("=" * 60)

    # Bonus: Filter frequency response (once, global)
    print("\n  -> Generating filter frequency response...")
    plot_filter_frequency_response()

    all_stats = {}
    for subj_file in SUBJECTS:
        filepath = os.path.join(DATA_DIR, subj_file)
        if not os.path.exists(filepath):
            print(f"  [WARN] File not found: {filepath}")
            continue
        stats = process_subject(filepath)
        # Extract subject ID from filename
        subj_id = int(subj_file.replace("P300S", "").replace(".mat", ""))
        all_stats[subj_id] = stats

    # ---------------------------------------------------------------
    # Cross-Subject Summary Table
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" Cross-Subject Summary: P300 Peak Stats")
    print("=" * 60)

    fig_summary, ax_summary = plt.subplots(1, 1, figsize=(10, 6))
    ax_summary.axis("off")

    columns = ["Subject", "Channel", "Peak Latency (ms)", "Peak Amp (µV)", "SNR"]
    cell_text = []
    for subj_id in sorted(all_stats.keys()):
        top = all_stats[subj_id][0]
        cell_text.append([f"S{subj_id:02d}", top["Channel"],
                          f"{top['Peak_Latency_ms']:.1f}",
                          f"{top['Peak_Amp_uV']:.2f}",
                          f"{top['SNR']:.2f}"])

    table = ax_summary.table(cellText=cell_text, colLabels=columns, loc="center",
                             cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.3, 1.8)

    for j in range(len(columns)):
        table[(0, j)].set_facecolor("#2C3E50")
        table[(0, j)].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(cell_text) + 1):
        for j in range(len(columns)):
            table[(i, j)].set_facecolor("#D5F5E3" if i % 2 == 0 else "white")

    ax_summary.set_title("Cross-Subject: Best P300 Channel per Subject (Filtered + Baseline)",
                         fontsize=12, fontweight="bold", pad=20)
    fig_summary.tight_layout()
    fig_summary.savefig(os.path.join(PLOT_DIR, "00_cross_subject_summary.png"), dpi=150)
    plt.close(fig_summary)

    print(f"\nAll plots saved to: {PLOT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
