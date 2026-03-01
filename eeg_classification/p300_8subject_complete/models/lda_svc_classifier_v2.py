"""
P300 Classification v2: xDAWN + rLDA / SVC
============================================
What changed from v1:
  - xDAWN spatial filter (n_components=2) fitted INSIDE each CV fold
    on training epochs, applied to train + test epochs
  - Feature pipeline: xDAWN-projected epochs (n, 2, 250)
    → crop 200-500ms → bin 15 → flatten → 30-dim
    (2 xDAWN components × 15 bins = 30-dim)
  - Everything else identical: rLDA, SVC, WS-CV, LOSO, StandardScaler

Why xDAWN inside CV?
  xDAWN uses labels (y_train) to learn the spatial filter.
  If fitted on all data before splitting, it leaks target information
  from test epochs into the filter → inflated results.
  Fitting only on train data prevents this leakage.

xDAWN recap:
  - Learns n_components spatial filters that maximise SNR of
    the averaged target ERP relative to the background EEG.
  - Each component = linear combination of 8 electrode signals
    that best separates target from non-target.
  - After xDAWN: epoch (8, 250) → (2, 250)
  - This makes the feature representation subject-general because
    the spatial filter adapts to the training data's P300 geometry.

Input : features/p300_features_v2.npz
        (contains both X_epochs and X_features)

Usage:
    python models/lda_svc_classifier_v2.py
"""

import os
import json
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.linalg import eigh

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FEATURES_PATH = "/Users/siznayak/Documents/others/MTech/EEG_Classification/features/p300_features_v2.npz"
RESULTS_DIR   = "/Users/siznayak/Documents/others/MTech/EEG_Classification/results"
PLOTS_DIR     = os.path.join(RESULTS_DIR, "plots_v2")

CH_NAMES = ["Fz", "Cz", "P3", "Pz", "P4", "PO7", "PO8", "Oz"]
N_XDAWN_COMPONENTS = 2    # standard for P300: 2 signal + 2 noise components

# Feature constants (same 200-500ms window, now on 2 xDAWN components)
FS             = 250
PRE_MS         = 200
P300_START_IDX = int((200 + PRE_MS) * FS / 1000)   # 100
P300_END_IDX   = int((500 + PRE_MS) * FS / 1000)   # 175
N_BINS         = 15
BIN_SIZE       = (P300_END_IDX - P300_START_IDX) // N_BINS   # 5

# After xDAWN: 2 components × 15 bins = 30-dim
N_COMPONENTS_FEAT = N_XDAWN_COMPONENTS * N_BINS   # 30

# ---------------------------------------------------------------------------
# xDAWN (numpy implementation) + feature extraction
# ---------------------------------------------------------------------------

def fit_xdawn(X_train, y_train, n_components=2):
    """
    Fit xDAWN spatial filter on training epochs.

    xDAWN finds spatial filters W that maximise SNR of the target ERP:
        W = argmax  W^T * C_signal * W
                    W^T * C_noise  * W
    where:
        C_signal = covariance of the class-averaged target ERP
        C_noise  = total data covariance (signal + noise)

    This is solved as a generalised eigenvalue problem:
        C_signal * w = lambda * C_noise * w

    Args:
        X_train     : (n, 8, 250)
        y_train     : (n,)  0/1 labels
        n_components: number of top spatial filters to keep

    Returns:
        W : (8, n_components)  spatial filter matrix
            Apply as:  X_proj = W.T @ X_epoch   → (n_components, 250)
    """
    n, n_ch, n_t = X_train.shape

    # Averaged target ERP (signal template)  → (n_ch, n_t)
    target_mask = y_train == 1
    erp_target  = X_train[target_mask].mean(axis=0)  # (n_ch, n_t)

    # C_signal: covariance of the signal template
    # Treat each time point as an observation: (n_t, n_ch)
    A = erp_target.T  # (n_t, n_ch)
    C_signal = A.T @ A  # (n_ch, n_ch)

    # C_noise: total data covariance across all epochs
    # Reshape all epochs to (n*n_t, n_ch)
    X_flat   = X_train.transpose(0, 2, 1).reshape(-1, n_ch)  # (n*n_t, n_ch)
    C_noise  = X_flat.T @ X_flat  # (n_ch, n_ch)

    # Regularise C_noise for numerical stability (Tikhonov regularisation)
    # Use 1% of mean diagonal — enough to ensure positive definiteness
    reg = 0.01 * np.trace(C_noise) / n_ch
    C_noise += np.eye(n_ch) * reg

    # Generalised eigenvalue problem: C_signal w = lambda C_noise w
    # eigh returns eigenvalues in ascending order → take last n_components
    eigenvalues, eigenvectors = eigh(C_signal, C_noise)
    W = eigenvectors[:, -n_components:]  # (n_ch, n_components)  top filters
    return W


def apply_xdawn_and_extract(X_train, y_train, X_test):
    """
    Fit xDAWN on training epochs, project train + test, then extract features.

    Args:
        X_train : (n_train, 8, 250)
        y_train : (n_train,)   0/1 labels
        X_test  : (n_test,  8, 250)

    Returns:
        feat_train : (n_train, N_COMPONENTS_FEAT)   e.g. (n_train, 30)
        feat_test  : (n_test,  N_COMPONENTS_FEAT)
    """
    W = fit_xdawn(X_train, y_train, n_components=N_XDAWN_COMPONENTS)
    # W: (8, n_comp).  For each epoch e (8,250): proj = W.T @ e → (n_comp,250)
    # einsum "nct,ck->nkt" : n=epochs, c=channels, t=time, k=components
    X_tr_proj = np.einsum("nct,ck->nkt", X_train, W)   # (n_train, 2, 250)
    X_te_proj = np.einsum("nct,ck->nkt", X_test,  W)   # (n_test,  2, 250)

    feat_train = _bin_features(X_tr_proj)
    feat_test  = _bin_features(X_te_proj)
    return feat_train, feat_test


def _bin_features(X_proj):
    """
    Crop 200-500ms → 15 bins → flatten.

    Args:
        X_proj : (n, n_comp, 250)
    Returns:
        (n, n_comp * 15)
    """
    n, n_comp, _ = X_proj.shape
    X_p300  = X_proj[:, :, P300_START_IDX:P300_END_IDX]              # (n, n_comp, 75)
    X_binned = X_p300.reshape(n, n_comp, N_BINS, BIN_SIZE).mean(axis=3)  # (n, n_comp, 15)
    return X_binned.reshape(n, -1).astype(np.float32)                 # (n, 30)


# ---------------------------------------------------------------------------
# Helpers (same as v1)
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_score=None):
    return {
        "accuracy"         : float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision"        : float(precision_score(y_true, y_pred, zero_division=0)),
        "recall"           : float(recall_score(y_true, y_pred, zero_division=0)),
        "f1"               : float(f1_score(y_true, y_pred, zero_division=0)),
        "auc"              : float(roc_auc_score(y_true, y_score)) if y_score is not None else None,
        "confusion_matrix" : confusion_matrix(y_true, y_pred).tolist(),
    }


def get_score(model, X_test):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    return model.decision_function(X_test)


def build_model(name):
    if name == "rLDA":
        return LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
    elif name == "SVC":
        return SVC(kernel="linear", C=0.1, class_weight="balanced",
                   probability=True, random_state=42, max_iter=5000)
    raise ValueError(f"Unknown model: {name}")


def summarise(fold_metrics_list):
    keys = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "auc"]
    summary = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics_list if m[k] is not None]
        summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return summary


def print_summary(model_name, cv_name, summary):
    print(f"\n  {'='*60}")
    print(f"  SUMMARY  {model_name}  |  {cv_name}")
    print(f"  {'='*60}")
    for k, v in summary.items():
        print(f"  {k:<20}: {v['mean']:.3f} ± {v['std']:.3f}")
    print(f"  {'='*60}\n")


# ---------------------------------------------------------------------------
# Strategy A: Within-Subject CV
# ---------------------------------------------------------------------------

def within_subject_cv(X_epochs, y, subject_id, model_name, n_splits=5):
    print(f"\n{'='*70}")
    print(f"  Within-Subject CV  |  {model_name}  |  StratifiedKFold(k={n_splits})")
    print(f"  xDAWN({N_XDAWN_COMPONENTS} components) fitted per fold on training epochs")
    print(f"{'='*70}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_fold_metrics = []

    for sid in np.unique(subject_id):
        mask     = subject_id == sid
        X_subj   = X_epochs[mask]   # (n_subj, 8, 250)
        y_subj   = y[mask]

        subj_fold_metrics = []
        for fold, (tr, te) in enumerate(skf.split(X_subj, y_subj)):
            X_tr_ep, X_te_ep = X_subj[tr], X_subj[te]
            y_tr, y_te       = y_subj[tr], y_subj[te]

            # xDAWN + feature extraction (leakage-free)
            X_tr_feat, X_te_feat = apply_xdawn_and_extract(X_tr_ep, y_tr, X_te_ep)

            # StandardScaler on features
            scaler  = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr_feat)
            X_te_sc = scaler.transform(X_te_feat)

            model   = build_model(model_name)
            model.fit(X_tr_sc, y_tr)

            y_pred  = model.predict(X_te_sc)
            y_score = get_score(model, X_te_sc)
            m       = compute_metrics(y_te, y_pred, y_score)
            subj_fold_metrics.append(m)

        subj_summary = summarise(subj_fold_metrics)
        print(f"  S{sid:02d}  Acc={subj_summary['accuracy']['mean']:.3f}±{subj_summary['accuracy']['std']:.3f}  "
              f"BAcc={subj_summary['balanced_accuracy']['mean']:.3f}±{subj_summary['balanced_accuracy']['std']:.3f}  "
              f"F1={subj_summary['f1']['mean']:.3f}±{subj_summary['f1']['std']:.3f}  "
              f"AUC={subj_summary['auc']['mean']:.3f}±{subj_summary['auc']['std']:.3f}")
        all_fold_metrics.extend(subj_fold_metrics)

    overall = summarise(all_fold_metrics)
    print_summary(model_name, f"Within-Subject k={n_splits}", overall)
    return overall, all_fold_metrics


# ---------------------------------------------------------------------------
# Strategy B: LOSO
# ---------------------------------------------------------------------------

def loso_cv(X_epochs, y, subject_id, model_name):
    print(f"\n{'='*70}")
    print(f"  Cross-Subject LOSO  |  {model_name}  |  Leave-One-Subject-Out")
    print(f"  xDAWN({N_XDAWN_COMPONENTS} components) fitted per fold on training subjects")
    print(f"{'='*70}")

    logo            = LeaveOneGroupOut()
    fold_metrics    = []
    per_subject_res = {}

    for train_idx, test_idx in logo.split(X_epochs, y, groups=subject_id):
        test_sid = subject_id[test_idx[0]]
        X_tr_ep  = X_epochs[train_idx]   # (n_train, 8, 250)
        X_te_ep  = X_epochs[test_idx]    # (n_test,  8, 250)
        y_tr     = y[train_idx]
        y_te     = y[test_idx]

        # xDAWN fitted on all 7 training subjects, applied to test subject
        X_tr_feat, X_te_feat = apply_xdawn_and_extract(X_tr_ep, y_tr, X_te_ep)

        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_feat)
        X_te_sc = scaler.transform(X_te_feat)

        model   = build_model(model_name)
        model.fit(X_tr_sc, y_tr)

        y_pred  = model.predict(X_te_sc)
        y_score = get_score(model, X_te_sc)
        m       = compute_metrics(y_te, y_pred, y_score)

        print(f"    Test S{test_sid:02d}  "
              f"Acc={m['accuracy']:.3f}  BAcc={m['balanced_accuracy']:.3f}  "
              f"F1={m['f1']:.3f}  AUC={m['auc']:.3f}")
        fold_metrics.append(m)
        per_subject_res[int(test_sid)] = m

    overall = summarise(fold_metrics)
    print_summary(model_name, "LOSO", overall)
    return overall, fold_metrics, per_subject_res


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison_bar(lda_summary, svc_summary, cv_name, save_dir, suffix="v2"):
    metrics  = ["accuracy", "balanced_accuracy", "f1", "auc"]
    labels   = ["Accuracy", "Bal. Accuracy", "F1-Score", "AUC-ROC"]
    lda_vals = [lda_summary[m]["mean"] for m in metrics]
    lda_err  = [lda_summary[m]["std"]  for m in metrics]
    svc_vals = [svc_summary[m]["mean"] for m in metrics]
    svc_err  = [svc_summary[m]["std"]  for m in metrics]

    x     = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width/2, lda_vals, width, yerr=lda_err, label="rLDA",
                color="#2196F3", alpha=0.85, capsize=5)
    b2 = ax.bar(x + width/2, svc_vals, width, yerr=svc_err, label="SVC (linear)",
                color="#FF5722", alpha=0.85, capsize=5)

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Score")
    ax.set_title(f"v2 (xDAWN + CAR)  —  rLDA vs SVC  |  {cv_name}",
                 fontsize=12, fontweight="bold")
    ax.legend()
    ax.yaxis.grid(True, alpha=0.3)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(save_dir,
                        f"comparison_{cv_name.lower().replace(' ','_')}_{suffix}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_per_subject_bacc(lda_subj, svc_subj, cv_name, save_dir):
    """Per-subject balanced accuracy — the key metric for LOSO."""
    subjects = sorted(lda_subj.keys())
    lda_vals = [lda_subj[s]["balanced_accuracy"] for s in subjects]
    svc_vals = [svc_subj[s]["balanced_accuracy"] for s in subjects]
    labels   = [f"S{s:02d}" for s in subjects]

    x     = np.arange(len(subjects))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, lda_vals, width, label="rLDA",         color="#2196F3", alpha=0.85)
    ax.bar(x + width/2, svc_vals, width, label="SVC (linear)", color="#FF5722", alpha=0.85)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance (0.50)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title(f"v2 Per-Subject Balanced Accuracy  —  {cv_name}",
                 fontsize=12, fontweight="bold")
    ax.legend()
    ax.yaxis.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir,
                        f"per_subject_bacc_{cv_name.lower().replace(' ','_')}_v2.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_confusion_matrices(lda_res, svc_res, subject_ids, cv_name, save_dir):
    n = len(subject_ids)
    fig, axes = plt.subplots(n, 2, figsize=(8, n * 3))
    fig.suptitle(f"v2 Confusion Matrices – {cv_name}", fontsize=13, fontweight="bold")

    for i, sid in enumerate(subject_ids):
        for j, (res_dict, mname) in enumerate([(lda_res, "rLDA"), (svc_res, "SVC")]):
            ax  = axes[i, j]
            cm  = np.array(res_dict[sid]["confusion_matrix"])
            im  = ax.imshow(cm, interpolation="nearest", cmap="Blues")
            ax.set_title(f"S{sid:02d} – {mname}", fontsize=9)
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["Non-tgt", "Target"])
            ax.set_yticklabels(["Non-tgt", "Target"])
            for r in range(2):
                for c in range(2):
                    ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                            color="white" if cm[r, c] > cm.max() / 2 else "black")
            plt.colorbar(im, ax=ax)

    plt.tight_layout()
    path = os.path.join(save_dir,
                        f"confusion_matrices_{cv_name.lower().replace(' ','_')}_v2.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_v1_v2_comparison(v1_json_path, v2_results, save_dir):
    """Side-by-side bar: v1 vs v2 on LOSO balanced accuracy."""
    if not os.path.exists(v1_json_path):
        print("  (v1 results not found — skipping v1 vs v2 comparison plot)")
        return

    with open(v1_json_path) as f:
        v1 = json.load(f)

    metrics = ["accuracy", "balanced_accuracy", "f1", "auc"]
    labels  = ["Accuracy", "Bal. Acc", "F1", "AUC"]
    models  = ["rLDA", "SVC"]
    colors  = {"v1": "#90CAF9", "v2": "#1565C0"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("v1 vs v2  —  LOSO Performance", fontsize=13, fontweight="bold")

    for ax, mname in zip(axes, models):
        v1_vals = [v1["loso"][mname][m]["mean"] for m in metrics]
        v2_vals = [v2_results["loso"][mname][m]["mean"] for m in metrics]
        v1_err  = [v1["loso"][mname][m]["std"]  for m in metrics]
        v2_err  = [v2_results["loso"][mname][m]["std"]  for m in metrics]

        x     = np.arange(len(metrics))
        width = 0.35
        b1 = ax.bar(x - width/2, v1_vals, width, yerr=v1_err,
                    label="v1 (no xDAWN)", color=colors["v1"], alpha=0.9, capsize=5)
        b2 = ax.bar(x + width/2, v2_vals, width, yerr=v2_err,
                    label="v2 (xDAWN+CAR)", color=colors["v2"], alpha=0.9, capsize=5)

        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.10)
        ax.set_ylabel("Score")
        ax.set_title(f"{mname} — LOSO", fontsize=11)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance")
        ax.legend(fontsize=8)
        ax.yaxis.grid(True, alpha=0.3)

        for bar in list(b1) + list(b2):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    path = os.path.join(save_dir, "v1_vs_v2_loso_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  P300 Classification v2: xDAWN + rLDA / SVC")
    print("  xDAWN spatial filter (2 components) inside each CV fold")
    print("  Feature: xDAWN → 200-500ms → 15 bins → 30-dim")
    print("  Strategy A: Within-Subject StratifiedKFold(5)")
    print("  Strategy B: Cross-Subject Leave-One-Subject-Out")
    print("=" * 70)

    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(
            f"v2 features file not found: {FEATURES_PATH}\n"
            "Run features/extract_p300_features_v2.py first."
        )

    print(f"\n  Loading: {FEATURES_PATH}")
    data = np.load(FEATURES_PATH, allow_pickle=True)
    X_epochs   = data["X_epochs"]    # (n, 8, 250) — used for xDAWN
    y          = data["y"]
    subject_id = data["subject_id"]

    print(f"  X_epochs   : {X_epochs.shape}")
    print(f"  y          : {y.shape}  (target rate: {y.mean()*100:.1f}%)")
    print(f"  Subjects   : {np.unique(subject_id)}")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    results = {}

    # -----------------------------------------------------------------------
    # Strategy A: Within-Subject
    # -----------------------------------------------------------------------
    print("\n\n" + "#"*70)
    print("# STRATEGY A: Within-Subject StratifiedKFold(5)")
    print("#"*70)

    lda_ws_summary, _ = within_subject_cv(X_epochs, y, subject_id, "rLDA")
    svc_ws_summary, _ = within_subject_cv(X_epochs, y, subject_id, "SVC")

    results["within_subject"] = {
        "rLDA": lda_ws_summary,
        "SVC" : svc_ws_summary,
    }
    plot_comparison_bar(lda_ws_summary, svc_ws_summary, "Within-Subject", PLOTS_DIR)

    # -----------------------------------------------------------------------
    # Strategy B: LOSO
    # -----------------------------------------------------------------------
    print("\n\n" + "#"*70)
    print("# STRATEGY B: Cross-Subject Leave-One-Subject-Out")
    print("#"*70)

    lda_loso_summary, _, lda_subj_res = loso_cv(X_epochs, y, subject_id, "rLDA")
    svc_loso_summary, _, svc_subj_res = loso_cv(X_epochs, y, subject_id, "SVC")

    results["loso"] = {
        "rLDA": lda_loso_summary,
        "SVC" : svc_loso_summary,
        "per_subject_rLDA": {str(k): v for k, v in lda_subj_res.items()},
        "per_subject_SVC" : {str(k): v for k, v in svc_subj_res.items()},
    }

    plot_comparison_bar(lda_loso_summary, svc_loso_summary, "LOSO", PLOTS_DIR)
    plot_per_subject_bacc(lda_subj_res, svc_subj_res, "LOSO", PLOTS_DIR)
    plot_confusion_matrices(lda_subj_res, svc_subj_res,
                            sorted(lda_subj_res.keys()), "LOSO", PLOTS_DIR)

    # -----------------------------------------------------------------------
    # v1 vs v2 comparison
    # -----------------------------------------------------------------------
    v1_json = os.path.join(RESULTS_DIR, "classification_results.json")
    plot_v1_v2_comparison(v1_json, results, PLOTS_DIR)

    # -----------------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------------
    json_path = os.path.join(RESULTS_DIR, "classification_results_v2.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {json_path}")

    # -----------------------------------------------------------------------
    # Final table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  FINAL COMPARISON TABLE  (v2: xDAWN + CAR)")
    print("=" * 70)
    print(f"  {'Metric':<22} {'rLDA (WS)':>12} {'SVC (WS)':>12} {'rLDA (LOSO)':>14} {'SVC (LOSO)':>12}")
    print(f"  {'-'*74}")
    for metric in ["accuracy", "balanced_accuracy", "f1", "auc"]:
        lda_ws   = lda_ws_summary[metric]
        svc_ws   = svc_ws_summary[metric]
        lda_loso = lda_loso_summary[metric]
        svc_loso = svc_loso_summary[metric]
        print(f"  {metric:<22} "
              f"{lda_ws['mean']:.3f}±{lda_ws['std']:.3f}  "
              f"{svc_ws['mean']:.3f}±{svc_ws['std']:.3f}  "
              f"{lda_loso['mean']:.3f}±{lda_loso['std']:.3f}  "
              f"{svc_loso['mean']:.3f}±{svc_loso['std']:.3f}")
    print(f"  {'='*74}")
    print("\n  Done!\n")


if __name__ == "__main__":
    main()
