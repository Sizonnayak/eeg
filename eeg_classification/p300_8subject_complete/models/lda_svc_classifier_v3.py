"""
P300 Classification v3: Tuned xDAWN (k=4,6) + rLDA / SVC
===========================================================
Fixes from v2:
  1. Labels are now correct: y=1 = TARGET (P300-evoking, rare 16.7%)
  2. xDAWN components swept: k=2, k=4, k=6 (avoids the 30-dim bottleneck)
  3. Feature size after xDAWN:
       k=2 → 2×15 = 30-dim
       k=4 → 4×15 = 60-dim
       k=6 → 6×15 = 90-dim
     (max k=6 because we only have 8 channels)
  4. Baseline comparison at k=0 = v1 features (120-dim, no xDAWN)
  5. Same temporal recipe: 200–500ms → 15 bins → flatten
  6. xDAWN fitted inside each CV fold (no leakage)

Evaluation:
  A. Within-subject StratifiedKFold(5)
  B. Cross-subject LOSO

Key metric to watch: AUC-ROC (threshold-free) and Balanced Accuracy
  F1 is sensitive to threshold — AUC + BAcc tell the true story.

Input : features/p300_features_v2.npz  (X_epochs + correct y)
        (uses v2 preprocessed data: 0.1-30Hz + CAR)

Usage:
    python models/lda_svc_classifier_v3.py
"""

import os
import json
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from scipy.linalg import eigh
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FEATURES_PATH = "/Users/siznayak/Documents/others/MTech/EEG_Classification/features/p300_features_v2.npz"
RESULTS_DIR   = "/Users/siznayak/Documents/others/MTech/EEG_Classification/results"
PLOTS_DIR     = os.path.join(RESULTS_DIR, "plots_v3")

CH_NAMES = ["Fz", "Cz", "P3", "Pz", "P4", "PO7", "PO8", "Oz"]
N_CHANNELS = 8

# Feature constants (same temporal recipe as v1)
FS             = 250
PRE_MS         = 200
P300_START_IDX = int((200 + PRE_MS) * FS / 1000)   # 100
P300_END_IDX   = int((500 + PRE_MS) * FS / 1000)   # 175
N_BINS         = 15
BIN_SIZE       = (P300_END_IDX - P300_START_IDX) // N_BINS   # 5

# xDAWN component counts to sweep
# k=0 means no xDAWN (baseline v1 features, 120-dim)
XDAWN_K_VALUES = [0, 2, 4, 6]

# ---------------------------------------------------------------------------
# xDAWN (numpy, corrected — y=1 is TARGET)
# ---------------------------------------------------------------------------

def fit_xdawn(X_train, y_train, n_components):
    """
    Fit xDAWN spatial filters on training epochs.
    y_train==1 is the TARGET class (P300-evoking).

    Returns W: (n_channels, n_components)
    """
    n, n_ch, n_t = X_train.shape
    target_mask  = y_train == 1

    if target_mask.sum() < 2:
        # fallback: return identity-like filter
        return np.eye(n_ch)[:, :n_components]

    # Signal template: average target ERP
    erp_target = X_train[target_mask].mean(axis=0)   # (n_ch, n_t)

    # C_signal: outer product of template
    A = erp_target.T                                  # (n_t, n_ch)
    C_signal = A.T @ A                                # (n_ch, n_ch)

    # C_noise: total data covariance (signal + noise)
    X_flat  = X_train.transpose(0, 2, 1).reshape(-1, n_ch)  # (n*n_t, n_ch)
    C_noise = X_flat.T @ X_flat                              # (n_ch, n_ch)

    # Tikhonov regularisation (1% of mean eigenvalue)
    reg     = 0.01 * np.trace(C_noise) / n_ch
    C_noise += np.eye(n_ch) * reg

    # Generalised eigenvalue: C_signal w = λ C_noise w
    # eigh returns ascending order → top n_components are last columns
    _, eigenvectors = eigh(C_signal, C_noise)
    W = eigenvectors[:, -n_components:]    # (n_ch, n_components)
    return W


def apply_xdawn(X, W):
    """
    Project epochs through spatial filter W.
    X: (n, n_ch, n_t)  →  (n, n_components, n_t)
    einsum 'nct,ck->nkt': n=epochs, c=channels, t=time, k=components
    """
    return np.einsum("nct,ck->nkt", X, W)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(X_proj):
    """
    Crop 200-500ms → 15 bins → flatten.
    X_proj: (n, n_comp, 250)  →  (n, n_comp*15)
    """
    n, n_comp, _ = X_proj.shape
    X_p300   = X_proj[:, :, P300_START_IDX:P300_END_IDX]              # (n, n_comp, 75)
    X_binned = X_p300.reshape(n, n_comp, N_BINS, BIN_SIZE).mean(axis=3)  # (n, n_comp, 15)
    return X_binned.reshape(n, -1).astype(np.float32)                  # (n, n_comp*15)


def get_features(X_train, y_train, X_test, k):
    """
    k=0: use raw epochs → 8×15 = 120-dim (same as v1)
    k>0: apply xDAWN(k) → k×15 dims
    """
    if k == 0:
        feat_tr = extract_features(X_train)
        feat_te = extract_features(X_test)
    else:
        W       = fit_xdawn(X_train, y_train, n_components=k)
        feat_tr = extract_features(apply_xdawn(X_train, W))
        feat_te = extract_features(apply_xdawn(X_test,  W))
    return feat_tr, feat_te


# ---------------------------------------------------------------------------
# Helpers
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
    keys    = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "auc"]
    summary = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics_list if m[k] is not None]
        summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return summary


# ---------------------------------------------------------------------------
# CV functions
# ---------------------------------------------------------------------------

def within_subject_cv(X_epochs, y, subject_id, model_name, k_xdawn, n_splits=5):
    skf              = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_fold_metrics = []

    for sid in np.unique(subject_id):
        mask   = subject_id == sid
        X_subj = X_epochs[mask]
        y_subj = y[mask]

        subj_fold_metrics = []
        for tr, te in skf.split(X_subj, y_subj):
            X_tr_ep, X_te_ep = X_subj[tr], X_subj[te]
            y_tr, y_te       = y_subj[tr], y_subj[te]

            feat_tr, feat_te = get_features(X_tr_ep, y_tr, X_te_ep, k_xdawn)

            scaler  = StandardScaler()
            X_tr_sc = scaler.fit_transform(feat_tr)
            X_te_sc = scaler.transform(feat_te)

            model   = build_model(model_name)
            model.fit(X_tr_sc, y_tr)

            y_pred  = model.predict(X_te_sc)
            y_score = get_score(model, X_te_sc)
            subj_fold_metrics.append(compute_metrics(y_te, y_pred, y_score))

        all_fold_metrics.extend(subj_fold_metrics)

    return summarise(all_fold_metrics)


def loso_cv(X_epochs, y, subject_id, model_name, k_xdawn):
    logo            = LeaveOneGroupOut()
    fold_metrics    = []
    per_subject_res = {}

    for train_idx, test_idx in logo.split(X_epochs, y, groups=subject_id):
        test_sid = subject_id[test_idx[0]]
        X_tr_ep  = X_epochs[train_idx]
        X_te_ep  = X_epochs[test_idx]
        y_tr     = y[train_idx]
        y_te     = y[test_idx]

        feat_tr, feat_te = get_features(X_tr_ep, y_tr, X_te_ep, k_xdawn)

        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(feat_tr)
        X_te_sc = scaler.transform(feat_te)

        model   = build_model(model_name)
        model.fit(X_tr_sc, y_tr)

        y_pred  = model.predict(X_te_sc)
        y_score = get_score(model, X_te_sc)
        m       = compute_metrics(y_te, y_pred, y_score)
        fold_metrics.append(m)
        per_subject_res[int(test_sid)] = m

    return summarise(fold_metrics), per_subject_res


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_xdawn_sweep(sweep_results, cv_name, save_dir):
    """
    Plot AUC and Balanced Accuracy vs xDAWN components (k=0,2,4,6)
    for both rLDA and SVC.
    """
    k_labels  = [f"k={k}\n({'no xDAWN' if k==0 else f'{k*15}d'})" for k in XDAWN_K_VALUES]
    metrics   = ["auc", "balanced_accuracy"]
    m_labels  = ["AUC-ROC", "Balanced Accuracy"]
    colors    = {"rLDA": "#2196F3", "SVC": "#FF5722"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"xDAWN Component Sweep  —  {cv_name}  (v3)", fontsize=13, fontweight="bold")

    for ax, metric, m_label in zip(axes, metrics, m_labels):
        for model_name in ["rLDA", "SVC"]:
            means = [sweep_results[k][model_name][metric]["mean"] for k in XDAWN_K_VALUES]
            stds  = [sweep_results[k][model_name][metric]["std"]  for k in XDAWN_K_VALUES]
            ax.errorbar(range(len(XDAWN_K_VALUES)), means, yerr=stds,
                        marker="o", label=model_name,
                        color=colors[model_name], capsize=5, linewidth=2)

        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance")
        ax.set_xticks(range(len(XDAWN_K_VALUES)))
        ax.set_xticklabels(k_labels)
        ax.set_ylim(0.4, 1.0)
        ax.set_ylabel(m_label)
        ax.set_title(m_label)
        ax.legend()
        ax.yaxis.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(save_dir, f"xdawn_sweep_{cv_name.lower().replace(' ','_')}_v3.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_best_loso_per_subject(per_subject_results, best_k, model_name, save_dir):
    """Per-subject AUC for the best xDAWN config."""
    subjects = sorted(per_subject_results.keys())
    aucs     = [per_subject_results[s]["auc"] for s in subjects]
    baccs    = [per_subject_results[s]["balanced_accuracy"] for s in subjects]
    labels   = [f"S{s:02d}" for s in subjects]

    x     = np.arange(len(subjects))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, aucs,  width, label="AUC-ROC",         color="#2196F3", alpha=0.85)
    ax.bar(x + width/2, baccs, width, label="Bal. Accuracy",    color="#4CAF50", alpha=0.85)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"v3 LOSO Per-Subject  —  {model_name}  k={best_k}",
                 fontsize=12, fontweight="bold")
    ax.legend()
    ax.yaxis.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(save_dir, f"v3_loso_per_subject_{model_name}_k{best_k}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_v1_v2_v3_summary(v1_json, v2_json, ws_results, loso_results, save_dir):
    """4-panel comparison: v1, v2-best, v3-best on WS and LOSO."""
    try:
        with open(v1_json) as f: v1 = json.load(f)
        with open(v2_json) as f: v2 = json.load(f)
    except FileNotFoundError:
        print("  (Skipping v1/v2/v3 comparison — JSON files not found)")
        return

    # Find best k for v3 by rLDA LOSO AUC
    best_k_rdla = max(XDAWN_K_VALUES,
                      key=lambda k: loso_results[k]["rLDA"]["auc"]["mean"])

    metrics = ["auc", "balanced_accuracy"]
    labels  = ["AUC-ROC", "Bal. Accuracy"]
    configs = [
        ("v1\n(baseline)", v1["within_subject"]["rLDA"], v1["loso"]["rLDA"]),
        (f"v3 k={best_k_rdla}\n(xDAWN best)",
         ws_results[best_k_rdla]["rLDA"], loso_results[best_k_rdla]["rLDA"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("v1 vs v3 Best  —  rLDA  |  Within-Subject & LOSO",
                 fontsize=13, fontweight="bold")

    cv_names  = ["Within-Subject", "LOSO"]
    cv_keys   = [0, 1]
    colors    = ["#90CAF9", "#1565C0"]

    for ax, metric, m_label in zip(axes, metrics, labels):
        for cv_idx, cv_name in enumerate(cv_names):
            x       = np.arange(len(configs))
            width   = 0.35
            bar_vals = []
            bar_errs = []
            for cfg_label, ws_d, loso_d in configs:
                d = ws_d if cv_idx == 0 else loso_d
                bar_vals.append(d[metric]["mean"])
                bar_errs.append(d[metric]["std"])

            offset = (cv_idx - 0.5) * width
            bars   = ax.bar(x + offset, bar_vals, width, yerr=bar_errs,
                            label=cv_name, color=colors[cv_idx],
                            alpha=0.9, capsize=5)

        ax.set_xticks(x)
        ax.set_xticklabels([c[0] for c in configs])
        ax.set_ylim(0, 1.0)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance")
        ax.set_ylabel(m_label)
        ax.set_title(m_label)
        ax.legend(fontsize=8)
        ax.yaxis.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(save_dir, "v1_v2_v3_summary.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  P300 Classification v3: xDAWN Sweep (k=0,2,4,6) + rLDA / SVC")
    print("  k=0 = v1 baseline (120-dim), k>0 = xDAWN → k×15 dims")
    print("  TARGET = y=1 (P300-evoking, rare ~16.7%)")
    print("=" * 70)

    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Features not found: {FEATURES_PATH}")

    print(f"\n  Loading: {FEATURES_PATH}")
    data       = np.load(FEATURES_PATH, allow_pickle=True)
    X_epochs   = data["X_epochs"]    # (n, 8, 250)
    y          = data["y"]
    subject_id = data["subject_id"]

    print(f"  X_epochs   : {X_epochs.shape}")
    print(f"  y          : {y.shape}  (target y=1: {y.mean()*100:.1f}%)")
    print(f"  Subjects   : {np.unique(subject_id)}")

    os.makedirs(PLOTS_DIR, exist_ok=True)

    ws_results   = {}
    loso_results = {}

    print("\n" + "=" * 70)
    print("  SWEEPING xDAWN COMPONENTS")
    print("=" * 70)

    for k in XDAWN_K_VALUES:
        k_label  = f"k={k} ({'no xDAWN, 120d' if k==0 else f'xDAWN, {k*15}d'})"
        feat_dim = 8 * N_BINS if k == 0 else k * N_BINS
        print(f"\n{'─'*70}")
        print(f"  xDAWN components = {k}   →   feature dim = {feat_dim}")
        print(f"{'─'*70}")

        ws_results[k]   = {}
        loso_results[k] = {}

        for model_name in ["rLDA", "SVC"]:
            print(f"\n  [{model_name}] Within-Subject StratifiedKFold(5) ...")
            ws_sum = within_subject_cv(X_epochs, y, subject_id, model_name, k)
            ws_results[k][model_name] = ws_sum
            print(f"  → BAcc={ws_sum['balanced_accuracy']['mean']:.3f}±{ws_sum['balanced_accuracy']['std']:.3f}  "
                  f"AUC={ws_sum['auc']['mean']:.3f}±{ws_sum['auc']['std']:.3f}  "
                  f"F1={ws_sum['f1']['mean']:.3f}±{ws_sum['f1']['std']:.3f}")

            print(f"\n  [{model_name}] LOSO ...")
            loso_sum, loso_subj = loso_cv(X_epochs, y, subject_id, model_name, k)
            loso_results[k][model_name] = loso_sum
            loso_results[k][f"{model_name}_per_subject"] = {str(s): v for s, v in loso_subj.items()}
            print(f"  → BAcc={loso_sum['balanced_accuracy']['mean']:.3f}±{loso_sum['balanced_accuracy']['std']:.3f}  "
                  f"AUC={loso_sum['auc']['mean']:.3f}±{loso_sum['auc']['std']:.3f}  "
                  f"F1={loso_sum['f1']['mean']:.3f}±{loso_sum['f1']['std']:.3f}")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    plot_xdawn_sweep(ws_results,   "Within-Subject", PLOTS_DIR)
    plot_xdawn_sweep(loso_results, "LOSO",           PLOTS_DIR)

    # Best k for rLDA LOSO by AUC
    best_k_rdla = max(XDAWN_K_VALUES,
                      key=lambda k: loso_results[k]["rLDA"]["auc"]["mean"])
    best_k_svc  = max(XDAWN_K_VALUES,
                      key=lambda k: loso_results[k]["SVC"]["auc"]["mean"])
    print(f"\n  Best k for rLDA LOSO (AUC): k={best_k_rdla}")
    print(f"  Best k for SVC  LOSO (AUC): k={best_k_svc}")

    lda_best_subj = {int(s): v
                     for s, v in loso_results[best_k_rdla]["rLDA_per_subject"].items()}
    svc_best_subj = {int(s): v
                     for s, v in loso_results[best_k_svc]["SVC_per_subject"].items()}
    plot_best_loso_per_subject(lda_best_subj, best_k_rdla, "rLDA", PLOTS_DIR)
    plot_best_loso_per_subject(svc_best_subj, best_k_svc,  "SVC",  PLOTS_DIR)

    plot_v1_v2_v3_summary(
        os.path.join(RESULTS_DIR, "classification_results.json"),
        os.path.join(RESULTS_DIR, "classification_results_v2.json"),
        ws_results, loso_results, PLOTS_DIR
    )

    # -----------------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------------
    # Convert int keys to str for JSON
    results = {
        "within_subject": {str(k): ws_results[k]   for k in XDAWN_K_VALUES},
        "loso"          : {str(k): loso_results[k]  for k in XDAWN_K_VALUES},
    }
    json_path = os.path.join(RESULTS_DIR, "classification_results_v3.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {json_path}")

    # -----------------------------------------------------------------------
    # Final summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  FINAL SWEEP TABLE  (rLDA)")
    print("=" * 70)
    print(f"  {'k':>4}  {'Feat dim':>10}  {'WS BAcc':>10}  {'WS AUC':>10}  {'LOSO BAcc':>12}  {'LOSO AUC':>10}")
    print(f"  {'-'*66}")
    for k in XDAWN_K_VALUES:
        fd  = 8 * N_BINS if k == 0 else k * N_BINS
        ws  = ws_results[k]["rLDA"]
        lo  = loso_results[k]["rLDA"]
        print(f"  {k:>4}  {fd:>10}  "
              f"{ws['balanced_accuracy']['mean']:>8.3f}±{ws['balanced_accuracy']['std']:.3f}  "
              f"{ws['auc']['mean']:>8.3f}±{ws['auc']['std']:.3f}  "
              f"{lo['balanced_accuracy']['mean']:>10.3f}±{lo['balanced_accuracy']['std']:.3f}  "
              f"{lo['auc']['mean']:>8.3f}±{lo['auc']['std']:.3f}")
    print(f"  {'='*66}")
    print("\n  Done!\n")


if __name__ == "__main__":
    main()
