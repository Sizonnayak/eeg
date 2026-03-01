"""
P300 Classification: rLDA vs SVC
==================================
Two evaluation strategies:
  A. Within-subject : StratifiedKFold(5) per subject
  B. Cross-subject  : Leave-One-Subject-Out (LOSO)

Both strategies:
  - StandardScaler fitted only on training fold
  - class_weight='balanced' to handle 83/17 imbalance
  - Metrics: Accuracy, Balanced Accuracy, Precision, Recall, F1, AUC-ROC

Models:
  1. rLDA  - LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
  2. SVC   - SVC(kernel='linear', class_weight='balanced', C=0.1)
             (linear kernel recommended for high-dim ERP features)

Usage:
    python models/lda_svc_classifier.py
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
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FEATURES_PATH = "/Users/siznayak/Documents/others/MTech/EEG_Classification/features/p300_features.npz"
RESULTS_DIR   = "/Users/siznayak/Documents/others/MTech/EEG_Classification/results"
PLOTS_DIR     = os.path.join(RESULTS_DIR, "plots")

CH_NAMES = ["Fz", "Cz", "P3", "Pz", "P4", "PO7", "PO8", "Oz"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_score=None):
    metrics = {
        "accuracy"          : float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy" : float(balanced_accuracy_score(y_true, y_pred)),
        "precision"         : float(precision_score(y_true, y_pred, zero_division=0)),
        "recall"            : float(recall_score(y_true, y_pred, zero_division=0)),
        "f1"                : float(f1_score(y_true, y_pred, zero_division=0)),
        "auc"               : float(roc_auc_score(y_true, y_score)) if y_score is not None else None,
        "confusion_matrix"  : confusion_matrix(y_true, y_pred).tolist(),
    }
    return metrics


def get_score(model, X_test):
    """Return probability or decision-function scores for AUC."""
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


def print_fold(fold_label, metrics):
    print(f"    {fold_label:30s}  "
          f"Acc={metrics['accuracy']:.3f}  "
          f"BAcc={metrics['balanced_accuracy']:.3f}  "
          f"F1={metrics['f1']:.3f}  "
          f"AUC={metrics['auc']:.3f}")


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
# Strategy A: Within-Subject  (StratifiedKFold per subject)
# ---------------------------------------------------------------------------

def within_subject_cv(X, y, subject_id, model_name, n_splits=5):
    print(f"\n{'='*70}")
    print(f"  Within-Subject CV  |  {model_name}  |  StratifiedKFold(k={n_splits})")
    print(f"{'='*70}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_fold_metrics = []

    for sid in np.unique(subject_id):
        mask   = subject_id == sid
        X_subj = X[mask]
        y_subj = y[mask]

        subj_fold_metrics = []
        for fold, (tr, te) in enumerate(skf.split(X_subj, y_subj)):
            X_tr, X_te = X_subj[tr], X_subj[te]
            y_tr, y_te = y_subj[tr], y_subj[te]

            scaler = StandardScaler()
            X_tr   = scaler.fit_transform(X_tr)
            X_te   = scaler.transform(X_te)

            model = build_model(model_name)
            model.fit(X_tr, y_tr)

            y_pred  = model.predict(X_te)
            y_score = get_score(model, X_te)
            m       = compute_metrics(y_te, y_pred, y_score)
            subj_fold_metrics.append(m)

        subj_summary = summarise(subj_fold_metrics)
        print(f"  S{sid:02d}  Acc={subj_summary['accuracy']['mean']:.3f}±{subj_summary['accuracy']['std']:.3f}  "
              f"F1={subj_summary['f1']['mean']:.3f}±{subj_summary['f1']['std']:.3f}  "
              f"AUC={subj_summary['auc']['mean']:.3f}±{subj_summary['auc']['std']:.3f}")
        all_fold_metrics.extend(subj_fold_metrics)

    overall = summarise(all_fold_metrics)
    print_summary(model_name, f"Within-Subject k={n_splits}", overall)
    return overall, all_fold_metrics


# ---------------------------------------------------------------------------
# Strategy B: Cross-Subject LOSO
# ---------------------------------------------------------------------------

def loso_cv(X, y, subject_id, model_name):
    print(f"\n{'='*70}")
    print(f"  Cross-Subject LOSO  |  {model_name}  |  Leave-One-Subject-Out")
    print(f"{'='*70}")

    logo             = LeaveOneGroupOut()
    fold_metrics     = []
    per_subject_res  = {}

    for train_idx, test_idx in logo.split(X, y, groups=subject_id):
        test_sid = subject_id[test_idx[0]]
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_te   = scaler.transform(X_te)

        model = build_model(model_name)
        model.fit(X_tr, y_tr)

        y_pred  = model.predict(X_te)
        y_score = get_score(model, X_te)
        m       = compute_metrics(y_te, y_pred, y_score)

        print_fold(f"Test S{test_sid:02d}", m)
        fold_metrics.append(m)
        per_subject_res[int(test_sid)] = m

    overall = summarise(fold_metrics)
    print_summary(model_name, "LOSO", overall)
    return overall, fold_metrics, per_subject_res


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrices(lda_res, svc_res, subject_ids, cv_name, save_dir):
    """Plot side-by-side confusion matrices for each test subject."""
    n = len(subject_ids)
    fig, axes = plt.subplots(n, 2, figsize=(8, n * 3))
    fig.suptitle(f"Confusion Matrices – {cv_name}", fontsize=13, fontweight="bold")

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
    path = os.path.join(save_dir, f"confusion_matrices_{cv_name.lower().replace(' ', '_')}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_comparison_bar(lda_summary, svc_summary, cv_name, save_dir):
    """Bar chart comparing rLDA vs SVC on key metrics."""
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
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"rLDA vs SVC  –  {cv_name}", fontsize=12, fontweight="bold")
    ax.legend()
    ax.yaxis.grid(True, alpha=0.3)

    # Annotate bars
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(save_dir, f"comparison_{cv_name.lower().replace(' ', '_')}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_per_subject_f1(lda_subj, svc_subj, cv_name, save_dir):
    """Per-subject F1 for both models."""
    subjects = sorted(lda_subj.keys())
    lda_f1   = [lda_subj[s]["f1"] for s in subjects]
    svc_f1   = [svc_subj[s]["f1"] for s in subjects]
    labels   = [f"S{s:02d}" for s in subjects]

    x     = np.arange(len(subjects))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, lda_f1, width, label="rLDA",      color="#2196F3", alpha=0.85)
    ax.bar(x + width/2, svc_f1, width, label="SVC (linear)", color="#FF5722", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1-Score (target class)")
    ax.set_title(f"Per-Subject F1  –  {cv_name}", fontsize=12, fontweight="bold")
    ax.legend()
    ax.yaxis.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f"per_subject_f1_{cv_name.lower().replace(' ', '_')}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_lda_channel_weights(X, y, subject_id, save_dir):
    """
    Fit one rLDA on all data, visualise mean absolute weight per channel.
    Shows which channels the LDA focuses on.
    """
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    lda    = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
    lda.fit(X_sc, y)

    # Coefficients: (1, 120) for binary LDA  →  reshape to (8, 15)
    coefs = lda.coef_[0].reshape(len(CH_NAMES), 15)
    ch_importance = np.abs(coefs).mean(axis=1)   # mean over bins per channel

    fig, ax = plt.subplots(figsize=(8, 4))
    colors  = ["#E74C3C" if c == ch_importance.max() else "#3498DB" for c in ch_importance]
    ax.bar(CH_NAMES, ch_importance, color=colors, alpha=0.85)
    ax.set_ylabel("Mean |LDA weight| (averaged over 15 bins)")
    ax.set_title("rLDA Channel Importance (all subjects)", fontsize=12, fontweight="bold")
    ax.yaxis.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "lda_channel_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  P300 Classification: rLDA vs SVC")
    print("  Strategy A: Within-Subject StratifiedKFold(5)")
    print("  Strategy B: Cross-Subject Leave-One-Subject-Out")
    print("=" * 70)

    # Load features
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(
            f"Features file not found: {FEATURES_PATH}\n"
            "Run features/extract_p300_features.py first."
        )

    print(f"\n  Loading features: {FEATURES_PATH}")
    data = np.load(FEATURES_PATH, allow_pickle=True)
    X          = data["X_features"]    # (33473, 120)
    y          = data["y"]             # (33473,)
    subject_id = data["subject_id"]    # (33473,)

    print(f"  X_features : {X.shape}")
    print(f"  y           : {y.shape}  (target rate: {y.mean()*100:.1f}%)")
    print(f"  Subjects    : {np.unique(subject_id)}")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    results = {}

    # -----------------------------------------------------------------------
    # Strategy A: Within-Subject
    # -----------------------------------------------------------------------
    print("\n\n" + "#"*70)
    print("# STRATEGY A: Within-Subject StratifiedKFold(5)")
    print("#"*70)

    lda_ws_summary, _ = within_subject_cv(X, y, subject_id, "rLDA",  n_splits=5)
    svc_ws_summary, _ = within_subject_cv(X, y, subject_id, "SVC",   n_splits=5)

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

    lda_loso_summary, _, lda_subj_res = loso_cv(X, y, subject_id, "rLDA")
    svc_loso_summary, _, svc_subj_res = loso_cv(X, y, subject_id, "SVC")

    results["loso"] = {
        "rLDA": lda_loso_summary,
        "SVC" : svc_loso_summary,
        "per_subject_rLDA": {str(k): v for k, v in lda_subj_res.items()},
        "per_subject_SVC" : {str(k): v for k, v in svc_subj_res.items()},
    }

    plot_comparison_bar(lda_loso_summary, svc_loso_summary, "LOSO", PLOTS_DIR)
    plot_per_subject_f1(lda_subj_res, svc_subj_res, "LOSO", PLOTS_DIR)
    plot_confusion_matrices(lda_subj_res, svc_subj_res,
                            sorted(lda_subj_res.keys()), "LOSO", PLOTS_DIR)

    # -----------------------------------------------------------------------
    # LDA channel importance
    # -----------------------------------------------------------------------
    plot_lda_channel_weights(X, y, subject_id, PLOTS_DIR)

    # -----------------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------------
    json_path = os.path.join(RESULTS_DIR, "classification_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {json_path}")

    # -----------------------------------------------------------------------
    # Final comparison table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  FINAL COMPARISON TABLE")
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
