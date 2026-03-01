"""
P300 Classification v4: Riemannian Alignment + Tangent Space + LDA
===================================================================
Pipeline:
  1. XdawnCovariances (pyriemann)
       For each epoch, apply xDAWN spatial filter (nfilter components),
       then compute a (2*nfilter+1, 2*nfilter+1) covariance matrix that
       captures both signal and noise subspaces.  This step is FITTED
       on training data inside each CV fold (no leakage).

  2. Euclidean Alignment (EA) — per-subject or per-fold reference
       Align covariance matrices across subjects by whitening each
       subject's data with its own mean covariance:
           C_aligned = R^{-1/2} @ C @ R^{-T/2}
       where R = geometric mean of training covariances.
       This removes inter-subject scale and orientation differences
       before tangent-space mapping.

  3. TangentSpace (pyriemann)
       Map aligned covariance matrices to the Euclidean tangent space
       at the Riemannian mean.  Output is a flat feature vector of
       size n*(n+1)/2 where n = 2*nfilter+1.
         nfilter=1 → n=3 → 6-dim
         nfilter=2 → n=5 → 15-dim
         nfilter=4 → n=9 → 45-dim

  4. StandardScaler + rLDA (or SVC)
       Same classifiers as v1/v2/v3.

Why Riemannian for LOSO?
  Raw amplitude features (v1-v3) are sensitive to inter-subject
  differences in electrode impedance, skull thickness, P300 latency
  shift, and scalp topography.  Covariance matrices capture the
  relative channel-channel relationships which are more stable.
  EA alignment further reduces the "reference bias" per subject.

Metrics now include PR-AUC (average precision) alongside ROC-AUC,
which better reflects detection quality under 83:17 imbalance.

nfilter sweep: 1, 2, 4   (default: 2, matching MOABB XdawnCovariances)

Input : preprocessing/p300_preprocessed_v2.npz  (0.1-30Hz + CAR epochs)
Output: results/classification_results_v4.json
        results/plots_v4/*.png

Usage:
    python models/lda_svc_classifier_v4.py
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
    roc_auc_score, average_precision_score,
    confusion_matrix
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_covariance

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PREPROC_PATH = "/Users/siznayak/Documents/others/MTech/EEG_Classification/preprocessing/p300_preprocessed_v2.npz"
RESULTS_DIR  = "/Users/siznayak/Documents/others/MTech/EEG_Classification/results"
PLOTS_DIR    = os.path.join(RESULTS_DIR, "plots_v4")

# xDAWN filter count sweep
NFILTER_VALUES = [1, 2, 4]
# feature dim per nfilter: n*(n+1)/2 where n = 2*nfilter+1
# nfilter=1 → 6, nfilter=2 → 15, nfilter=4 → 45

# ---------------------------------------------------------------------------
# Riemannian pipeline
# ---------------------------------------------------------------------------

def euclidean_alignment(C_train, C_test):
    """
    Per-subject Euclidean Alignment (EA).
    Whitens covariance matrices by the square-root inverse of their
    Euclidean mean, making the mean identity-like for both train and test.

    Reference:
        He & Wu (2019) "Transfer learning for brain-computer interfaces:
        A Euclidean space data alignment approach"
        IEEE TNSRE 28(8):1817-1830.

    Args:
        C_train : (n_train, p, p)  covariance matrices
        C_test  : (n_test,  p, p)

    Returns:
        C_train_aligned, C_test_aligned  (same shapes)
    """
    # Reference matrix = Euclidean mean of training covs
    R = C_train.mean(axis=0)                        # (p, p)

    # Square-root inverse of R
    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.maximum(eigvals, 1e-10)             # numerical safety
    R_invsqrt = eigvecs @ np.diag(eigvals ** -0.5) @ eigvecs.T  # (p,p)

    # Whiten: C_aligned = R^{-1/2} C R^{-1/2}
    C_tr_al = R_invsqrt @ C_train @ R_invsqrt
    C_te_al = R_invsqrt @ C_test  @ R_invsqrt
    return C_tr_al, C_te_al


def fit_transform_fold(X_train, y_train, X_test, nfilter, use_alignment=True):
    """
    Full Riemannian pipeline for one CV fold.

    1. Fit XdawnCovariances on training data
    2. Transform train + test → covariance matrices
    3. Euclidean alignment (optional, on by default)
    4. Fit TangentSpace on training covs
    5. Map train + test → tangent vectors
    6. Return flat feature arrays

    Args:
        X_train      : (n_train, 8, 250)
        y_train      : (n_train,)   y=1 = target
        X_test       : (n_test,  8, 250)
        nfilter      : int   xDAWN filter count
        use_alignment: bool  apply EA before tangent space

    Returns:
        feat_train : (n_train, feat_dim)
        feat_test  : (n_test,  feat_dim)
    """
    # Step 1 & 2: XdawnCovariances (fit on train, transform both)
    # Try estimators in order — some folds are ill-conditioned for specific ones
    fitted_cov = None
    for est in ["lwf", "oas", "cov", "scm"]:
        try:
            xdawn_cov = XdawnCovariances(
                nfilter         = nfilter,
                estimator       = est,
                xdawn_estimator = est,
                classes         = [1],   # target class = 1
            )
            xdawn_cov.fit(X_train, y_train)
            fitted_cov = xdawn_cov
            break
        except Exception:
            continue

    if fitted_cov is None:
        raise RuntimeError("XdawnCovariances failed with all estimators (lwf, oas, cov, scm)")

    C_train = fitted_cov.transform(X_train)   # (n_train, n, n)
    C_test  = fitted_cov.transform(X_test)    # (n_test,  n, n)

    # Step 3: Euclidean alignment
    if use_alignment:
        C_train, C_test = euclidean_alignment(C_train, C_test)

    # Step 4 & 5: TangentSpace (fit on train, transform both)
    ts = TangentSpace(metric="riemann")
    ts.fit(C_train)
    feat_train = ts.transform(C_train)   # (n_train, feat_dim)
    feat_test  = ts.transform(C_test)    # (n_test,  feat_dim)

    return feat_train.astype(np.float32), feat_test.astype(np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_score=None):
    m = {
        "accuracy"          : float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy" : float(balanced_accuracy_score(y_true, y_pred)),
        "precision"         : float(precision_score(y_true, y_pred, zero_division=0)),
        "recall"            : float(recall_score(y_true, y_pred, zero_division=0)),
        "f1"                : float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc"           : None,
        "auc_pr"            : None,   # PR-AUC (average precision)
        "confusion_matrix"  : confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_score is not None:
        m["auc_roc"] = float(roc_auc_score(y_true, y_score))
        m["auc_pr"]  = float(average_precision_score(y_true, y_score))
    return m


def get_score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]   # P(y=1=target)
    return model.decision_function(X)


def build_model(name):
    if name == "rLDA":
        return LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
    elif name == "SVC":
        return SVC(kernel="linear", C=0.1, class_weight="balanced",
                   probability=True, random_state=42, max_iter=5000)
    raise ValueError(f"Unknown model: {name}")


def summarise(fold_metrics):
    keys = ["accuracy", "balanced_accuracy", "precision", "recall",
            "f1", "auc_roc", "auc_pr"]
    summary = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics if m.get(k) is not None]
        summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return summary


def print_row(label, metrics):
    print(f"  {label:<30}  "
          f"BAcc={metrics['balanced_accuracy']:.3f}  "
          f"AUC={metrics['auc_roc']:.3f}  "
          f"PR={metrics['auc_pr']:.3f}  "
          f"F1={metrics['f1']:.3f}")


# ---------------------------------------------------------------------------
# CV strategies
# ---------------------------------------------------------------------------

def within_subject_cv(X_epochs, y, subject_id, model_name, nfilter, n_splits=5):
    skf       = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_folds = []

    for sid in np.unique(subject_id):
        mask   = subject_id == sid
        X_subj = X_epochs[mask]
        y_subj = y[mask]

        for tr, te in skf.split(X_subj, y_subj):
            X_tr, X_te = X_subj[tr], X_subj[te]
            y_tr, y_te = y_subj[tr], y_subj[te]

            feat_tr, feat_te = fit_transform_fold(X_tr, y_tr, X_te, nfilter)

            scaler  = StandardScaler()
            X_tr_sc = scaler.fit_transform(feat_tr)
            X_te_sc = scaler.transform(feat_te)

            model   = build_model(model_name)
            model.fit(X_tr_sc, y_tr)
            y_pred  = model.predict(X_te_sc)
            y_score = get_score(model, X_te_sc)
            all_folds.append(compute_metrics(y_te, y_pred, y_score))

    return summarise(all_folds)


def loso_cv(X_epochs, y, subject_id, model_name, nfilter):
    logo            = LeaveOneGroupOut()
    all_folds       = []
    per_subject_res = {}

    for train_idx, test_idx in logo.split(X_epochs, y, groups=subject_id):
        test_sid = subject_id[test_idx[0]]
        X_tr_ep  = X_epochs[train_idx]
        X_te_ep  = X_epochs[test_idx]
        y_tr     = y[train_idx]
        y_te     = y[test_idx]

        feat_tr, feat_te = fit_transform_fold(X_tr_ep, y_tr, X_te_ep, nfilter)

        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(feat_tr)
        X_te_sc = scaler.transform(feat_te)

        model   = build_model(model_name)
        model.fit(X_tr_sc, y_tr)
        y_pred  = model.predict(X_te_sc)
        y_score = get_score(model, X_te_sc)
        m       = compute_metrics(y_te, y_pred, y_score)

        print(f"    Test S{test_sid:02d}  "
              f"BAcc={m['balanced_accuracy']:.3f}  "
              f"AUC={m['auc_roc']:.3f}  "
              f"PR={m['auc_pr']:.3f}  "
              f"F1={m['f1']:.3f}")

        all_folds.append(m)
        per_subject_res[int(test_sid)] = m

    return summarise(all_folds), per_subject_res


# ---------------------------------------------------------------------------
# Threshold calibration helpers
# ---------------------------------------------------------------------------

def find_best_threshold(y_val, score_val):
    """
    Find the score threshold that maximises Youden's J = sensitivity + specificity - 1
    = balanced_accuracy * 2 - 1  on a validation set.

    Scans all unique score values as candidate thresholds.
    Returns the threshold that gives highest balanced accuracy.
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_val, score_val, pos_label=1)
    # Youden's J = tpr - fpr = sensitivity + specificity - 1
    j_scores = tpr - fpr
    best_idx  = np.argmax(j_scores)
    return float(thresholds[best_idx])


def compute_metrics_at_threshold(y_true, y_score, threshold):
    """Apply a custom threshold then compute all metrics."""
    y_pred = (y_score >= threshold).astype(int)
    return compute_metrics(y_true, y_pred, y_score)


def loso_cv_calibrated(X_epochs, y, subject_id, model_name, nfilter):
    """
    LOSO with per-fold threshold calibration.

    Strategy:
      - For each test subject (8 folds):
        1. Training data = 7 subjects
        2. Hold out one of the 7 training subjects as validation (leave-one-subject-out
           within the training set — i.e. use the subject closest to mean as val)
           Simpler: use LeaveOneGroupOut on the 7 training subjects, pick the fold
           with the highest validation AUC to select threshold.
           Even simpler (implemented here): use a random 10% stratified split of
           training data as validation for threshold selection.
        3. Fit model on remaining 90% train, find threshold on 10% val.
        4. Apply threshold to test subject.

    This gives a proper held-out threshold without touching the test subject at all.
    """
    from sklearn.model_selection import train_test_split

    logo            = LeaveOneGroupOut()
    all_folds       = []
    per_subject_res = {}

    for train_idx, test_idx in logo.split(X_epochs, y, groups=subject_id):
        test_sid = subject_id[test_idx[0]]
        X_tr_ep  = X_epochs[train_idx]
        X_te_ep  = X_epochs[test_idx]
        y_tr     = y[train_idx]
        y_te     = y[test_idx]

        # Split training data: 90% model training, 10% threshold calibration
        tr_idx, val_idx = train_test_split(
            np.arange(len(y_tr)), test_size=0.10,
            stratify=y_tr, random_state=42
        )
        X_fit_ep  = X_tr_ep[tr_idx];  y_fit  = y_tr[tr_idx]
        X_val_ep  = X_tr_ep[val_idx]; y_val  = y_tr[val_idx]

        # Fit Riemannian pipeline on training 90%
        feat_fit, feat_val = fit_transform_fold(X_fit_ep, y_fit, X_val_ep, nfilter)
        feat_fit2, feat_te = fit_transform_fold(X_fit_ep, y_fit, X_te_ep,  nfilter)

        scaler   = StandardScaler()
        fit_sc   = scaler.fit_transform(feat_fit)
        val_sc   = scaler.transform(feat_val)
        te_sc    = scaler.transform(feat_te)

        # Also need scaler for full-train features when predicting test
        # Re-fit on full training data for final model
        feat_full_tr, feat_te2 = fit_transform_fold(X_tr_ep, y_tr, X_te_ep, nfilter)
        scaler2  = StandardScaler()
        full_sc  = scaler2.fit_transform(feat_full_tr)
        te2_sc   = scaler2.transform(feat_te2)

        # Train calibration model on 90%
        cal_model = build_model(model_name)
        cal_model.fit(fit_sc, y_fit)
        val_score = get_score(cal_model, val_sc)
        threshold = find_best_threshold(y_val, val_score)

        # Train final model on all 7 training subjects
        final_model = build_model(model_name)
        final_model.fit(full_sc, y_tr)
        te_score = get_score(final_model, te2_sc)

        m = compute_metrics_at_threshold(y_te, te_score, threshold)

        print(f"    Test S{test_sid:02d}  thresh={threshold:.3f}  "
              f"BAcc={m['balanced_accuracy']:.3f}  "
              f"AUC={m['auc_roc']:.3f}  "
              f"PR={m['auc_pr']:.3f}  "
              f"F1={m['f1']:.3f}")

        all_folds.append(m)
        per_subject_res[int(test_sid)] = m

    return summarise(all_folds), per_subject_res


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_nfilter_sweep(ws_results, loso_results, save_dir):
    """AUC-ROC and PR-AUC vs nfilter for rLDA and SVC."""
    k_labels  = [f"nf={k}" for k in NFILTER_VALUES]
    metrics   = ["auc_roc", "auc_pr", "balanced_accuracy"]
    m_labels  = ["AUC-ROC", "PR-AUC", "Balanced Accuracy"]
    colors    = {"rLDA": "#2196F3", "SVC": "#FF5722"}

    for cv_name, cv_res in [("Within-Subject", ws_results), ("LOSO", loso_results)]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Riemannian v4  —  nfilter Sweep  —  {cv_name}",
                     fontsize=13, fontweight="bold")

        for ax, metric, m_label in zip(axes, metrics, m_labels):
            for mname in ["rLDA", "SVC"]:
                means = [cv_res[k][mname][metric]["mean"] for k in NFILTER_VALUES]
                stds  = [cv_res[k][mname][metric]["std"]  for k in NFILTER_VALUES]
                ax.errorbar(range(len(NFILTER_VALUES)), means, yerr=stds,
                            marker="o", label=mname,
                            color=colors[mname], capsize=5, linewidth=2)
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance")
            ax.set_xticks(range(len(NFILTER_VALUES)))
            ax.set_xticklabels(k_labels)
            ax.set_ylim(0.3, 1.0)
            ax.set_ylabel(m_label); ax.set_title(m_label)
            ax.legend(fontsize=8); ax.yaxis.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = os.path.join(save_dir,
                             f"v4_nfilter_sweep_{cv_name.lower().replace(' ','_')}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")


def plot_v1_v4_loso_comparison(v1_json, loso_results, best_nf, save_dir):
    """Side-by-side v1 vs v4-best on LOSO."""
    if not os.path.exists(v1_json):
        print("  (Skipping v1 vs v4 plot — v1 JSON not found)")
        return

    with open(v1_json) as f:
        v1 = json.load(f)

    # v1 uses "auc" key, v4 uses "auc_roc"
    # map for comparison
    metrics  = [("auc_roc", "auc", "AUC-ROC"),
                ("balanced_accuracy", "balanced_accuracy", "Bal. Accuracy")]
    models   = ["rLDA", "SVC"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"v1 (120d raw) vs v4 Riemannian (nf={best_nf})  —  LOSO",
                 fontsize=13, fontweight="bold")

    for ax, (v4_key, v1_key, label) in zip(axes, metrics):
        x     = np.arange(len(models))
        width = 0.35

        v1_vals = [v1["loso"][m][v1_key]["mean"] for m in models]
        v1_errs = [v1["loso"][m][v1_key]["std"]  for m in models]
        v4_vals = [loso_results[best_nf][m][v4_key]["mean"] for m in models]
        v4_errs = [loso_results[best_nf][m][v4_key]["std"]  for m in models]

        ax.bar(x - width/2, v1_vals, width, yerr=v1_errs,
               label="v1 (raw 120d)", color="#90CAF9", alpha=0.9, capsize=5)
        ax.bar(x + width/2, v4_vals, width, yerr=v4_errs,
               label=f"v4 Riemann nf={best_nf}", color="#1565C0", alpha=0.9, capsize=5)

        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance")
        ax.set_xticks(x); ax.set_xticklabels(models)
        ax.set_ylim(0, 1.0); ax.set_ylabel(label); ax.set_title(label)
        ax.legend(fontsize=8); ax.yaxis.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(save_dir, "v1_vs_v4_loso.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_per_subject_loso(per_subj, nf, model_name, save_dir):
    subjects = sorted(per_subj.keys())
    aucs  = [per_subj[s]["auc_roc"] for s in subjects]
    prs   = [per_subj[s]["auc_pr"]  for s in subjects]
    baccs = [per_subj[s]["balanced_accuracy"] for s in subjects]
    xlabels = [f"S{s:02d}" for s in subjects]

    x     = np.arange(len(subjects))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width, aucs,  width, label="AUC-ROC",      color="#2196F3", alpha=0.85)
    ax.bar(x,         prs,   width, label="PR-AUC",        color="#4CAF50", alpha=0.85)
    ax.bar(x + width, baccs, width, label="Bal. Accuracy", color="#FF9800", alpha=0.85)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance")
    ax.set_xticks(x); ax.set_xticklabels(xlabels)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title(f"v4 LOSO Per-Subject  —  {model_name}  nf={nf}",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.yaxis.grid(True, alpha=0.3)
    plt.tight_layout()

    fname = os.path.join(save_dir, f"v4_loso_per_subject_{model_name}_nf{nf}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  P300 Classification v4: Riemannian (EA + XdawnCov + TS + LDA)")
    print("  XdawnCovariances → Euclidean Alignment → TangentSpace → rLDA/SVC")
    print(f"  nfilter sweep: {NFILTER_VALUES}")
    print("  PR-AUC added alongside ROC-AUC")
    print("=" * 70)

    if not os.path.exists(PREPROC_PATH):
        raise FileNotFoundError(f"Not found: {PREPROC_PATH}\nRun preprocessing/p300_preprocess_v2.py first.")

    print(f"\n  Loading: {PREPROC_PATH}")
    data       = np.load(PREPROC_PATH, allow_pickle=True)
    X_epochs   = data["X"]           # (n, 8, 250)
    y          = data["y"]
    subject_id = data["subject_id"]

    print(f"  X_epochs   : {X_epochs.shape}")
    print(f"  y=1 (target): {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"  Subjects    : {np.unique(subject_id)}")

    os.makedirs(PLOTS_DIR, exist_ok=True)

    ws_results   = {}
    loso_results = {}

    # -----------------------------------------------------------------------
    # Sweep nfilter
    # -----------------------------------------------------------------------
    for nf in NFILTER_VALUES:
        n_dim = (2 * nf + 1)
        feat_dim = n_dim * (n_dim + 1) // 2
        print(f"\n{'='*70}")
        print(f"  nfilter={nf}  →  cov size {n_dim}×{n_dim}  →  tangent dim={feat_dim}")
        print(f"{'='*70}")

        ws_results[nf]   = {}
        loso_results[nf] = {}

        for model_name in ["rLDA", "SVC"]:
            print(f"\n  [{model_name}] Within-Subject StratifiedKFold(5) ...")
            ws_sum = within_subject_cv(X_epochs, y, subject_id, model_name, nf)
            ws_results[nf][model_name] = ws_sum
            print(f"  → BAcc={ws_sum['balanced_accuracy']['mean']:.3f}±{ws_sum['balanced_accuracy']['std']:.3f}  "
                  f"AUC={ws_sum['auc_roc']['mean']:.3f}±{ws_sum['auc_roc']['std']:.3f}  "
                  f"PR={ws_sum['auc_pr']['mean']:.3f}±{ws_sum['auc_pr']['std']:.3f}")

            print(f"\n  [{model_name}] LOSO ...")
            loso_sum, loso_subj = loso_cv(X_epochs, y, subject_id, model_name, nf)
            loso_results[nf][model_name] = loso_sum
            loso_results[nf][f"{model_name}_per_subject"] = {
                str(s): v for s, v in loso_subj.items()
            }
            print(f"  → BAcc={loso_sum['balanced_accuracy']['mean']:.3f}±{loso_sum['balanced_accuracy']['std']:.3f}  "
                  f"AUC={loso_sum['auc_roc']['mean']:.3f}±{loso_sum['auc_roc']['std']:.3f}  "
                  f"PR={loso_sum['auc_pr']['mean']:.3f}±{loso_sum['auc_pr']['std']:.3f}")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    plot_nfilter_sweep(ws_results, loso_results, PLOTS_DIR)

    best_nf_rdla = max(NFILTER_VALUES,
                       key=lambda k: loso_results[k]["rLDA"]["auc_roc"]["mean"])
    best_nf_svc  = max(NFILTER_VALUES,
                       key=lambda k: loso_results[k]["SVC"]["auc_roc"]["mean"])
    print(f"\n  Best nfilter for rLDA LOSO AUC: {best_nf_rdla}")
    print(f"  Best nfilter for SVC  LOSO AUC: {best_nf_svc}")

    lda_best = {int(s): v
                for s, v in loso_results[best_nf_rdla]["rLDA_per_subject"].items()}
    svc_best = {int(s): v
                for s, v in loso_results[best_nf_svc]["SVC_per_subject"].items()}
    plot_per_subject_loso(lda_best, best_nf_rdla, "rLDA", PLOTS_DIR)
    plot_per_subject_loso(svc_best, best_nf_svc,  "SVC",  PLOTS_DIR)

    plot_v1_v4_loso_comparison(
        os.path.join(RESULTS_DIR, "classification_results.json"),
        loso_results, best_nf_rdla, PLOTS_DIR
    )

    # -----------------------------------------------------------------------
    # Threshold-calibrated LOSO (best nfilter only, rLDA)
    # -----------------------------------------------------------------------
    print(f"\n{'#'*70}")
    print(f"# LOSO + Threshold Calibration  (Youden's J on 10% val split)")
    print(f"# nfilter={best_nf_rdla}, rLDA")
    print(f"{'#'*70}")

    cal_loso_sum  = {}
    cal_loso_subj = {}

    for model_name in ["rLDA", "SVC"]:
        print(f"\n  [{model_name}] Calibrated LOSO nf={best_nf_rdla} ...")
        c_sum, c_subj = loso_cv_calibrated(
            X_epochs, y, subject_id, model_name, best_nf_rdla
        )
        cal_loso_sum[model_name]  = c_sum
        cal_loso_subj[model_name] = {str(s): v for s, v in c_subj.items()}
        print(f"  → BAcc={c_sum['balanced_accuracy']['mean']:.3f}±{c_sum['balanced_accuracy']['std']:.3f}  "
              f"AUC={c_sum['auc_roc']['mean']:.3f}±{c_sum['auc_roc']['std']:.3f}  "
              f"PR={c_sum['auc_pr']['mean']:.3f}±{c_sum['auc_pr']['std']:.3f}")

    # -----------------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------------
    results = {
        "within_subject"     : {str(k): ws_results[k]   for k in NFILTER_VALUES},
        "loso"               : {str(k): loso_results[k]  for k in NFILTER_VALUES},
        "loso_calibrated"    : {
            "nfilter"  : best_nf_rdla,
            "rLDA"     : cal_loso_sum.get("rLDA"),
            "SVC"      : cal_loso_sum.get("SVC"),
            "per_subject_rLDA": cal_loso_subj.get("rLDA"),
            "per_subject_SVC" : cal_loso_subj.get("SVC"),
        },
    }
    json_path = os.path.join(RESULTS_DIR, "classification_results_v4.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {json_path}")

    # -----------------------------------------------------------------------
    # Final table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  FINAL SWEEP TABLE  (rLDA)")
    print("=" * 70)
    print(f"  {'nf':>4}  {'TS dim':>8}  "
          f"{'WS BAcc':>12}  {'WS AUC':>10}  {'WS PR':>8}  "
          f"{'LO BAcc':>12}  {'LO AUC':>10}  {'LO PR':>8}")
    print(f"  {'-'*84}")
    for nf in NFILTER_VALUES:
        fd = (2*nf+1); td = fd*(fd+1)//2
        ws = ws_results[nf]["rLDA"]
        lo = loso_results[nf]["rLDA"]
        print(f"  {nf:>4}  {td:>8}  "
              f"  {ws['balanced_accuracy']['mean']:.3f}±{ws['balanced_accuracy']['std']:.3f}  "
              f"  {ws['auc_roc']['mean']:.3f}±{ws['auc_roc']['std']:.3f}  "
              f"  {ws['auc_pr']['mean']:.3f}±{ws['auc_pr']['std']:.3f}  "
              f"  {lo['balanced_accuracy']['mean']:.3f}±{lo['balanced_accuracy']['std']:.3f}  "
              f"  {lo['auc_roc']['mean']:.3f}±{lo['auc_roc']['std']:.3f}  "
              f"  {lo['auc_pr']['mean']:.3f}±{lo['auc_pr']['std']:.3f}")
    print(f"  {'='*84}")

    # Calibrated LOSO summary
    print(f"\n  CALIBRATED LOSO  (nf={best_nf_rdla}, Youden's-J threshold)")
    print(f"  {'-'*60}")
    for mname in ["rLDA", "SVC"]:
        c = cal_loso_sum[mname]
        print(f"  {mname:<6}  BAcc={c['balanced_accuracy']['mean']:.3f}±{c['balanced_accuracy']['std']:.3f}  "
              f"AUC={c['auc_roc']['mean']:.3f}±{c['auc_roc']['std']:.3f}  "
              f"PR={c['auc_pr']['mean']:.3f}±{c['auc_pr']['std']:.3f}  "
              f"F1={c['f1']['mean']:.3f}±{c['f1']['std']:.3f}")
    print(f"  {'='*60}")

    # v1 reference
    v1_json = os.path.join(RESULTS_DIR, "classification_results.json")
    if os.path.exists(v1_json):
        with open(v1_json) as f:
            v1 = json.load(f)
        print(f"\n  v1 reference (rLDA, 120d no xDAWN):")
        print(f"  WS   BAcc={v1['within_subject']['rLDA']['balanced_accuracy']['mean']:.3f}  "
              f"AUC={v1['within_subject']['rLDA']['auc']['mean']:.3f}")
        print(f"  LOSO BAcc={v1['loso']['rLDA']['balanced_accuracy']['mean']:.3f}  "
              f"AUC={v1['loso']['rLDA']['auc']['mean']:.3f}")
        print(f"  PR-AUC random baseline = prevalence = {y.mean():.4f}")

    print("\n  Done!\n")


if __name__ == "__main__":
    main()
