#!/usr/bin/env python3
"""
run_linear_baselines.py — TICA + Sparse PCA on synthetic v2.

Evaluates: MCC, regime classification (F1/AUROC), Cohen's d driver recall.
No GPU needed.

Usage:
    python baseline/evaluation/run_linear_baselines.py
"""

import sys
import numpy as np
from pathlib import Path
from scipy.linalg import eigh
from sklearn.decomposition import SparsePCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from scipy.optimize import linear_sum_assignment

PROJ = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJ / "data/synthetic_well_v2/raw"
SRC_DIR = PROJ / "data/synthetic_well_v2/source"
OUT_BASE = PROJ / "baseline/results"

sys.path.insert(0, str(PROJ / "scripts/data_prep/synthetic"))
from config import Z_DIM, DIM_NAMES, REGIME_DRIVING_FACTORS, REGIME_INVARIANT_FACTORS

LAG_TICA = 1  # time-lag for TICA (in subsampled frames)


# ── Metrics ──────────────────────────────────────────────────────────────

def compute_mcc(z_true, z_learned):
    n_true, n_learned = z_true.shape[1], z_learned.shape[1]
    corr = np.zeros((n_true, n_learned))
    for j in range(n_true):
        for k in range(n_learned):
            corr[j, k] = abs(np.corrcoef(z_true[:, j], z_learned[:, k])[0, 1])
    row_ind, col_ind = linear_sum_assignment(-corr)
    mcc = corr[row_ind, col_ind].mean()
    mapping = dict(zip(row_ind.tolist(), col_ind.tolist()))
    return mcc, mapping, corr


def eval_regime(z_learned, regimes, z_mapping):
    """Regime F1, AUROC, Cohen's d driver recall."""
    clf = LogisticRegression(max_iter=500).fit(z_learned, regimes)
    pred = clf.predict(z_learned)
    prob = clf.predict_proba(z_learned)[:, 1]
    f1 = f1_score(regimes, pred, average='binary')
    auroc = roc_auc_score(regimes, prob)

    # Cohen's d per component
    mask0, mask1 = regimes == 0, regimes == 1
    n_comp = z_learned.shape[1]
    cohens_d = np.zeros(n_comp)
    for k in range(n_comp):
        mu0, mu1 = z_learned[mask0, k].mean(), z_learned[mask1, k].mean()
        std_pool = np.sqrt((z_learned[mask0, k].var() + z_learned[mask1, k].var()) / 2 + 1e-10)
        cohens_d[k] = abs(mu1 - mu0) / std_pool

    top_k = min(4, n_comp)
    top_shift = set(np.argsort(cohens_d)[-top_k:])
    true_drivers_learned = [z_mapping[j] for j in REGIME_DRIVING_FACTORS if j in z_mapping]
    driver_recall = len(set(true_drivers_learned) & top_shift) / len(REGIME_DRIVING_FACTORS)

    return f1, auroc, driver_recall, cohens_d


# ── TICA ─────────────────────────────────────────────────────────────────

def fit_tica(trajs, n_components, lag=1):
    """Manual TICA: time-lagged covariance eigendecomposition."""
    # Collect pairs (x_t, x_{t+lag}) from each trajectory
    pairs_t, pairs_tlag = [], []
    for traj in trajs:
        pairs_t.append(traj[:-lag])
        pairs_tlag.append(traj[lag:])
    X0 = np.concatenate(pairs_t, axis=0)
    X1 = np.concatenate(pairs_tlag, axis=0)

    # Mean-center
    mu = X0.mean(axis=0)
    X0c = X0 - mu
    X1c = X1 - mu

    N = len(X0c)
    C00 = X0c.T @ X0c / N
    C01 = X0c.T @ X1c / N
    C01_sym = (C01 + C01.T) / 2

    # Regularize C00
    C00 += np.eye(C00.shape[0]) * 1e-6

    eigvals, eigvecs = eigh(C01_sym, C00)
    # Sort by descending eigenvalue (slowest modes)
    order = np.argsort(eigvals)[::-1]
    components = eigvecs[:, order[:n_components]]

    return mu, components


def transform_tica(X, mu, components):
    return (X - mu) @ components


# ── Main ─────────────────────────────────────────────────────────────────

def run_method(name, z_learned_all, z_true_all, regimes_all):
    """Compute and print all metrics for one method."""
    out_dir = OUT_BASE / name / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)

    mcc, z_mapping, corr_matrix = compute_mcc(z_true_all, z_learned_all)
    f1, auroc, driver_recall, cohens_d = eval_regime(
        z_learned_all, regimes_all, z_mapping)

    print(f"\n{'='*60}")
    print(f"  {name.upper()}")
    print(f"{'='*60}")
    print(f"  MCC:              {mcc:.4f}")
    print(f"  Regime F1:        {f1:.4f}")
    print(f"  Regime AUROC:     {auroc:.4f}")
    print(f"  Driver Recall@4:  {driver_recall:.2f}")

    print(f"\n  Component mapping:")
    for true_j in sorted(z_mapping):
        role = DIM_NAMES[true_j] if true_j < len(DIM_NAMES) else f"dim_{true_j}"
        driver_tag = " [DRIVING]" if true_j in REGIME_DRIVING_FACTORS else ""
        lk = z_mapping[true_j]
        print(f"    {role}{driver_tag} -> comp_{lk} "
              f"(corr={corr_matrix[true_j, lk]:.3f}, d={cohens_d[lk]:.3f})")

    np.savez(out_dir / "eval_summary.npz",
             mcc=mcc, f1=f1, auroc=auroc,
             driver_recall=driver_recall, cohens_d=cohens_d,
             z_mapping=z_mapping, corr_matrix=corr_matrix)
    print(f"\n  Saved to {out_dir}")
    return mcc, f1, auroc, driver_recall


def main():
    print("=" * 60)
    print("Linear Baselines on Synthetic v2")
    print("=" * 60)

    # ── Load raw trajectories ──
    print("\n[1/4] Loading raw trajectories...")
    raw = np.load(RAW_DIR / "trajectories.npz", allow_pickle=True)
    x_trajs = [raw['x_trajs'][i].astype(np.float32) for i in range(len(raw['x_trajs']))]
    z_trajs = [raw['z_trajs'][i].astype(np.float32) for i in range(len(raw['z_trajs']))]
    state_labels = [raw['state_labels'][i] for i in range(len(raw['state_labels']))]
    n_trajs = len(x_trajs)
    print(f"  {n_trajs} trajectories, obs_dim={x_trajs[0].shape[1]}, z_dim={z_trajs[0].shape[1]}")

    # Concatenate for evaluation (skip first LAG_TICA frames per traj for alignment)
    x_all = np.concatenate([t[LAG_TICA:] for t in x_trajs], axis=0)
    z_all = np.concatenate([t[LAG_TICA:] for t in z_trajs], axis=0)
    regimes_all = np.concatenate([s[LAG_TICA:] for s in state_labels], axis=0)

    # Filter out intermediate frames (regime == -1 or other)
    valid = (regimes_all == 0) | (regimes_all == 1)
    x_eval = x_all[valid]
    z_eval = z_all[valid]
    regimes_eval = regimes_all[valid]
    print(f"  Valid frames (regime 0/1): {len(x_eval)} / {len(x_all)}")

    # ── TICA ──
    print("\n[2/4] Fitting TICA (lag={})...".format(LAG_TICA))
    mu_tica, comps_tica = fit_tica(x_trajs, n_components=Z_DIM, lag=LAG_TICA)
    z_tica = transform_tica(x_eval, mu_tica, comps_tica)
    mcc_tica, f1_tica, auroc_tica, dr_tica = run_method("tica", z_tica, z_eval, regimes_eval)

    # ── Sparse PCA ──
    print("\n[3/4] Fitting Sparse PCA...")
    # Subsample for fitting (SparsePCA is slow on large data)
    rng = np.random.RandomState(42)
    fit_idx = rng.choice(len(x_eval), size=min(50000, len(x_eval)), replace=False)
    for alpha in [1.0, 10.0]:
        tag = f"sparse_pca_a{alpha:.0f}"
        print(f"\n  SparsePCA alpha={alpha}...")
        spca = SparsePCA(n_components=Z_DIM, alpha=alpha, random_state=42, max_iter=200)
        spca.fit(x_eval[fit_idx])
        z_spca = spca.transform(x_eval)
        run_method(tag, z_spca, z_eval, regimes_eval)

    # ── Summary table ──
    print(f"\n{'='*60}")
    print("Done. Results saved to baseline/results/")


if __name__ == "__main__":
    main()
