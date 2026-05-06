#!/usr/bin/env python3
"""
run_linear_baselines_rna.py — TICA + Sparse PCA on RNA hairpin (cpQ).

No ground truth latents → evaluate regime classification (F1/AUROC)
and module purity (how well components align with known structural modules).
"""

import numpy as np
from pathlib import Path
from scipy.linalg import eigh
from sklearn.decomposition import SparsePCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

PROJ = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJ / "data/RNA/prepared_frozendist_v4_cpQ/data.npz"
META_PATH = PROJ / "data/RNA/prepared_frozendist_v4_cpQ/metadata.npz"
OUT_BASE = PROJ / "baseline/results"

N_COMPONENTS = 4  # 4 structural modules
LAG_TICA = 1

# Module ground truth: variable index -> module name
# 14 residues x 2 features (dev_dist, std_dist) = 28 variables
# dev_dist: indices 0-13, std_dist: indices 14-27
MODULE_MAP = {
    'OuterStem':    [0, 1, 12, 13, 14, 15, 26, 27],   # G1,G2,C13,C14
    'InnerStem':    [2, 3, 10, 11, 16, 17, 24, 25],    # C3,A4,U11,G12
    'ClosingPair':  [4, 9, 18, 23],                      # C5,G10
    'Loop':         [5, 6, 7, 8, 19, 20, 21, 22],       # U6,U7,C8,G9
}


def compute_module_purity(loading_matrix, module_map, n_components):
    """For each component, find which module has highest total loading."""
    abs_load = np.abs(loading_matrix)  # (n_vars, n_components)
    results = []
    for k in range(n_components):
        col = abs_load[:, k]
        best_module, best_score = None, -1
        total = col.sum() + 1e-10
        for mod_name, indices in module_map.items():
            score = col[indices].sum() / total
            if score > best_score:
                best_score = score
                best_module = mod_name
        results.append((best_module, best_score))
    purity = np.mean([r[1] for r in results])
    return purity, results


def eval_regime(z_learned, regimes):
    clf = LogisticRegression(max_iter=500).fit(z_learned, regimes)
    pred = clf.predict(z_learned)
    prob = clf.predict_proba(z_learned)[:, 1]
    f1 = f1_score(regimes, pred, average='binary')
    auroc = roc_auc_score(regimes, prob)
    return f1, auroc


def run_method(name, z_learned, regimes, loading_matrix):
    out_dir = OUT_BASE / name / "rna"
    out_dir.mkdir(parents=True, exist_ok=True)

    f1, auroc = eval_regime(z_learned, regimes)
    purity, comp_modules = compute_module_purity(
        loading_matrix, MODULE_MAP, N_COMPONENTS)

    # Cohen's d
    mask0, mask1 = regimes == 0, regimes == 1
    cohens_d = np.zeros(z_learned.shape[1])
    for k in range(z_learned.shape[1]):
        mu0, mu1 = z_learned[mask0, k].mean(), z_learned[mask1, k].mean()
        std_pool = np.sqrt((z_learned[mask0, k].var() + z_learned[mask1, k].var()) / 2 + 1e-10)
        cohens_d[k] = abs(mu1 - mu0) / std_pool

    print(f"\n{'='*60}")
    print(f"  {name.upper()} on RNA")
    print(f"{'='*60}")
    print(f"  Regime F1:        {f1:.4f}")
    print(f"  Regime AUROC:     {auroc:.4f}")
    print(f"  Module Purity:    {purity:.4f}")
    print(f"\n  Component -> Module mapping:")
    for k, (mod, score) in enumerate(comp_modules):
        print(f"    comp_{k} -> {mod:15s} (purity={score:.3f}, d={cohens_d[k]:.3f})")

    np.savez(out_dir / "eval_summary.npz",
             f1=f1, auroc=auroc, purity=purity,
             cohens_d=cohens_d, comp_modules=comp_modules)
    print(f"\n  Saved to {out_dir}")
    return f1, auroc, purity


def main():
    print("=" * 60)
    print("Linear Baselines on RNA (cpQ)")
    print("=" * 60)

    # Load windowed data
    print("\n[1/4] Loading data...")
    data = np.load(DATA_PATH)
    xt = data['xt'].astype(np.float32)  # (N, 3, 28)
    ct = data['ct'].squeeze().astype(np.int64)
    meta = np.load(META_PATH, allow_pickle=True)
    var_names = meta['variable_names']
    print(f"  {len(xt)} samples, {xt.shape[-1]} features, "
          f"regime 0: {(ct==0).sum()}, regime 1: {(ct==1).sum()}")

    # Use current frame for PCA/classification, all frames for TICA
    x_current = xt[:, -1, :]  # (N, 28)
    x_lag0 = xt[:, -1, :]
    x_lag1 = xt[:, 0, :]  # lag=2 back

    # ── TICA ──
    print("\n[2/4] Fitting TICA (lag={})...".format(LAG_TICA))
    # Use windowed pairs: x_lag1 (t-2) and x_lag0 (t)
    mu = x_lag1.mean(axis=0)
    X0c = x_lag1 - mu
    X1c = x_lag0 - mu
    N = len(X0c)
    C00 = X0c.T @ X0c / N + np.eye(28) * 1e-6
    C01_sym = (X0c.T @ X1c / N + X1c.T @ X0c / N) / 2
    eigvals, eigvecs = eigh(C01_sym, C00)
    tica_comps = eigvecs[:, np.argsort(eigvals)[::-1][:N_COMPONENTS]]
    z_tica = (x_current - mu) @ tica_comps
    run_method("tica", z_tica, ct, tica_comps)

    # ── Sparse PCA ──
    print("\n[3/4] Fitting Sparse PCA...")
    rng = np.random.RandomState(42)
    fit_idx = rng.choice(len(x_current), size=min(50000, len(x_current)), replace=False)
    for alpha in [1.0, 10.0]:
        tag = f"sparse_pca_a{alpha:.0f}"
        print(f"\n  SparsePCA alpha={alpha}...")
        spca = SparsePCA(n_components=N_COMPONENTS, alpha=alpha,
                         random_state=42, max_iter=200)
        spca.fit(x_current[fit_idx])
        z_spca = spca.transform(x_current)
        loading = spca.components_.T  # (28, n_components)
        run_method(tag, z_spca, ct, loading)

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
