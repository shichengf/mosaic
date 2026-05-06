#!/usr/bin/env python3
"""
run_linear_baselines_general.py — TICA + Sparse PCA for any prepared dataset.

Usage:
    python baseline/evaluation/run_linear_baselines_general.py --dataset rna
"""

import argparse
import numpy as np
from pathlib import Path
from scipy.linalg import eigh
from sklearn.decomposition import SparsePCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

PROJ = Path(__file__).resolve().parent.parent.parent
OUT_BASE = PROJ / "baseline/results"


def compute_module_purity(loading_matrix, var_groups, n_components):
    modules = sorted(set(var_groups))
    mod_indices = {m: [i for i, g in enumerate(var_groups) if g == m] for m in modules}
    abs_load = np.abs(loading_matrix)
    results = []
    for k in range(n_components):
        col = abs_load[:, k]
        total = col.sum() + 1e-10
        best_mod = max(modules, key=lambda m: col[mod_indices[m]].sum())
        best_score = col[mod_indices[best_mod]].sum() / total
        results.append((best_mod, best_score))
    return np.mean([r[1] for r in results]), results


def eval_regime(z, regimes):
    clf = LogisticRegression(max_iter=500).fit(z, regimes)
    pred = clf.predict(z)
    prob = clf.predict_proba(z)[:, 1]
    f1 = f1_score(regimes, pred, average='binary')
    auroc = roc_auc_score(regimes, prob)
    return f1, auroc


def run_method(name, ds, z_learned, regimes, loading, var_groups, n_comp):
    out_dir = OUT_BASE / name / ds
    out_dir.mkdir(parents=True, exist_ok=True)

    f1, auroc = eval_regime(z_learned, regimes)
    purity, comp_mods = compute_module_purity(loading, var_groups, n_comp)

    mask0, mask1 = regimes == 0, regimes == 1
    cohens_d = np.zeros(n_comp)
    for k in range(n_comp):
        mu0, mu1 = z_learned[mask0, k].mean(), z_learned[mask1, k].mean()
        sp = np.sqrt((z_learned[mask0, k].var() + z_learned[mask1, k].var()) / 2 + 1e-10)
        cohens_d[k] = abs(mu1 - mu0) / sp

    print(f"\n{'='*60}")
    print(f"  {name.upper()} on {ds}")
    print(f"{'='*60}")
    print(f"  Regime F1:      {f1:.4f}")
    print(f"  Regime AUROC:   {auroc:.4f}")
    print(f"  Module Purity:  {purity:.4f}")
    for k, (mod, sc) in enumerate(comp_mods):
        print(f"    comp_{k} -> {mod:20s} (purity={sc:.3f}, d={cohens_d[k]:.3f})")

    np.savez(out_dir / "eval_summary.npz",
             f1=f1, auroc=auroc, purity=purity,
             cohens_d=cohens_d, comp_modules=comp_mods)
    print(f"  Saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_dir", default=None,
                        help="Override data directory (default: data/{dataset}/prepared)")
    args = parser.parse_args()
    ds = args.dataset

    data_dir = Path(args.data_dir) if args.data_dir else PROJ / f"data/{ds}/prepared"
    data = np.load(data_dir / "data.npz")
    meta = np.load(data_dir / "metadata.npz", allow_pickle=True)

    xt = data['xt'].astype(np.float32)
    ct = data['ct'].squeeze().astype(np.int64)
    var_groups = list(meta['variable_groups'])
    n_modules = len(set(var_groups))
    N, W, D = xt.shape

    print(f"Dataset: {ds}, N={N}, D={D}, modules={n_modules}")

    x_curr = xt[:, -1, :]
    x_lag = xt[:, 0, :]

    # TICA
    mu = x_lag.mean(0)
    X0c, X1c = x_lag - mu, x_curr - mu
    C00 = X0c.T @ X0c / N + np.eye(D) * 1e-6
    C01s = (X0c.T @ X1c / N + X1c.T @ X0c / N) / 2
    eigvals, eigvecs = eigh(C01s, C00)
    n_comp = min(n_modules, D)
    tica_comps = eigvecs[:, np.argsort(eigvals)[::-1][:n_comp]]
    z_tica = (x_curr - mu) @ tica_comps
    run_method("tica", ds, z_tica, ct, tica_comps, var_groups, n_comp)

    # Sparse PCA
    rng = np.random.RandomState(42)
    fit_n = min(50000, N)
    fit_idx = rng.choice(N, fit_n, replace=False) if N > fit_n else np.arange(N)
    for alpha in [1.0, 10.0]:
        tag = f"sparse_pca_a{alpha:.0f}"
        spca = SparsePCA(n_components=n_comp, alpha=alpha,
                         random_state=42, max_iter=200)
        spca.fit(x_curr[fit_idx])
        z_spca = spca.transform(x_curr)
        run_method(tag, ds, z_spca, ct, spca.components_.T, var_groups, n_comp)

    print(f"\nDone: {ds}")


if __name__ == "__main__":
    main()
