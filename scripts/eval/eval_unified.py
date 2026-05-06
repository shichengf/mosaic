#!/usr/bin/env python3
"""
Unified evaluation module for all MOSAIC supplementary experiments.

Consolidates metric code that previously lived in multiple scripts:
  - eval_rna_crl_baselines.py
  - eval_corrected_metrics.py
  - eval_b2e_baselines_5seeds.py
  - eval_sparse_regression_baseline_v2.py

Design principles:
  - Single source of truth for MCC, Cohen's d, Z@top3, X_Z@top3, regime_acc,
    module_pct, domain_consistent.
  - All corrected metric choices hard-wired:
      * MATCH_THRESH = 0.5  (Hungarian match must have |r| >= 0.5)
      * X_Z@top3 uses k = |primary ∪ shared|
      * Unmatched top-3 z contributes precision 0 (not dropped from mean)

Typical usage:

    from eval_unified import Evaluator, Dataset

    # For RNA / cross-domain (no z_true; reports module stats, driver, domain consistency)
    ev = Evaluator(Dataset.RNA)
    row = ev.evaluate(
        method='TDRL', seed=0,
        encode_fn=encode_fn,
        influence_matrix=A,
    )
"""
from __future__ import annotations

import enum
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression

# ═══════════════════════════════════════════════════════════════════
# Global config
# ═══════════════════════════════════════════════════════════════════
MATCH_THRESH = 0.5   # |r| threshold for valid Hungarian match
# Repo root: this file lives at <repo>/scripts/eval/eval_unified.py.
PROJ_ROOT = str(Path(__file__).resolve().parent.parent.parent)


# ═══════════════════════════════════════════════════════════════════
# Dataset registry
# ═══════════════════════════════════════════════════════════════════
class Dataset(enum.Enum):
    SYNTHETIC  = 'synthetic'   # near-linear original
    RNA        = 'rna'
    OMNI       = 'omni'
    DISRUPTION = 'disruption'
    CLIMATE    = 'climate'


@dataclass
class DatasetSpec:
    """Per-dataset metadata: where data lives, what the GT is, how to check drivers."""
    name: str
    data_npz: str                                      # path to prepared data.npz
    raw_trajectories: Optional[str] = None             # for MCC on synthetic
    meta_npz: Optional[str] = None                     # for variable_names/_groups
    true_z_dim: Optional[int] = None                   # number of true latent factors (synth only)
    gt_supports: Optional[Dict[int, Dict]] = None      # synth only: {factor_idx → dict}
    drivers: Optional[set] = None                      # indices of true drivers
    rna_modules: Optional[Dict[str, List[int]]] = None
    rna_driver_module: Optional[str] = None
    domain_driver_targets: Optional[List[str]] = None  # tokens expected in top driver's loaded vars
    domain_driver_modules: Optional[List[str]] = None  # or variable_groups the driver should live in
    target_k: Optional[int] = None                     # for RNA X_Z@top_k (k=|Loop|=8)


# ------ Synthetic / original (6 factors, D=30) ------
_SYNTH_GT = {
    0: {'name': 'x',  'primary': {0, 1, 2, 3},         'shared': {4}},
    1: {'name': 'z1', 'primary': {5, 6, 7},            'shared': {8}},
    2: {'name': 'z2', 'primary': {9, 10, 11, 12},      'shared': {4}},
    3: {'name': 'z3', 'primary': {13, 14, 15, 16},     'shared': {8}},
    4: {'name': 'u1', 'primary': {17, 18, 19, 20},     'shared': {21}},
    5: {'name': 'u2', 'primary': {22, 23, 24, 25, 26}, 'shared': {21}},
}
_SYNTH_DRIVERS = {0, 1, 2}

# ------ RNA modules (28 obs = 14 residues × {μ, σ}) ------
_RNA_MODULES = {
    'Loop':        [5, 6, 7, 8, 19, 20, 21, 22],       # U6, U7, C8, G9 (μ and σ)
    'ClosingPair': [4, 9, 18, 23],                     # C5, G10
    'InnerStem':   [2, 3, 10, 11, 16, 17, 24, 25],     # C3, A4, U11, G12
    'OuterStem':   [0, 1, 12, 13, 14, 15, 26, 27],     # G1, G2, C13, C14
}

# ------ Cross-domain expected drivers (per user prompt + variable_groups) ------
_SPECS = {
    Dataset.SYNTHETIC: DatasetSpec(
        name='synthetic',
        data_npz=f'{PROJ_ROOT}/data/synthetic_well_v2/source/data.npz',
        raw_trajectories=f'{PROJ_ROOT}/data/synthetic_well_v2/raw/trajectories.npz',
        true_z_dim=6,
        gt_supports=_SYNTH_GT,
        drivers=_SYNTH_DRIVERS,
    ),
    Dataset.RNA: DatasetSpec(
        name='rna',
        data_npz=f'{PROJ_ROOT}/data/RNA/prepared_frozendist_v4_cpQ/data.npz',
        meta_npz=f'{PROJ_ROOT}/data/RNA/prepared_frozendist_v4_cpQ/metadata.npz',
        rna_modules=_RNA_MODULES,
        rna_driver_module='Loop',
        target_k=8,  # |Loop|
    ),
    Dataset.OMNI: DatasetSpec(
        name='omni',
        data_npz=f'{PROJ_ROOT}/data/omni/prepared/data.npz',
        meta_npz=f'{PROJ_ROOT}/data/omni/prepared/metadata.npz',
        domain_driver_modules=['Coupling'],
    ),
    Dataset.DISRUPTION: DatasetSpec(
        name='disruption',
        data_npz=f'{PROJ_ROOT}/data/disruption/prepared/data.npz',
        meta_npz=f'{PROJ_ROOT}/data/disruption/prepared/metadata.npz',
        domain_driver_targets=['q95'],         # q95 is the physically expected driver
        domain_driver_modules=['MHD', 'Shape'],
    ),
    Dataset.CLIMATE: DatasetSpec(
        name='climate',
        data_npz=f'{PROJ_ROOT}/data/climate/prepared_screened/data.npz',
        meta_npz=f'{PROJ_ROOT}/data/climate/prepared_screened/metadata.npz',
        domain_driver_modules=['Nino3.4', 'Nino3', 'Nino4', 'Nino1+2'],
    ),
}


# ═══════════════════════════════════════════════════════════════════
# Low-level metrics
# ═══════════════════════════════════════════════════════════════════
def cohens_d_per_z(z: np.ndarray, ct: np.ndarray) -> np.ndarray:
    """Per-dimension Cohen's d for binary regime labels ct ∈ {0, 1}."""
    m0, m1 = ct == 0, ct == 1
    return np.array([
        abs(z[m0, k].mean() - z[m1, k].mean()) /
        np.sqrt((z[m0, k].var(ddof=1) + z[m1, k].var(ddof=1)) / 2 + 1e-10)
        for k in range(z.shape[1])
    ])


def regime_accuracy(z: np.ndarray, ct: np.ndarray, seed: int = 0) -> float:
    """Logistic-regression probe accuracy on the given (z, regime) split."""
    clf = LogisticRegression(max_iter=500, random_state=seed)
    clf.fit(z, ct)
    return float(clf.score(z, ct))


def hungarian_match(z_true: np.ndarray, z_learned: np.ndarray) -> Tuple[float, np.ndarray, Dict[int, int]]:
    """Pearson-|r| Hungarian matching with threshold.

    Returns:
        mcc: mean |r| over all assignments (before thresholding)
        corr: true_dim × learned_dim matrix of |r|
        inv_map: {learned_k: true_j}  only for pairs with |r| >= MATCH_THRESH
    """
    T, L = z_true.shape[1], z_learned.shape[1]
    corr = np.zeros((T, L))
    for j in range(T):
        for k in range(L):
            r = pearsonr(z_true[:, j], z_learned[:, k])[0]
            corr[j, k] = abs(r) if np.isfinite(r) else 0.0
    ri, ci = linear_sum_assignment(-corr)
    mcc = float(corr[ri, ci].mean())
    inv_map = {int(c): int(r) for r, c in zip(ri, ci) if corr[r, c] >= MATCH_THRESH}
    return mcc, corr, inv_map


def xz_top3(top3_z: List[int],
            inv_map: Dict[int, int],
            influence: np.ndarray,
            gt: Dict[int, Dict],
            drivers: set) -> Tuple[float, List[Tuple]]:
    """Corrected X_Z@top3.

    For each learned top-3 z:
      - if Hungarian match is a driver: precision = |top_k ∩ full_support| / k
        where k = |primary ∪ shared| (full support size)
      - else: precision = 0

    Returns mean precision and per-z breakdown.
    """
    precisions, detail = [], []
    for zi in top3_z:
        mi = inv_map.get(zi)
        if mi is None or mi not in drivers:
            precisions.append(0.0)
            detail.append(('unmatched/invariant', zi, 0, 0))
            continue
        full = gt[mi]['primary'] | gt[mi]['shared']
        k = len(full)
        top_k = set(np.argsort(influence[:, zi])[-k:])
        hits = len(top_k & full)
        precisions.append(hits / k)
        detail.append((gt[mi]['name'], zi, hits, k))
    return float(np.mean(precisions)), detail


def module_percentages(inf_driver: np.ndarray,
                       modules: Dict[str, List[int]]) -> Tuple[Dict[str, float], int, str]:
    """Aggregate influence over known module memberships.

    Returns (pct_dict, driver_rank, driver_module_name)
    """
    mod_inf = {m: float(inf_driver[obs].sum()) for m, obs in modules.items()}
    total = sum(mod_inf.values()) + 1e-10
    pct = {k: round(v / total * 100, 2) for k, v in mod_inf.items()}
    best = max(pct, key=pct.get)
    return pct, best, sorted(pct.keys(), key=lambda m: -pct[m])


# ═══════════════════════════════════════════════════════════════════
# Evaluator class
# ═══════════════════════════════════════════════════════════════════
@dataclass
class EvaluationRow:
    dataset: str
    method: str
    seed: int
    regime_acc: float = float('nan')
    driver_z: int = -1
    driver_d: float = float('nan')
    top3_z: List[int] = field(default_factory=list)
    d_vals: List[float] = field(default_factory=list)
    # Synthetic-only
    mcc: float = float('nan')
    matched_factors: List[Optional[str]] = field(default_factory=list)
    z_top3: str = ''
    x_z_top3: float = float('nan')
    xz_detail: List[Tuple] = field(default_factory=list)
    # RNA / cross-domain module-level
    module_pct: Dict[str, float] = field(default_factory=dict)
    driver_module: str = ''
    driver_rank_by_target: int = -1
    x_z_top_k: float = float('nan')
    x_z_top_4: float = float('nan')
    # Cross-domain driver variables
    top_driver_vars: List[str] = field(default_factory=list)
    domain_consistent: Optional[bool] = None
    # Debug
    notes: str = ''

    def to_flat(self) -> Dict:
        """Flatten for CSV row. Nested dicts become JSON-y strings."""
        import json
        out = {
            'dataset': self.dataset, 'method': self.method, 'seed': self.seed,
            'regime_acc': round(self.regime_acc, 4) if np.isfinite(self.regime_acc) else '',
            'driver_z': self.driver_z,
            'driver_d': round(self.driver_d, 3) if np.isfinite(self.driver_d) else '',
            'top3_z': ','.join(map(str, self.top3_z)),
            'd_vals': ','.join(f'{d:.3f}' for d in self.d_vals),
            'mcc': round(self.mcc, 4) if np.isfinite(self.mcc) else '',
            'matched_factors': ','.join(str(m) for m in self.matched_factors),
            'z_top3': self.z_top3,
            'x_z_top3': round(self.x_z_top3, 4) if np.isfinite(self.x_z_top3) else '',
            'module_pct': json.dumps(self.module_pct) if self.module_pct else '',
            'driver_module': self.driver_module,
            'driver_rank_by_target': self.driver_rank_by_target,
            'x_z_top_k': round(self.x_z_top_k, 4) if np.isfinite(self.x_z_top_k) else '',
            'x_z_top_4': round(self.x_z_top_4, 4) if np.isfinite(self.x_z_top_4) else '',
            'top_driver_vars': ','.join(self.top_driver_vars),
            'domain_consistent': '' if self.domain_consistent is None else
                                 ('YES' if self.domain_consistent else 'NO'),
            'notes': self.notes,
        }
        return out


class Evaluator:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.spec = _SPECS[dataset]
        self._data = None
        self._meta = None

    # ---------- data access (lazy) ----------
    @property
    def data(self):
        if self._data is None:
            self._data = np.load(self.spec.data_npz, allow_pickle=True)
        return self._data

    @property
    def meta(self):
        if self._meta is None and self.spec.meta_npz:
            self._meta = np.load(self.spec.meta_npz, allow_pickle=True)
        return self._meta

    @property
    def X_last(self) -> np.ndarray:
        return self.data['xt'][:, -1, :].astype(np.float32)

    @property
    def ct(self) -> np.ndarray:
        return self.data['ct'].squeeze().astype(int)

    # ---------- main evaluate ----------
    def evaluate(self,
                 method: str,
                 seed: int,
                 z_full: np.ndarray,
                 influence: np.ndarray,
                 z_true: Optional[np.ndarray] = None,
                 z_learned_for_mcc: Optional[np.ndarray] = None,
                 notes: str = '') -> EvaluationRow:
        """
        Args:
            method: method name for the row (e.g., 'MOSAIC', 'TDRL')
            seed: seed id
            z_full: learned z on the full eval dataset (N, z_dim)
            influence: non-negative (D, z_dim) matrix: column j = influence of z_j on obs
            z_true: synthetic only — (N_subset, true_z_dim) for MCC + Hungarian matching
            z_learned_for_mcc: synthetic only — encoded z on the same N_subset as z_true
                               (If None, use z_full[:len(z_true)].)
            notes: free-text appended to row.notes
        """
        row = EvaluationRow(dataset=self.spec.name, method=method, seed=seed, notes=notes)

        ct = self.ct
        # 1) Regime accuracy (probe LR)
        row.regime_acc = regime_accuracy(z_full, ct, seed=seed)

        # 2) Cohen's d → driver + top-3
        d_vals = cohens_d_per_z(z_full, ct)
        row.d_vals = d_vals.round(3).tolist()
        row.driver_z = int(np.argmax(d_vals))
        row.driver_d = float(d_vals[row.driver_z])
        row.top3_z = np.argsort(d_vals)[-3:][::-1].tolist()

        # 3) Synthetic-only: MCC + Z@top3 + X_Z@top3
        if self.spec.true_z_dim is not None and z_true is not None:
            z_lm = z_learned_for_mcc if z_learned_for_mcc is not None else z_full[:len(z_true)]
            mcc, corr, inv_map = hungarian_match(z_true, z_lm)
            row.mcc = mcc
            row.matched_factors = [
                self.spec.gt_supports[inv_map[z]]['name']
                if z in inv_map else None
                for z in row.top3_z
            ]
            n_drivers = sum(1 for mi in [inv_map.get(z) for z in row.top3_z]
                            if mi is not None and mi in self.spec.drivers)
            row.z_top3 = f'{n_drivers}/3'
            xz, detail = xz_top3(row.top3_z, inv_map, influence,
                                 self.spec.gt_supports, self.spec.drivers)
            row.x_z_top3 = xz
            row.xz_detail = detail

        # 4) RNA / cross-domain: module-level driver analysis
        if self.spec.rna_modules is not None:
            inf_driver = influence[:, row.driver_z]
            pct, best, rank_order = module_percentages(inf_driver, self.spec.rna_modules)
            row.module_pct = pct
            row.driver_module = best
            # rank of target module (Loop)
            tgt = self.spec.rna_driver_module
            row.driver_rank_by_target = rank_order.index(tgt) + 1
            # X_Z@top_k for target module (k = |target|)
            target_obs = set(self.spec.rna_modules[tgt])
            k = self.spec.target_k
            top_k = set(np.argsort(inf_driver)[-k:])
            row.x_z_top_k = len(top_k & target_obs) / k
            top_4 = set(np.argsort(inf_driver)[-4:])
            row.x_z_top_4 = len(top_4 & target_obs) / 4

        # 5) Cross-domain: top driver's loaded variable names + module consistency
        if (self.spec.domain_driver_targets is not None
            or self.spec.domain_driver_modules is not None) and self.meta is not None:
            inf_driver = influence[:, row.driver_z]
            var_names = self.meta['variable_names']
            var_groups = self.meta['variable_groups']
            # top-3 loaded variables
            top3_vars_idx = np.argsort(inf_driver)[-3:][::-1]
            row.top_driver_vars = [str(var_names[i]) for i in top3_vars_idx]
            top3_groups = [str(var_groups[i]) for i in top3_vars_idx]
            # domain consistency: any of top-3 match target vars OR be in target modules
            hit = False
            if self.spec.domain_driver_targets:
                for v in row.top_driver_vars:
                    for tgt in self.spec.domain_driver_targets:
                        if tgt.lower() in v.lower():
                            hit = True; break
            if (not hit) and self.spec.domain_driver_modules:
                for g in top3_groups:
                    if g in self.spec.domain_driver_modules:
                        hit = True; break
            row.domain_consistent = hit
            # also stash module of the driver's most-loaded variable
            row.driver_module = top3_groups[0]

        return row


# ═══════════════════════════════════════════════════════════════════
# Convenience: influence via per-z probe (additive decoder)
# ═══════════════════════════════════════════════════════════════════
def additive_influence_probe(decoder: Callable[[torch.Tensor], torch.Tensor],
                              z_dim: int,
                              D: Optional[int] = None) -> np.ndarray:
    """|f(z_j=+1) - f(z_j=-1)| for each z (others zero). Returns (D, z_dim)."""
    cols = []
    with torch.no_grad():
        for j in range(z_dim):
            zp = torch.zeros(1, z_dim); zp[0, j] = 1.0
            zn = torch.zeros(1, z_dim); zn[0, j] = -1.0
            out = (decoder(zp) - decoder(zn)).abs().squeeze()
            cols.append(out.numpy() if torch.is_tensor(out) else np.asarray(out))
    return np.stack(cols, axis=1)


def encoder_jacobian_influence(encoder_fn: Callable[[torch.Tensor], torch.Tensor],
                                x_sample: torch.Tensor,
                                z_dim: int) -> np.ndarray:
    """Encoder-Jacobian influence (D, z_dim) averaged over a sample batch.

    Useful for methods like SPIB where the decoder doesn't predict raw obs.
    Interpretation: |∂ z_j / ∂ x_i|  ≈ influence of x_i on z_j, transposed.
    """
    x_sample = x_sample.clone().detach().requires_grad_(True)
    z_out = encoder_fn(x_sample)                                 # (B, z_dim)
    D = x_sample.shape[-1]
    jac_cols = np.zeros((D, z_dim), dtype=np.float64)
    for j in range(z_dim):
        grad = torch.autograd.grad(z_out[:, j].sum(), x_sample, retain_graph=True)[0]
        jac_cols[:, j] = grad.abs().mean(dim=0).detach().cpu().numpy()
    return jac_cols


def lasso_influence(z_full: np.ndarray, x_obs: np.ndarray,
                    alpha: float = 0.01, max_iter: int = 5000) -> np.ndarray:
    """Per-obs Lasso(z → x_i), absolute coefficients as influence. (D, z_dim)."""
    from sklearn.linear_model import Lasso
    D = x_obs.shape[1]; z_dim = z_full.shape[1]
    W = np.zeros((D, z_dim))
    z_std = (z_full - z_full.mean(0)) / (z_full.std(0) + 1e-8)
    for i in range(D):
        xi = x_obs[:, i]
        xi_std = (xi - xi.mean()) / (xi.std() + 1e-8)
        lr = Lasso(alpha=alpha, max_iter=max_iter, tol=1e-3)
        lr.fit(z_std, xi_std)
        W[i] = np.abs(lr.coef_)
    return W


# ═══════════════════════════════════════════════════════════════════
# Encode utility for MOSAIC TimeVaryingProcess_v2 (our main model)
# ═══════════════════════════════════════════════════════════════════
def encode_mosaic(model, X: np.ndarray, batch: int = 4096) -> np.ndarray:
    """Run MOSAIC encoder over windowed obs, return posterior mean."""
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            _, mu, _, _ = model.net(torch.tensor(X[i:i + batch]))
            out.append(mu.numpy())
    return np.concatenate(out)


__all__ = [
    'Dataset', 'DatasetSpec', 'Evaluator', 'EvaluationRow',
    'cohens_d_per_z', 'regime_accuracy', 'hungarian_match', 'xz_top3',
    'module_percentages', 'additive_influence_probe', 'encoder_jacobian_influence',
    'lasso_influence', 'encode_mosaic', 'MATCH_THRESH',
]


# ═══════════════════════════════════════════════════════════════════
# CLI: load a phase-2 checkpoint, encode data, report metrics.
# Skips MCC / Z@top3 / X_Z@top3 because the synthetic prepare step
# doesn't persist the per-frame z_true alignment needed for them.
# ═══════════════════════════════════════════════════════════════════
def _cli():
    import argparse, glob, json, dataclasses

    ap = argparse.ArgumentParser(description="Evaluate a MOSAIC phase-2 checkpoint.")
    ap.add_argument("--benchmark", required=True, choices=[d.value for d in Dataset])
    ap.add_argument("--ckpt_glob", required=True,
                    help="Glob for phase-2 best-*.ckpt files.")
    ap.add_argument("--data_dir", required=True,
                    help="Directory containing data.npz (overrides spec path).")
    ap.add_argument("--out", required=True,
                    help="Output directory for summary.json.")
    args = ap.parse_args()

    ckpts = sorted(glob.glob(args.ckpt_glob))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints matched: {args.ckpt_glob}")
    ckpt = ckpts[0]
    print(f"[eval] ckpt: {ckpt}")

    data_dir = Path(args.data_dir).resolve()
    data_npz = data_dir / "data.npz"
    if not data_npz.exists():
        raise FileNotFoundError(f"data.npz not found at {data_npz}")
    print(f"[eval] data: {data_npz}")

    # influence_A.npz is saved by phase 2 in the log_dir (4 levels up from the
    # checkpoint file: best.ckpt → checkpoints/ → version_X/ → lightning_logs/ → log_dir).
    log_dir = Path(ckpt).resolve().parent.parent.parent.parent
    inf_path = log_dir / "influence_A.npz"
    if not inf_path.exists():
        raise FileNotFoundError(f"influence_A.npz not found at {inf_path}")
    A = np.load(inf_path)["A"].astype(np.float32)  # (D, z_dim)
    print(f"[eval] influence A: {inf_path} shape={A.shape}")

    # Load checkpoint and build encoder fn.
    from mosaic.models.mosaic import TimeVaryingProcess_v2
    model = TimeVaryingProcess_v2.load_from_checkpoint(ckpt, map_location="cpu")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    data = np.load(data_npz, allow_pickle=True)
    X = data["xt"].astype(np.float32)         # (N, seq_len, D)
    X_last = X[:, -1, :]                       # one frame per example (matches Evaluator.X_last)
    ct = data["ct"].squeeze().astype(int)
    print(f"[eval] xt={X.shape} ct={ct.shape}")

    # Encoder expects (B, input_dim); flatten last-frame view and run in batches.
    z_chunks = []
    with torch.no_grad():
        for i in range(0, len(X_last), 4096):
            xb = torch.from_numpy(X_last[i:i + 4096]).to(device)
            _, mu, _, _ = model.net(xb)
            z_chunks.append(mu.cpu().numpy())
    z_full = np.concatenate(z_chunks)  # (N, z_dim)
    print(f"[eval] z_full={z_full.shape}")

    # Patch the spec's data_npz so Evaluator.ct uses our --data_dir copy.
    ev = Evaluator(Dataset(args.benchmark))
    ev.spec.data_npz = str(data_npz)
    if ev.spec.meta_npz is None and (data_dir / "metadata.npz").exists():
        ev.spec.meta_npz = str(data_dir / "metadata.npz")

    row = ev.evaluate(method="MOSAIC", seed=0,
                      z_full=z_full, influence=A,
                      z_true=None,  # MCC needs frame-level z alignment (not saved)
                      notes="cli_smoke")

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(dataclasses.asdict(row), indent=2, default=str))

    print("\n=== Evaluation summary ===")
    print(f"  benchmark    : {row.dataset}")
    print(f"  regime_acc   : {row.regime_acc:.4f}")
    print(f"  driver_z     : {row.driver_z}  (cohen_d={row.driver_d:.3f})")
    print(f"  top3_z       : {row.top3_z}")
    print(f"  d_vals       : {row.d_vals}")
    if row.mcc is not None:
        print(f"  MCC          : {row.mcc:.4f}")
        print(f"  Z@top3       : {row.z_top3}")
        print(f"  X_Z@top3     : {row.x_z_top3}")
    else:
        print("  MCC / Z@top3 / X_Z@top3 : skipped (z_true alignment unavailable)")
    print(f"\n[eval] wrote {summary_path}")


if __name__ == "__main__":
    _cli()
