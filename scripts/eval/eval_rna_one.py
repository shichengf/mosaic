#!/usr/bin/env python3
"""
Evaluate a single (method, seed) CRL baseline checkpoint on RNA.
Appends one row to experiments/rna_baseline_5seeds.csv.

Called from a SLURM submission script after training completes.
"""
import os, sys, glob, argparse, csv, warnings
from pathlib import Path
import numpy as np
import torch
warnings.filterwarnings('ignore')

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJ = str(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT / "scripts/eval"))
sys.path.insert(0, PROJ)
sys.path.insert(0, str(REPO_ROOT / "baselines/tdrl_lily"))
sys.path.insert(0, str(REPO_ROOT / "baselines/ctrlns"))
import pytorch_lightning as _pl
sys.modules['lightning'] = type(sys)('lightning')
sys.modules['lightning.pytorch'] = _pl

from eval_unified import Evaluator, Dataset, encoder_jacobian_influence

# Import the loader functions from the task1_2 script (don't duplicate)
from run_task1_2_crossdomain_eval import (
    load_tdrl, load_bvae, load_ivae, load_svae, load_ctrlns, _encode_from_net,
)

OUT_CSV = str(REPO_ROOT / 'experiments/rna_baseline_5seeds.csv')
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

CSV_FIELDS = [
    'dataset', 'method', 'seed', 'regime_acc', 'driver_z', 'driver_d', 'top3_z',
    'd_vals', 'mcc', 'matched_factors', 'z_top3', 'x_z_top3', 'module_pct',
    'driver_module', 'driver_rank_by_target', 'x_z_top_k', 'x_z_top_4',
    'top_driver_vars', 'domain_consistent', 'notes',
]


def find_rna_ckpt(method, seed):
    """Find the checkpoint for a given seed-tagged RNA run.

    Strictly refuses to fall back to a different-seed checkpoint.
    """
    if seed == 0:
        # Legacy layouts: lightning may live directly under *_rna or under */rna_<method>/.
        patterns = [
            f'baseline/results/{method}_rna/*/rna_{method}/lightning_logs/version_*/checkpoints/*.ckpt',
            f'baseline/results/{method}_rna/lightning_logs/version_*/checkpoints/*.ckpt',
        ]
    else:
        # Only seed-specific directories — no fallback to another seed's ckpt.
        patterns = [
            f'baseline/results/{method}_rna_seed{seed}/lightning_logs/version_*/checkpoints/*.ckpt',
            f'baseline/results/{method}_rna_seed{seed}/*/rna_{method}/lightning_logs/version_*/checkpoints/*.ckpt',
        ]
    for p in patterns:
        ckpts = sorted(REPO_ROOT.glob(p))
        if ckpts:
            return ckpts[-1]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True, choices=['tdrl','bvae','ivae','svae','ctrlns'])
    ap.add_argument('--seed', type=int, required=True)
    args = ap.parse_args()

    ckpt = find_rna_ckpt(args.method, args.seed)
    if ckpt is None:
        print(f'[skip] no ckpt for {args.method} seed={args.seed}')
        return 1
    print(f'[eval] {args.method} seed={args.seed}  ckpt={ckpt}')

    os.chdir(PROJ)
    ev = Evaluator(Dataset.RNA)
    D, z_dim = 28, 4

    if args.method == 'tdrl':
        m = load_tdrl(ckpt, D, z_dim)
    elif args.method == 'bvae':
        m = load_bvae(ckpt, D, z_dim, beta=8.0)
    elif args.method == 'ivae':
        m = load_ivae(ckpt, D, z_dim)
    elif args.method == 'svae':
        m = load_svae(ckpt, D, z_dim)
    elif args.method == 'ctrlns':
        m = load_ctrlns(ckpt, D, z_dim)

    z_full = _encode_from_net(m, ev.X_last)
    enc = lambda x: m.net(x)[1] if hasattr(m, 'net') else m(x)[1]
    inf = encoder_jacobian_influence(enc, torch.tensor(ev.X_last[:1024], dtype=torch.float32), z_dim)
    row = ev.evaluate(method=args.method.upper(), seed=args.seed,
                      z_full=z_full, influence=inf)

    # Append
    exists = os.path.exists(OUT_CSV)
    with open(OUT_CSV, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row.to_flat())
    print(f'[done] {args.method} seed={args.seed}: acc={row.regime_acc:.3f} '
          f'driver={row.driver_z} d={row.driver_d:.2f} '
          f'Loop%={row.module_pct.get("Loop", 0):.1f} rank=#{row.driver_rank_by_target}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
