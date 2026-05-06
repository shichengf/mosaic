#!/usr/bin/env python3
"""run_crossseed.py — Run cross-seed MOSAIC experiments.

Wraps the two-stage pipeline (``train_phase1.py`` then ``train_phase2.py``)
for multiple seeds on a chosen benchmark, then runs ``eval_unified.py`` on
each phase-2 checkpoint.

Usage::

    # Synthetic 5-seed run on a single GPU
    python scripts/train/run_crossseed.py --benchmark synthetic --seeds 0 42 123 456 789

    # RNA 5-seed run, GPU 1
    python scripts/train/run_crossseed.py --benchmark rna --seeds 0 42 123 456 789 --gpu 1

    # Just print the commands without running them (useful for SLURM arrays)
    python scripts/train/run_crossseed.py --benchmark synthetic --seeds 0 42 --dry_run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
PHASE1 = REPO / "scripts" / "train" / "train_phase1.py"
PHASE2 = REPO / "scripts" / "train" / "train_phase2.py"
EVAL = REPO / "scripts" / "eval" / "eval_unified.py"


# ----------------------------------------------------------------------
# Per-benchmark hyperparameters used in the paper.
# ----------------------------------------------------------------------
# Paper-canonical configs.
#  * "two_phase": Phase 1 (dense decoder) → Phase 2 (additive + sparsity).
#    Canonical synthetic & cross-domain runs use this.
#  * "single_phase": one shot with additive MLP from epoch 0 + slot gate.
#    RNA in the paper uses this mode.
BENCHMARK_CFG = {
    "synthetic": dict(
        mode="two_phase",
        data_dir="data/synthetic_well_v2/source",
        z_dim=8, hidden_dim=128,
        hpz_p1=32, hpz_p2=4,
        lambda_sparse=50,
        beta=0.002, gamma_kld=0.02,
        epochs_p1=80, epochs_p2=80,
    ),
    "rna": dict(
        mode="two_phase",
        data_dir="data/RNA/prepared_frozendist_v4_cpQ",
        z_dim=4, hidden_dim=128,
        hpz_p1=32, hpz_p2=4,
        lambda_sparse=50,
        beta=0.002, gamma_kld=0.02,
        epochs_p1=80, epochs_p2=80,
    ),
    "omni": dict(
        mode="two_phase",
        data_dir="data/omni/prepared",
        z_dim=8, hidden_dim=128,
        hpz_p1=32, hpz_p2=4,
        lambda_sparse=50,
        beta=0.002, gamma_kld=0.02,
        epochs_p1=80, epochs_p2=80,
    ),
    "climate": dict(
        mode="two_phase",
        data_dir="data/climate/prepared",
        z_dim=8, hidden_dim=128,
        hpz_p1=32, hpz_p2=4,
        lambda_sparse=50,
        beta=0.002, gamma_kld=0.02,
        epochs_p1=80, epochs_p2=80,
    ),
    "tep": dict(
        mode="two_phase",
        data_dir="data/tep/prepared",
        z_dim=8, hidden_dim=128,
        hpz_p1=32, hpz_p2=4,
        lambda_sparse=50,
        beta=0.002, gamma_kld=0.02,
        epochs_p1=80, epochs_p2=80,
    ),
    "disruption": dict(
        mode="two_phase",
        data_dir="data/disruption/prepared",
        z_dim=8, hidden_dim=128,
        hpz_p1=32, hpz_p2=4,
        lambda_sparse=50,
        beta=0.002, gamma_kld=0.02,
        epochs_p1=80, epochs_p2=80,
    ),
}


def build_commands(benchmark: str, seed: int, gpu: int):
    """Return a list of shell-style command strings for one seed."""
    cfg = BENCHMARK_CFG[benchmark]
    base = REPO / "experiments" / "crossseed" / benchmark / f"seed_{seed}"
    data_dir = REPO / cfg["data_dir"]
    cmds = []

    if cfg["mode"] == "two_phase":
        # Canonical two-stage MOSAIC: dense Phase 1 → frozen-encoder additive Phase 2.
        p1_dir = base / "phase1"
        p2_dir = base / "phase2"
        eval_dir = base / "eval"

        # Phase 1: dense decoder, no sparsity, full-capacity hpz_p1.
        cmds.append(
            f"python {PHASE1} \\\n"
            f"  --data_dir {data_dir} --log_dir {p1_dir} \\\n"
            f"  --z_dim {cfg['z_dim']} --hidden_dim {cfg['hidden_dim']} --hidden_per_z {cfg['hpz_p1']} \\\n"
            f"  --decoder_type dense \\\n"
            f"  --lambda_sparse 0 --lambda_diversity 0 --lambda_balance 0 \\\n"
            f"  --lambda_col 0 --lambda_row 0 \\\n"
            f"  --beta {cfg['beta']} --gamma_kld {cfg['gamma_kld']} \\\n"
            f"  --warmup_epochs 0 --rampup_epochs 0 --freeze_epochs 0 \\\n"
            f"  --epochs {cfg['epochs_p1']} --seed {seed} --gpu {gpu}"
        )

        # Phase 2: additive decoder, sparsity, hpz_p2.
        cmds.append(
            f"python {PHASE2} \\\n"
            f"  --data_dir {data_dir} --phase1_dir {p1_dir} --log_dir {p2_dir} \\\n"
            f"  --z_dim {cfg['z_dim']} --hidden_dim {cfg['hidden_dim']} --hidden_per_z {cfg['hpz_p2']} \\\n"
            f"  --lambda_sparse {cfg['lambda_sparse']} \\\n"
            f"  --beta {cfg['beta']} --gamma_kld {cfg['gamma_kld']} \\\n"
            f"  --epochs {cfg['epochs_p2']} --seed {seed} --gpu {gpu}"
        )

        ckpt_glob = f"{p2_dir}/lightning_logs/version_*/checkpoints/best-*.ckpt"

    elif cfg["mode"] == "single_phase":
        # RNA paper config: one-shot additive MLP + slot gate.
        train_dir = base / "train"
        eval_dir = base / "eval"

        cmds.append(
            f"python {PHASE1} \\\n"
            f"  --data_dir {data_dir} --log_dir {train_dir} \\\n"
            f"  --z_dim {cfg['z_dim']} --hidden_dim {cfg['hidden_dim']} --hidden_per_z {cfg['hidden_per_z']} \\\n"
            f"  --decoder_type additive \\\n"
            f"  --lambda_sparse {cfg['lambda_sparse']} --sparsity_mode w2 \\\n"
            f"  --lambda_diversity {cfg['lambda_diversity']} \\\n"
            + (f"  --use_slot_gate --lambda_slot_gate {cfg['lambda_slot_gate']} \\\n"
               if cfg.get('use_slot_gate') else "")
            + f"  --lambda_balance 0 --lambda_col 0 --lambda_row 0 \\\n"
            f"  --beta {cfg['beta']} --gamma_kld {cfg['gamma_kld']} \\\n"
            f"  --warmup_epochs {cfg['warmup_epochs']} "
            f"--rampup_epochs {cfg['rampup_epochs']} "
            f"--freeze_epochs {cfg['freeze_epochs']} \\\n"
            f"  --epochs {cfg['epochs']} --seed {seed} --gpu {gpu}"
        )

        ckpt_glob = f"{train_dir}/lightning_logs/version_*/checkpoints/best-*.ckpt"
    else:
        raise ValueError(f"Unknown mode: {cfg['mode']}")

    # Evaluation
    cmds.append(
        f"python {EVAL} \\\n"
        f"  --benchmark {benchmark} \\\n"
        f"  --ckpt_glob '{ckpt_glob}' \\\n"
        f"  --data_dir {data_dir} --out {eval_dir}"
    )
    return cmds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=list(BENCHMARK_CFG))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 123, 456, 789])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true",
                    help="Print commands instead of executing them")
    args = ap.parse_args()

    for seed in args.seeds:
        print(f"\n========== {args.benchmark} seed={seed} ==========")
        for cmd in build_commands(args.benchmark, seed, args.gpu):
            print(cmd)
            if not args.dry_run:
                rc = subprocess.call(cmd, shell=True, cwd=str(REPO))
                if rc != 0:
                    sys.exit(rc)


if __name__ == "__main__":
    main()
