#!/usr/bin/env python3
"""train_joint.py — Joint (single-stage) training ablation.

Contrast with the two-stage MOSAIC curriculum (train_phase1 + train_phase2):
  * Phase 1: dense decoder, no sparsity, encoder + prior trained to ELBO.
  * Phase 2: encoder/prior frozen, additive decoder fit with sparsity penalty.

Joint variant (this script):
  * Additive decoder from epoch 0.
  * Encoder + prior are NOT frozen (``--freeze_epochs 0``).
  * Sparsity penalty on from the start, with the same warmup → ramp schedule.
  * Epoch budget matches Phase 1 + Phase 2 combined.

Implementation: thin wrapper around ``train_phase1.py``, which already
supports all the flags we need. We set the joint-training defaults here so
the ablation is a single command.

Usage::

    python scripts/train/train_joint.py \\
        --data_dir data/synthetic_well_v2/source \\
        --log_dir experiments/joint/seed_42 \\
        --z_dim 8 --hidden_dim 128 --hidden_per_z 4 \\
        --beta 2e-3 --gamma_kld 2e-2 --lr 5e-4 \\
        --lambda_sparse 50 --warmup_epochs 5 --rampup_epochs 20 \\
        --epochs 160 --seed 42 --gpu 0
"""
import os, sys, argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PHASE1 = os.path.join(_HERE, 'train_phase1.py')


def build_argv():
    ap = argparse.ArgumentParser(description='Joint-training ablation wrapper')
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--log_dir',  required=True)
    ap.add_argument('--z_dim', type=int, default=8)
    ap.add_argument('--hidden_dim', type=int, default=128)
    ap.add_argument('--hidden_per_z', type=int, default=4)
    ap.add_argument('--beta', type=float, default=2e-3)
    ap.add_argument('--gamma_kld', type=float, default=2e-2)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--lambda_sparse', type=float, default=50.0)
    ap.add_argument('--warmup_epochs', type=int, default=5)
    ap.add_argument('--rampup_epochs', type=int, default=20)
    ap.add_argument('--epochs', type=int, default=160)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    # Build the argv we want to forward to train_phase1.py.
    # Key joint-training choices:
    #   decoder_type = additive  (NOT dense → additive switch)
    #   freeze_epochs = 0        (NOT freeze encoder at any point)
    #   lambda_sparse = 50       (same λ_max as Phase 2)
    #   warmup/ramp identical to Phase 2 schedule
    argv = [
        sys.executable, SCRIPT_PHASE1,
        '--data_dir',     args.data_dir,
        '--log_dir',      args.log_dir,
        '--z_dim',        str(args.z_dim),
        '--hidden_dim',   str(args.hidden_dim),
        '--hidden_per_z', str(args.hidden_per_z),
        '--decoder_type', 'additive',
        '--beta',         str(args.beta),
        '--gamma_kld',    str(args.gamma_kld),
        '--lr',           str(args.lr),
        '--lambda_sparse', str(args.lambda_sparse),
        '--lambda_diversity', '0',
        '--lambda_balance',   '0',
        '--warmup_epochs', str(args.warmup_epochs),
        '--rampup_epochs', str(args.rampup_epochs),
        '--freeze_epochs', '0',
        '--epochs',        str(args.epochs),
        '--batch_size',    str(args.batch_size),
        '--seed',          str(args.seed),
    ]
    if args.gpu is not None:
        argv += ['--gpu', str(args.gpu)]
    return argv


def main():
    argv = build_argv()
    print('[train_joint] forwarding to train_phase1.py with:')
    for i, tok in enumerate(argv[2:]):
        print(f'  {tok}', end='')
        if tok.startswith('--') or i == len(argv) - 3:
            print()
    print()
    os.execvp(argv[0], argv)


if __name__ == '__main__':
    main()
