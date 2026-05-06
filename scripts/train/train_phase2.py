#!/usr/bin/env python3
"""
twophase_p2.py — Phase 2: Linear decoder + group lasso on frozen encoder

Loads encoder + transition prior from Phase 1 (dense decoder).
Creates linear additive decoder: x = W @ z + bias, W is (D, z_dim).
Freezes encoder + transition prior, trains only W + bias with group lasso.

W IS the Jacobian. No hidden layer, no bypass, no ambiguity.
Group lasso on columns of W: ||W[:, j]||_1 penalizes total influence of z_j.
Entropy on rows of W: for each x_i, which z_j dominates? (column exclusivity)
"""

import os
import sys
import glob
import torch
import argparse
import inspect
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, Subset

import warnings
warnings.filterwarnings('ignore')

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mosaic.models import TimeVaryingProcess_v2
from mosaic.data import MDRegimeDataset


class FreezeEncoderPrior(pl.Callback):
    """Keep encoder and transition prior frozen throughout."""
    def on_train_epoch_start(self, trainer, pl_module):
        for name, param in pl_module.named_parameters():
            if 'net.decoder' in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)


class MetricPrintCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        parts = []
        for key in ['val_elbo_loss', 'val_recon_loss', 'val_regime_acc',
                     'val_sparse_loss', 'val_concentration', 'val_active_z']:
            if key in m:
                parts.append(f"{key.replace('val_','')}: {m[key]:.4f}")
        if parts:
            print(f"[Epoch {trainer.current_epoch}] {', '.join(parts)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--phase1_dir", required=True)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--z_dim", type=int, required=True)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--hidden_per_z", type=int, default=0,
                        help="Hidden units per z in Phase 2 decoder. 0=linear, >0=MLP additive.")
    parser.add_argument("--lambda_sparse", type=float, default=50)
    parser.add_argument("--sparsity_mode", default='w2',
                        choices=['w2', 'functional', 'func_l1', 'group_lasso'],
                        help="Sparsity penalty type on Phase 2 decoder. "
                             "'w2'/'functional'=entropy (default), "
                             "'func_l1'=mean-|A|, "
                             "'group_lasso'=sum_j ||A[:,j]||_2.")
    parser.add_argument("--lambda_col", type=float, default=0.0,
                        help="Column exclusivity: push each X to bind one Z")
    parser.add_argument("--beta", type=float, default=0.002)
    parser.add_argument("--gamma_kld", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--nclass", type=int, default=2,
                        help="Number of regime classes (default 2; set to 3 for multi-regime)")
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    # Load data
    dataset = MDRegimeDataset(os.path.join(args.data_dir, "data.npz"))
    input_dim = dataset.data['xt'].shape[-1]
    seq_len = dataset.data['xt'].shape[1]
    lag = seq_len - 1

    # Reuse Phase 1 split
    split = np.load(os.path.join(args.phase1_dir, 'split_indices.npz'))
    train_data = Subset(dataset, split['train_indices'].tolist())
    val_data = Subset(dataset, split['val_indices'].tolist())
    test_data = Subset(dataset, split['test_indices'].tolist())
    print(f"Split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              pin_memory=True, num_workers=args.num_workers, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=512,
                            pin_memory=True, num_workers=args.num_workers, shuffle=False)

    # Find Phase 1 checkpoint (use latest version)
    ckpt_dirs = sorted(glob.glob(os.path.join(args.phase1_dir, 'lightning_logs/version_*/checkpoints')))
    if not ckpt_dirs:
        raise FileNotFoundError(f"No Phase 1 checkpoints in {args.phase1_dir}")
    ckpts = glob.glob(os.path.join(ckpt_dirs[-1], 'best-*.ckpt'))
    if not ckpts:
        raise FileNotFoundError(f"No best checkpoint in {ckpt_dirs[-1]}")
    p1_ckpt = ckpts[0]
    print(f"Phase 1 checkpoint: {p1_ckpt}")

    # Create Phase 2 model: additive decoder (linear if hidden_per_z=0, MLP otherwise)
    hpz = args.hidden_per_z
    model = TimeVaryingProcess_v2(
        input_dim=input_dim, length=1, z_dim=args.z_dim, lag=lag, nclass=args.nclass,
        hidden_dim=args.hidden_dim, embedding_dim=8,
        hidden_per_z=hpz,
        decoder_type='additive', trans_prior='NP',
        lr=5e-4, infer_mode='F',
        beta=args.beta, gamma=args.gamma_kld,
        lambda_sparse=args.lambda_sparse,
        sparsity_mode=args.sparsity_mode,
        lambda_diversity=1.0,
        lambda_col=args.lambda_col,  # column exclusivity: each X bound to ONE Z
        warmup_epochs=0, rampup_epochs=10, freeze_epochs=0,
        decoder_dist='gaussian',
    )

    # Load encoder + transition prior from Phase 1
    p1_sd = torch.load(p1_ckpt, map_location='cpu')['state_dict']
    model_sd = model.state_dict()
    loaded, skipped = 0, 0
    for key, val in p1_sd.items():
        if 'decoder' in key:
            skipped += 1
            continue
        if key in model_sd and model_sd[key].shape == val.shape:
            model_sd[key] = val
            loaded += 1
        else:
            skipped += 1
    model.load_state_dict(model_sd)
    print(f"Loaded {loaded} params from Phase 1, skipped {skipped}")

    decoder_desc = f"LINEAR (W is {input_dim}x{args.z_dim})" if hpz == 0 else f"MLP additive (hidden_per_z={hpz})"
    print(f"\nPhase 2: {decoder_desc}")
    print(f"λ_sparse={args.lambda_sparse}, sparsity_mode={args.sparsity_mode}")
    print(f"Encoder + transition prior: FROZEN")

    # CRITICAL: Freeze encoder + transition prior BEFORE optimizer creation
    frozen_count = 0
    for name, param in model.named_parameters():
        if 'net.decoder' not in name:
            param.requires_grad_(False)
            frozen_count += 1
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"Frozen {frozen_count} params, {trainable} trainable (decoder only)")

    # Setup
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    np.savez_compressed(os.path.join(log_dir, 'split_indices.npz'),
                        train_indices=split['train_indices'],
                        val_indices=split['val_indices'],
                        test_indices=split['test_indices'])

    callbacks = [
        FreezeEncoderPrior(),
        pl.callbacks.ModelCheckpoint(monitor='val_elbo_loss', save_top_k=1,
                                     mode='min', filename='best-{epoch}-{val_elbo_loss:.4f}'),
        MetricPrintCallback(),
    ]

    trainer_kwargs = dict(
        default_root_dir=log_dir, val_check_interval=0.5,
        max_epochs=args.epochs, callbacks=callbacks,
    )
    trainer_sig = inspect.signature(pl.Trainer.__init__).parameters
    if 'devices' in trainer_sig:
        if args.gpu is not None and torch.cuda.is_available():
            trainer_kwargs['accelerator'] = 'gpu'
            trainer_kwargs['devices'] = [args.gpu]
        else:
            trainer_kwargs['accelerator'] = 'auto'
            trainer_kwargs['devices'] = 1
    if 'enable_progress_bar' in trainer_sig:
        trainer_kwargs['enable_progress_bar'] = True

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, train_loader, val_loader)

    # Test
    test_loader = DataLoader(test_data, batch_size=512, pin_memory=True,
                             num_workers=args.num_workers, shuffle=False)
    test_results = trainer.test(model, test_loader, ckpt_path='best')
    print(f"Test results: {test_results}")

    # Support analysis
    # NOTE: AdditiveDecoder always sets BOTH W_lin and W2 as attributes; one of
    # them is None depending on hidden_per_z. Check `is not None`, not hasattr.
    W_lin = getattr(model.net.decoder, 'W_lin', None)
    W2    = getattr(model.net.decoder, 'W2',    None)
    if W_lin is not None:
        W = W_lin.detach().cpu().numpy()                    # (z_dim, D)
        W_abs = np.abs(W)
    elif W2 is not None:
        W2_np = W2.detach().cpu().numpy()                   # (z_dim, D, hpz)
        W_abs = np.linalg.norm(W2_np, axis=2)               # (z_dim, D)
        W = W_abs
    else:
        print("Unknown decoder type, skipping support analysis")
        return

    # Always export the canonical influence matrix A_ij = |f_j(+1)_i - f_j(-1)_i|
    # so downstream eval can read it without re-loading the checkpoint.
    try:
        zd = args.z_dim
        cols = []
        with torch.no_grad():
            dec = model.net.decoder
            dev = next(dec.parameters()).device
            for j in range(zd):
                zp = torch.zeros(1, zd, device=dev); zp[0, j] = 1.0
                zn = torch.zeros(1, zd, device=dev); zn[0, j] = -1.0
                out = (dec(zp) - dec(zn)).abs().squeeze().detach().cpu().numpy()
                cols.append(out)
        A = np.stack(cols, axis=1)                            # (D, z_dim)
        np.savez_compressed(os.path.join(log_dir, 'influence_A.npz'),
                            A=A, W_abs=W_abs)
        print(f"Saved canonical influence A {A.shape} to {log_dir}/influence_A.npz")
    except Exception as e:
        print(f"[warn] influence A export failed: {e}")

    print(f"\n=== Support Analysis ===")
    print(f"W shape: {W.shape} (z_dim x D)")

    # RNA-specific block expects D == 28 (14 residues × {μ, σ}). Skip otherwise
    # so non-RNA datasets (e.g. K8, D=45) exit cleanly with code 0.
    if input_dim != 28:
        print(f"  (input_dim={input_dim} ≠ 28; skipping RNA-specific module audit)")
        return

    RNA = 'GGCACUUCGGUGCC'
    MODULES = {
        'OuterStem':   [0, 1, 12, 13],
        'InnerStem':   [2, 3, 10, 11],
        'ClosingPair': [4, 9],
        'Loop':        [5, 6, 7, 8],
    }
    RES_TO_MOD = {}
    for mod, residues in MODULES.items():
        for r in residues:
            RES_TO_MOD[r] = mod

    def feat_name(i):
        r = i if i < 14 else i - 14
        t = 'μ' if i < 14 else 'σ'
        return f'{RNA[r]}{r+1}_{t}'

    # Per-z analysis
    assign = W_abs.T.argmax(axis=1)  # (D,) which z dominates each x
    from collections import Counter
    for j in range(W.shape[0]):
        feats = np.where(assign == j)[0]
        if len(feats) == 0:
            print(f"  z{j}: (empty)")
            continue
        residues = [f if f < 14 else f - 14 for f in feats]
        modules = [RES_TO_MOD[r] for r in residues]
        mod_counts = Counter(modules)
        dominant = mod_counts.most_common(1)[0]
        purity = dominant[1] / len(modules)
        names = [feat_name(f) for f in feats]
        print(f"  z{j} ({len(feats):2d} feats, purity={purity:.2f}, best={dominant[0]})")
        print(f"       {names}")
        print(f"       {dict(mod_counts)}")

    # Concentration
    maxes = W_abs.T.max(axis=1)
    sums = W_abs.T.sum(axis=1)
    conc = (maxes / (sums + 1e-8)).mean()
    print(f"\n  Mean concentration: {conc:.4f}")

    # Module coverage
    print(f"\n  --- Module coverage ---")
    for mod, gt_res in MODULES.items():
        gt_feats = gt_res + [r + 14 for r in gt_res]
        z_for_mod = [assign[f] for f in gt_feats]
        cnt = Counter(z_for_mod)
        dominant = cnt.most_common(1)[0]
        print(f"  {mod:12s}: {dict(cnt)}, best=z{dominant[0]} ({dominant[1]}/{len(gt_feats)})")


if __name__ == "__main__":
    main()
