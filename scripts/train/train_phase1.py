#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_phase1.py — Train MOSAIC (Additive Decoder + sparsity)

Fine-tunes from a v1 checkpoint:
  - Loads encoder + transition prior + embedding from v1
  - AdditiveDecoder initialized randomly
  - Freeze schedule: decoder-only warmup → joint training
  - Lambda schedule: no sparsity → ramp → full sparsity

Usage:
    # Fine-tune from v1 (recommended)
    python train_phase1.py \
        --v1_ckpt results_phase1/lightning_logs/version_3/checkpoints/best-epoch=21-val_elbo_loss=195.5096.ckpt

    # Train from scratch (no v1 weights)
    python train_phase1.py

    # Custom sparsity and GPU
    python train_phase1.py --v1_ckpt <path> --lambda_sparse 5e-3 --gpu 0
"""

import os
import sys
import torch
import argparse
import inspect
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

import warnings
warnings.filterwarnings('ignore')

# Make the `mosaic` package importable when this script is run directly
# (e.g. `python scripts/train/train_phase1.py`). pip install -e . is preferred.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mosaic.models import TimeVaryingProcess_v2
from mosaic.data import MDRegimeDataset


# ============================================================
# Callbacks
# ============================================================
class MetricPrintCallback(Callback):
    def on_validation_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        parts = []
        for key in ['val_elbo_loss', 'val_recon_loss', 'val_regime_acc',
                     'val_sparse_loss', 'val_concentration', 'val_active_z']:
            if key in m:
                parts.append(f"{key.replace('val_','')}: {m[key]:.4f}")
        if parts:
            lam = m.get('train_lambda', 0)
            print(f"[Epoch {trainer.current_epoch}] λ={lam:.1e} | {', '.join(parts)}")


# ============================================================
# Post-training visualization
# ============================================================
def visualize_results(model, dataloader, output_dir, device='cpu'):
    """Visualize latent space + module assignment."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    model.eval()
    model.to(device)

    # Extract latent representations
    all_mus, all_ct = [], []
    with torch.no_grad():
        for batch in dataloader:
            x = batch['xt'].to(device)
            c = batch['ct']
            batch_size, length, _ = x.shape
            x_flat = x.view(-1, model.input_dim)
            _, mus, _, _ = model.net(x_flat)
            mus = mus.reshape(batch_size, length, model.z_dim)
            all_mus.append(mus.mean(dim=1).cpu().numpy())
            all_ct.append(c.squeeze().cpu().numpy())

    all_mus = np.concatenate(all_mus)
    all_ct = np.concatenate(all_ct)

    # Save embeddings
    np.savez_compressed(os.path.join(output_dir, 'latent_embeddings_v2.npz'),
                        mus=all_mus, ct=all_ct)

    # --- Plot 1: t-SNE ---
    max_pts = 10000
    if len(all_mus) > max_pts:
        idx = np.random.RandomState(42).choice(len(all_mus), max_pts, replace=False)
        plot_mus, plot_ct = all_mus[idx], all_ct[idx]
    else:
        plot_mus, plot_ct = all_mus, all_ct

    print("Running t-SNE...")
    z_2d = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(plot_mus)

    fig, ax = plt.subplots(figsize=(10, 8))
    for regime, color, label in [(0, '#1f77b4', 'Normal'), (1, '#ff7f0e', 'Changing')]:
        mask = plot_ct == regime
        ax.scatter(z_2d[mask, 0], z_2d[mask, 1], c=color, label=label, s=5, alpha=0.5)
    ax.legend(fontsize=12, markerscale=3)
    ax.set_title('MOSAIC Latent Space — t-SNE')
    fig.savefig(os.path.join(output_dir, 'latent_tsne_v2.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # --- Plot 2: Module assignment heatmap (additive decoder only) ---
    influence = model.net.decoder.get_influence_matrix()
    if influence is not None:
        # Merge sin/cos: sqrt(sin^2 + cos^2) if output_dim is even
        if influence.shape[0] % 2 == 0 and influence.shape[0] > influence.shape[1]:
            n_angles = influence.shape[0] // 2
            inf_sin = influence[:n_angles, :]
            inf_cos = influence[n_angles:, :]
            inf_merged = np.sqrt(inf_sin**2 + inf_cos**2)
        else:
            n_angles = influence.shape[0]
            inf_merged = influence

        # Sort z-dims by total influence
        z_total = inf_merged.sum(axis=0)
        z_order = np.argsort(z_total)[::-1]
        inf_sorted = inf_merged[:, z_order]

        fig, axes = plt.subplots(2, 1, figsize=(16, 8), height_ratios=[4, 1],
                                  gridspec_kw={'hspace': 0.05})

        vmax = np.percentile(inf_sorted, 99)
        im = axes[0].imshow(inf_sorted.T, aspect='auto', cmap='hot', vmin=0, vmax=vmax)
        axes[0].set_ylabel('Latent z (sorted by influence)')
        axes[0].set_title('AdditiveDecoder Module Structure: ||W2[j,i,:]||₂')
        plt.colorbar(im, ax=axes[0], fraction=0.02)

        dominant_z = inf_merged.argmax(axis=1)
        concentration = inf_merged.max(axis=1) / (inf_merged.sum(axis=1) + 1e-10)
        axes[1].bar(range(n_angles), concentration, width=1.0, color='steelblue')
        axes[1].set_xlim(-0.5, n_angles - 0.5)
        axes[1].set_ylabel('Concentration')
        axes[1].set_xlabel('Observation index')

        fig.savefig(os.path.join(output_dir, 'module_structure_v2.png'), dpi=150, bbox_inches='tight')
        plt.close()

        n_active = len(np.unique(dominant_z))
        print(f"\n  Active z-dims: {n_active}/{model.z_dim}")
        print(f"  Mean concentration: {concentration.mean():.4f}")
        print(f"  Median z-dims for 80% attr: ", end="")
        n_needed = []
        for i in range(n_angles):
            row = np.sort(inf_merged[i])[::-1]
            cum = np.cumsum(row) / (row.sum() + 1e-10)
            n_needed.append(np.searchsorted(cum, 0.8) + 1)
        print(f"{np.median(n_needed):.0f}")
    else:
        print("\n  Dense decoder: no module structure to visualize")

    print(f"\n  [SAVED] {output_dir}/latent_tsne_v2.png")
    print(f"  [SAVED] {output_dir}/module_structure_v2.png")


# ============================================================
# Main
# ============================================================
def main(args):
    print("=" * 80)
    print("MOSAIC Training: Additive Decoder + Mechanism Sparsity")
    print("=" * 80)

    pl.seed_everything(args.seed)

    # Load data
    data_path = os.path.join(args.data_dir, "data.npz")
    dataset = MDRegimeDataset(data_path)

    input_dim = dataset.data['xt'].shape[-1]
    seq_len = dataset.data['xt'].shape[1]
    lag = seq_len - 1
    print(f"\nInferred: input_dim={input_dim}, seq_len={seq_len}, lag={lag}")

    # Train/val/test split
    if args.split_from:
        # Load existing split (e.g. from v1) to avoid data leakage
        split = np.load(args.split_from)
        from torch.utils.data import Subset
        train_data = Subset(dataset, split['train_indices'].tolist())
        val_data = Subset(dataset, split['val_indices'].tolist())
        test_data = Subset(dataset, split['test_indices'].tolist())
        print(f"Loaded split from {args.split_from}")
        print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    else:
        n_val = min(args.n_val, len(dataset) // 5)
        n_test = min(args.n_test, len(dataset) // 5)
        n_train = len(dataset) - n_val - n_test
        train_data, val_data, test_data = random_split(dataset, [n_train, n_val, n_test])
        print(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")

    # Save split indices for reproducibility
    split_path = os.path.join(args.log_dir, 'split_indices.npz')
    os.makedirs(args.log_dir, exist_ok=True)
    train_idx = train_data.indices if hasattr(train_data, 'indices') else list(range(len(train_data)))
    val_idx = val_data.indices if hasattr(val_data, 'indices') else list(range(len(val_data)))
    test_idx = test_data.indices if hasattr(test_data, 'indices') else list(range(len(test_data)))
    np.savez_compressed(split_path,
                        train_indices=np.array(train_idx),
                        val_indices=np.array(val_idx),
                        test_indices=np.array(test_idx))
    print(f"[SAVED] Split indices → {split_path}")

    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              pin_memory=True, num_workers=args.num_workers,
                              drop_last=False, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.val_batch_size,
                            pin_memory=True, num_workers=args.num_workers,
                            shuffle=False)

    # Create v2 model
    model = TimeVaryingProcess_v2(
        input_dim=input_dim,
        length=1,
        z_dim=args.z_dim,
        lag=lag,
        nclass=args.nclass,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        hidden_per_z=args.hidden_per_z,
        decoder_type=args.decoder_type,
        trans_prior=args.trans_prior,
        lr=args.lr,
        infer_mode='F',
        beta=args.beta,
        gamma=args.gamma_kld,
        lambda_sparse=args.lambda_sparse,
        lambda_col_entropy=args.lambda_col_entropy,
        lambda_balance=args.lambda_balance,
        lambda_diversity=args.lambda_diversity,
        lambda_slot_gate=args.lambda_slot_gate,
        lambda_regime_gate=args.lambda_regime_gate,
        lambda_trans_sparsity=args.lambda_trans_sparsity,
        use_slot_gate=args.use_slot_gate,
        sparsity_mode=args.sparsity_mode,
        lambda_col=getattr(args, 'lambda_col', 0.0),
        lambda_row=getattr(args, 'lambda_row', 0.0),
        use_ard=args.use_ard,
        ard_a=args.ard_a,
        ard_b=args.ard_b,
        warmup_epochs=args.warmup_epochs,
        rampup_epochs=args.rampup_epochs,
        freeze_epochs=args.freeze_epochs,
        decoder_dist='gaussian',
    )

    print(f"\nModel: z_dim={args.z_dim}, hidden_dim={args.hidden_dim}, "
          f"hidden_per_z={args.hidden_per_z}, decoder={args.decoder_type}")
    print(f"ELBO weights: β={args.beta}, γ={args.gamma_kld}")
    print(f"Sparsity: λ={args.lambda_sparse}, mode={args.sparsity_mode}, "
          f"warmup={args.warmup_epochs}, ramp={args.rampup_epochs}, freeze={args.freeze_epochs}")
    if args.use_slot_gate:
        print(f"Slot gate: on, λ={args.lambda_slot_gate}, active_threshold=0.2")

    # Load weights
    resume_ckpt = None
    if args.resume_ckpt:
        # Resume from v2 checkpoint (full model, including decoder)
        resume_ckpt = args.resume_ckpt
        print(f"\nResuming from v2 checkpoint: {resume_ckpt}")
    elif args.v1_ckpt:
        model.load_v1_weights(args.v1_ckpt)
    else:
        print("\nTraining from scratch (no checkpoint)")

    # Callbacks
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)

    # Save best (by elbo_loss) + always save last epoch
    monitor_metric = 'val_elbo_loss'
    callbacks = [
        ModelCheckpoint(monitor=monitor_metric, save_top_k=1, mode='min',
                        filename='best-{epoch}-{val_elbo_loss:.4f}'),
        ModelCheckpoint(save_last=True, filename='last-{epoch}'),
        MetricPrintCallback(),
    ]
    if args.checkpoint_every_n_epochs > 0:
        periodic_kwargs = dict(save_top_k=-1, filename='periodic-{epoch}')
        ckpt_sig = inspect.signature(ModelCheckpoint.__init__).parameters
        if 'every_n_epochs' in ckpt_sig:
            periodic_kwargs['every_n_epochs'] = args.checkpoint_every_n_epochs
        else:
            periodic_kwargs['period'] = args.checkpoint_every_n_epochs
        callbacks.insert(2, ModelCheckpoint(**periodic_kwargs))
    if args.early_stop_patience > 0:
        callbacks.append(EarlyStopping(
            monitor=monitor_metric, min_delta=0.0,
            patience=args.early_stop_patience, verbose=True, mode="min"))

    trainer_kwargs = dict(
        default_root_dir=log_dir,
        val_check_interval=0.5,
        max_epochs=args.epochs,
        callbacks=callbacks,
        gradient_clip_val=getattr(args, 'gradient_clip_val', None),
    )

    trainer_sig = inspect.signature(pl.Trainer.__init__).parameters

    # GPU selection with Lightning version compatibility
    if 'devices' in trainer_sig:
        if args.gpu is not None and torch.cuda.is_available():
            trainer_kwargs['accelerator'] = 'gpu'
            trainer_kwargs['devices'] = [args.gpu]
        else:
            trainer_kwargs['accelerator'] = 'auto'
            trainer_kwargs['devices'] = 1
    elif 'gpus' in trainer_sig:
        if torch.cuda.is_available():
            trainer_kwargs['gpus'] = [args.gpu] if args.gpu is not None else 1
        else:
            trainer_kwargs['gpus'] = 0

    if 'enable_progress_bar' in trainer_sig:
        trainer_kwargs['enable_progress_bar'] = True
    elif 'progress_bar_refresh_rate' in trainer_sig:
        trainer_kwargs['progress_bar_refresh_rate'] = 1

    if resume_ckpt and 'resume_from_checkpoint' in trainer_sig:
        trainer_kwargs['resume_from_checkpoint'] = resume_ckpt

    trainer = pl.Trainer(**trainer_kwargs)

    # Train
    print("\nStarting training...")
    fit_sig = inspect.signature(trainer.fit).parameters
    if 'ckpt_path' in fit_sig:
        trainer.fit(model, train_loader, val_loader, ckpt_path=resume_ckpt)
    else:
        trainer.fit(model, train_loader, val_loader)
    print("\nTraining complete!")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loader = DataLoader(test_data, batch_size=args.val_batch_size,
                             pin_memory=True, num_workers=args.num_workers,
                             shuffle=False)
    test_results = trainer.test(model, test_loader, ckpt_path='best')
    print(f"Test results: {test_results}")

    # Post-training visualization
    print("\nGenerating visualizations...")
    vis_dir = os.path.join(log_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    device = f'cuda:{args.gpu}' if args.gpu is not None and torch.cuda.is_available() else 'cpu'
    full_loader = DataLoader(dataset, batch_size=args.val_batch_size,
                             shuffle=False, num_workers=args.num_workers)
    visualize_results(model, full_loader, vis_dir, device=device)

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MOSAIC")

    # Data
    parser.add_argument("--data_dir", default="./data/synthetic/source")
    parser.add_argument("--log_dir", default="./results_phase2")

    # Checkpoints
    parser.add_argument("--v1_ckpt", default=None,
                        help="Path to v1 checkpoint for weight initialization")
    parser.add_argument("--resume_ckpt", default=None,
                        help="Path to v2 checkpoint to resume training from")
    parser.add_argument("--split_from", default=None,
                        help="Path to split_indices.npz from v1 to reuse the same train/val/test split")

    # Model architecture (must match v1 for weight loading)
    parser.add_argument("--z_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embedding_dim", type=int, default=8)
    parser.add_argument("--nclass", type=int, default=2)
    parser.add_argument("--trans_prior", default='NP', choices=['L', 'NP'])
    parser.add_argument("--hidden_per_z", type=int, default=8,
                        help="Hidden units per sub-decoder in AdditiveDecoder (default: 8)")
    parser.add_argument("--decoder_type", default='additive', choices=['additive', 'dense'],
                        help="Decoder architecture: 'additive' (default) or 'dense' (ablation)")
    parser.add_argument("--sparsity_mode", default='w2', choices=['w2', 'functional', 'func_l1'],
                        help="Sparsity target: 'w2' (W2 norms), 'functional' (entropy on |f(+1)-f(-1)|), 'func_l1' (L1 on |f(+1)-f(-1)|)")

    # ELBO weights (match v1 defaults)
    parser.add_argument("--beta", type=float, default=2e-3)
    parser.add_argument("--gamma_kld", type=float, default=2e-2)

    # Sparsity
    parser.add_argument("--lambda_sparse", type=float, default=1e-3,
                        help="Decoder sparsity weight (default: 1e-3)")
    parser.add_argument("--lambda_col_entropy", type=float, default=0.0,
                        help="Column entropy weight, scaled by lambda_sparse (default: 0)")
    parser.add_argument("--lambda_balance", type=float, default=0.0,
                        help="Balance loss weight to prevent z-collapse (default: 0)")
    parser.add_argument("--lambda_diversity", type=float, default=0.0,
                        help="Diversity loss weight. >0 penalizes z-dims with similar influence profiles (default: 0)")
    parser.add_argument("--use_slot_gate", action="store_true",
                        help="Enable a learnable global sigmoid gate per z-dim in the additive decoder")
    parser.add_argument("--lambda_slot_gate", type=float, default=0.0,
                        help="Slot gate penalty weight, scaled by lambda_sparse (default: 0)")
    parser.add_argument("--lambda_regime_gate", type=float, default=0.0,
                        help="Regime-dependent gating weight. >0 enables regime gating on decoder (default: 0)")
    parser.add_argument("--lambda_trans_sparsity", type=float, default=0.0,
                        help="Transition-aware sparsity weight. >0 modulates group-lasso by transition residuals (default: 0)")
    parser.add_argument("--lambda_col", type=float, default=0.0,
                        help="Column exclusivity weight: each x_i dominated by one z (default: 0)")
    parser.add_argument("--lambda_row", type=float, default=0.0,
                        help="Row L1 weight: weak penalty on total influence per z (default: 0)")
    # ARD prior
    parser.add_argument("--use_ard", action="store_true",
                        help="Use ARD prior for automatic z-dim selection")
    parser.add_argument("--ard_a", type=float, default=1.0,
                        help="Gamma hyperprior shape parameter (default: 1.0)")
    parser.add_argument("--ard_b", type=float, default=0.1,
                        help="Gamma hyperprior rate parameter. Higher = more pruning (default: 0.1)")

    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Epochs before sparsity starts (default: 5)")
    parser.add_argument("--rampup_epochs", type=int, default=20,
                        help="Epochs to ramp λ from 0 to target (default: 20)")
    parser.add_argument("--freeze_epochs", type=int, default=5,
                        help="Epochs to freeze encoder/transition (default: 5)")

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--val_batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--n_val", type=int, default=4096)
    parser.add_argument("--n_test", type=int, default=4096,
                        help="Number of test samples (default: 4096)")
    parser.add_argument("--early_stop_patience", type=int, default=0,
                        help="Early stopping patience. 0 disables early stopping (default: 0)")
    parser.add_argument("--gradient_clip_val", type=float, default=None,
                        help="Gradient clipping value (default: None = no clipping)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_every_n_epochs", type=int, default=10,
                        help="Save periodic checkpoint every N epochs (default: 10)")

    # GPU
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index to use (default: auto)")

    args = parser.parse_args()
    main(args)
