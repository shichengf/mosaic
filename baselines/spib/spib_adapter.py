#!/usr/bin/env python3
"""
Task 3: SPIB (State Predictive Information Bottleneck) adapter.

SPIB code in-repo: baselines/spib/

SPIB expects:
  traj_data:   (T, D) float32  observations
  traj_label:  (T, n_states) one-hot state labels
  t0, dt:      past frame at t, label is state at t+dt (dt=10 per our plan)
  SPIB trains to classify future state from current observation bottleneck z.

Our data (prepared .npz): xt (N, seq_len, D) + ct (N, 1).
We use xt[:, -1, :] as observations, ct as state labels.
One-hot encode ct to (N, 2).

Usage:
  python baselines/spib/spib_adapter.py --dataset synthetic_mono --seed 42 --out_dir experiments/spib/synthetic_mono_seed42

After training, evaluates via scripts/eval/eval_unified.py and appends a row to:
  experiments/spib/spib_{dataset}.csv
"""
from __future__ import annotations
import os, sys, argparse, json, csv, warnings, time
from pathlib import Path
import numpy as np
import torch
warnings.filterwarnings('ignore')

REPO_ROOT = Path(__file__).resolve().parents[2]
SPIB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SPIB_DIR))
sys.path.insert(0, str(REPO_ROOT / "scripts/eval"))

import SPIB, SPIB_training
from eval_unified import Evaluator, Dataset, encoder_jacobian_influence

# ═══════════════════════════════════════════════════════════════════
# Dataset → SPIB config
# ═══════════════════════════════════════════════════════════════════
DATASET_MAP = {
    'synthetic_mono': (Dataset.SYNTHETIC, 8),
    'rna':        (Dataset.RNA,        4),
    'omni':       (Dataset.OMNI,       5),
    'disruption': (Dataset.DISRUPTION, 5),
    'climate':    (Dataset.CLIMATE,    8),
}

CSV_FIELDS = [
    'dataset', 'method', 'seed', 'regime_acc', 'driver_z', 'driver_d', 'top3_z',
    'd_vals', 'mcc', 'matched_factors', 'z_top3', 'x_z_top3', 'module_pct',
    'driver_module', 'driver_rank_by_target', 'x_z_top_k', 'x_z_top_4',
    'top_driver_vars', 'domain_consistent', 'notes',
]


def one_hot(labels: np.ndarray, n_states: int = 2) -> torch.Tensor:
    out = np.zeros((len(labels), n_states), dtype=np.float32)
    out[np.arange(len(labels)), labels.astype(int)] = 1.0
    return torch.from_numpy(out)


def train_spib(dataset_name, seed, out_dir,
               dt=10, z_dim=None, batch_size=2048, beta=0.01,
               lr=1e-3, n1=128, n2=128, encoder_type='Nonlinear',
               refinements=0, threshold=0.01, patience=0,
               step_size=10, gamma=1.0, log_interval=5000, t0=0,
               n_states=2):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)

    ds_enum, default_z = DATASET_MAP[dataset_name]
    if z_dim is None:
        z_dim = default_z
    ev = Evaluator(ds_enum)
    X_last = ev.X_last.astype(np.float32)
    ct = ev.ct.astype(int)

    print(f'[spib] {dataset_name} seed={seed} z={z_dim} dt={dt}  N={len(X_last)}  D={X_last.shape[1]}')

    traj_data  = torch.from_numpy(X_last)
    # When sweeping K>2, replicate GT-2-state labels into first two slots and
    # let SPIB refinement (UpdateLabel=True + refinements>0) split them.
    traj_label = one_hot(ct, n_states=n_states)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    t_start = time.time()
    (data_shape, ptd_tr, ftd_tr, lab_tr, _w_tr,
     ptd_te, ftd_te, lab_te, _w_te) = SPIB_training.data_init(
         t0, dt, traj_data, traj_label, None)

    # Must use UpdateLabel=True because SPIB_training.train() unconditionally calls
    # IB.update_labels() and crashes on None when UpdateLabel=False (upstream bug).
    # We disable actual label updates by setting refinements=0 (no iterative refinement).
    IB = SPIB.SPIB(encoder_type=encoder_type,
                   z_dim=z_dim, output_dim=n_states, data_shape=data_shape,
                   device=device, UpdateLabel=True,
                   neuron_num1=n1, neuron_num2=n2).to(device)

    nan_flag = SPIB_training.train(
        IB, beta, ptd_tr, ftd_tr, lab_tr, None,
        ptd_te, ftd_te, lab_te, None,
        lr, step_size, gamma, batch_size, threshold, patience, refinements,
        f'{out_dir}/spib_', log_interval, device, index=0)
    if nan_flag:
        print(f'[spib] {dataset_name} seed={seed}: NaN during training, aborting.')
        return None
    print(f'[spib] trained in {time.time()-t_start:.1f}s')

    # Save checkpoint
    ckpt_path = f'{out_dir}/spib_final.pt'
    torch.save({'state_dict': IB.state_dict(),
                'config': {'z_dim': z_dim, 'encoder_type': encoder_type,
                           'dt': dt, 'seed': seed, 'n1': n1, 'n2': n2}},
               ckpt_path)
    print(f'[spib] saved ckpt → {ckpt_path}')

    # Move model to CPU for eval
    IB = IB.to(torch.device('cpu'))
    IB.eval()

    # Encode full dataset
    with torch.no_grad():
        z_full = []
        for i in range(0, len(X_last), 4096):
            x = torch.tensor(X_last[i:i+4096], dtype=torch.float32)
            _, _, z_mean, _ = IB.forward(x)
            z_full.append(z_mean.cpu().numpy())
    z_full = np.concatenate(z_full)

    # Influence via encoder Jacobian (SPIB has no reconstruction decoder over obs)
    def encoder_fn(x):
        _, _, z_mean, _ = IB.forward(x)
        return z_mean
    x_sample = torch.tensor(X_last[:1024], dtype=torch.float32)
    inf = encoder_jacobian_influence(encoder_fn, x_sample, z_dim)

    # Synthetic-only MCC needs z_true
    z_true = None; z_learned_for_mcc = None
    if dataset_name == 'synthetic_mono':
        raw = np.load(REPO_ROOT / 'data/synthetic_well_v2_mono/raw/trajectories.npz',
                      allow_pickle=True)
        z_true = raw['z_trajs'][0][:5000]
        x_obs = np.asarray(raw['x_trajs'][0][:5000], dtype=np.float32)
        with torch.no_grad():
            _, _, zmu_sub, _ = IB.forward(torch.tensor(x_obs))
        z_learned_for_mcc = zmu_sub.cpu().numpy()

    row = ev.evaluate(method='SPIB', seed=seed, z_full=z_full, influence=inf,
                      z_true=z_true, z_learned_for_mcc=z_learned_for_mcc,
                      notes=f'dt={dt},z={z_dim},beta={beta},K={n_states},refine={refinements}')

    # Append to per-dataset CSV
    csv_path = str(REPO_ROOT / f'experiments/spib/spib_{dataset_name}.csv')
    exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists: w.writeheader()
        w.writerow(row.to_flat())
    print(f'[spib] {dataset_name} seed={seed}: acc={row.regime_acc:.3f}  '
          f'driver_d={row.driver_d:.2f}  → {csv_path}')
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(DATASET_MAP.keys()))
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--dt', type=int, default=10)
    ap.add_argument('--beta', type=float, default=0.01)
    ap.add_argument('--batch_size', type=int, default=2048)
    ap.add_argument('--z_dim', type=int, default=None)
    ap.add_argument('--n_states', type=int, default=2,
                    help='SPIB output_dim K. For K>2, set refinements>0 to let SPIB split labels.')
    ap.add_argument('--refinements', type=int, default=0)
    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = str(REPO_ROOT / f'experiments/spib/{args.dataset}_seed{args.seed}')
    os.chdir(REPO_ROOT)

    train_spib(args.dataset, args.seed, args.out_dir,
               dt=args.dt, z_dim=args.z_dim,
               batch_size=args.batch_size, beta=args.beta,
               n_states=args.n_states, refinements=args.refinements)


if __name__ == '__main__':
    main()
