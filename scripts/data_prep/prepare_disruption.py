#!/usr/bin/env python3
"""
66_prepare_disruption.py — Prepare tokamak disruption data for MOSAIC

Synthetic data with 12 plasma diagnostic variables and 4 ground-truth modules.
Regime 0 = healthy plasma, Regime 1 = pre-disruptive.

Ground-truth modules (from plasma physics):
  - MHD: q95, li, locked_mode
  - Density: n_bar, greenwald_frac
  - Energy: w_mhd, beta_n, p_rad_frac
  - Shape: ip, elongation, triangularity, delta

Expected drivers: MHD (locked mode growth), Density (density limit)

Usage:
    python scripts/data_prep/prepare_disruption.py
    python scripts/data_prep/prepare_disruption.py --output_dir data/disruption/prepared
"""

import numpy as np
import argparse
from pathlib import Path

TOKAMAK_VARS = [
    'q95', 'li', 'locked_mode',
    'n_bar', 'greenwald_frac',
    'w_mhd', 'beta_n', 'p_rad_frac',
    'ip', 'elongation', 'triangularity', 'delta',
]

TOKAMAK_GROUPS = {
    'q95': 'MHD', 'li': 'MHD', 'locked_mode': 'MHD',
    'n_bar': 'Density', 'greenwald_frac': 'Density',
    'w_mhd': 'Energy', 'beta_n': 'Energy', 'p_rad_frac': 'Energy',
    'ip': 'Shape', 'elongation': 'Shape', 'triangularity': 'Shape', 'delta': 'Shape',
}


def generate_synthetic_disruption(n_per_class=3000, seq_len=3, shot_len=200, seed=42):
    """Generate synthetic tokamak diagnostic data with known module structure.

    Physics-motivated:
    - 4 latent sources (MHD instability, density, energy confinement, shape)
    - Sparse mixing to 12 observable diagnostics
    - AR(1) temporal dynamics within each shot
    - Regime 0: stable plasma (all sources near baseline)
    - Regime 1: pre-disruptive (MHD ramps, density exceeds limit)
    """
    rng = np.random.RandomState(seed)
    n_sources = 4
    n_vars = 12

    # Sparse mixing matrix: each source drives its module's variables
    mixing = np.zeros((n_sources, n_vars), dtype=np.float32)
    mixing[0, 0:3] = rng.uniform(0.7, 1.3, 3)    # MHD → q95, li, locked_mode
    mixing[1, 3:5] = rng.uniform(0.8, 1.5, 2)    # Density → n_bar, greenwald_frac
    mixing[2, 5:8] = rng.uniform(0.7, 1.3, 3)    # Energy → w_mhd, beta_n, p_rad_frac
    mixing[3, 8:12] = rng.uniform(0.6, 1.0, 4)   # Shape → ip, elongation, triangularity, delta

    all_windows = []
    all_labels = []

    for label in [0, 1]:
        for _ in range(n_per_class):
            # AR(1) latent sources
            sources = np.zeros((shot_len, n_sources), dtype=np.float32)
            rho = 0.95  # temporal autocorrelation
            sources[0] = rng.randn(n_sources) * 0.5

            for t in range(1, shot_len):
                innovation = rng.randn(n_sources) * 0.3
                sources[t] = rho * sources[t - 1] + innovation

                if label == 1:
                    # Pre-disruptive: MHD ramps up, density exceeds limit
                    progress = t / shot_len
                    if progress > 0.5:
                        ramp = (progress - 0.5) * 4.0  # 0→2 over second half
                        sources[t, 0] += ramp * 1.5   # MHD instability grows
                        sources[t, 1] += ramp * 0.8   # Density limit approach

            # Observe through mixing matrix + noise
            observations = sources @ mixing + 0.15 * rng.randn(shot_len, n_vars)

            # Create sliding windows within this shot
            if label == 0:
                # Healthy: use entire shot
                for i in range(shot_len - seq_len + 1):
                    all_windows.append(observations[i:i + seq_len])
                    all_labels.append(0)
            else:
                # Disruptive: only use windows where signal is active (2nd half)
                start_idx = shot_len // 2
                for i in range(start_idx, shot_len - seq_len + 1):
                    all_windows.append(observations[i:i + seq_len])
                    all_labels.append(1)

    return np.array(all_windows, dtype=np.float32), np.array(all_labels, dtype=np.float32)


def main(args):
    print("=" * 70)
    print("Tokamak Disruption Data Preparation -> TDRL format")
    print("=" * 70)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seq_len = args.lag + 1

    # Check for real data first
    raw_dir = Path(args.raw_dir)
    real_data_found = False
    if raw_dir.exists():
        h5_files = list(raw_dir.rglob('*.h5')) + list(raw_dir.rglob('*.hdf5'))
        npz_files = list(raw_dir.rglob('*.npz'))
        if h5_files or npz_files:
            print(f"Found real data files: {len(h5_files)} HDF5, {len(npz_files)} NPZ")
            real_data_found = True
            # TODO: implement real data loading if files become available

    if not real_data_found:
        print("No real data found. Generating synthetic tokamak data...")
        print(f"  12 diagnostic variables, 4 modules (MHD, Density, Energy, Shape)")
        print(f"  {args.n_per_class} shots per class, shot_len={args.shot_len}, lag={args.lag}")

        xt, ct = generate_synthetic_disruption(
            n_per_class=args.n_per_class,
            seq_len=seq_len,
            shot_len=args.shot_len,
            seed=42,
        )

    ct = ct.reshape(-1, 1)

    # Standardize per-feature
    shape = xt.shape
    xt_flat = xt.reshape(-1, shape[-1])
    mean = xt_flat.mean(axis=0, keepdims=True)
    std = xt_flat.std(axis=0, keepdims=True) + 1e-8
    xt = ((xt.reshape(-1, shape[-1]) - mean) / std).reshape(shape).astype(np.float32)

    # Balance
    n0 = int((ct == 0).sum())
    n1 = int((ct == 1).sum())
    print(f"Before balancing: healthy={n0}, disruptive={n1}")

    n_min = min(n0, n1)
    rng = np.random.RandomState(42)
    idx0 = rng.choice(np.where(ct.squeeze() == 0)[0], n_min, replace=False)
    idx1 = rng.choice(np.where(ct.squeeze() == 1)[0], n_min, replace=False)
    idx = rng.permutation(np.concatenate([idx0, idx1]))
    xt = xt[idx]
    ct = ct[idx]
    yt = xt.copy()

    # Verify temporal structure
    corr = np.mean([np.corrcoef(xt[:, 0, c], xt[:, 1, c])[0, 1] for c in range(xt.shape[-1])])
    print(f"corr(frame0, frame1): {corr:.3f}")

    var_names = np.array(TOKAMAK_VARS)
    var_groups = np.array([TOKAMAK_GROUPS[v] for v in TOKAMAK_VARS])

    print(f"Final: xt={xt.shape}, N/D={xt.shape[0] / xt.shape[-1]:.0f}")
    print(f"  regime 0={int((ct == 0).sum())}, regime 1={int((ct == 1).sum())}")

    np.savez_compressed(output_dir / 'data.npz', xt=xt, yt=yt, ct=ct)
    np.savez_compressed(output_dir / 'metadata.npz',
                        input_dim=xt.shape[-1],
                        n_variables=len(var_names),
                        variable_names=var_names,
                        variable_groups=var_groups,
                        lag=args.lag,
                        seq_len=seq_len,
                        dataset_name='SyntheticTokamak',
                        regime_description='healthy vs pre-disruptive plasma (synthetic)')
    print(f"[SAVED] {output_dir / 'data.npz'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', default='data/disruption/raw')
    parser.add_argument('--output_dir', default='data/disruption/prepared')
    parser.add_argument('--lag', type=int, default=2)
    parser.add_argument('--n_per_class', type=int, default=3000)
    parser.add_argument('--shot_len', type=int, default=200)
    args = parser.parse_args()
    main(args)
