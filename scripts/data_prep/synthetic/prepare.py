#!/usr/bin/env python3
"""
prepare_tdrl_format.py — x well-occupancy regime labeling for synthetic v2.

Regime 0 = State A (x < -threshold, particle in left well)
Regime 1 = State B (x > +threshold, particle in right well)
Intermediate frames (|x| <= threshold) are discarded.

Usage:
    python scripts/data_prep/synthetic/prepare.py
    python scripts/data_prep/synthetic/prepare.py --interaction_alpha 0.5
"""

import argparse
import numpy as np
from pathlib import Path
from config import *

PROJ = Path(__file__).resolve().parent.parent.parent.parent


def apply_boundary_buffer(states, buffer=BOUNDARY_BUFFER):
    """Set frames within `buffer` of a state boundary to -1 (discard)."""
    buffered = states.copy()
    n = len(states)
    for t in range(1, n):
        if states[t] != states[t - 1] and states[t] >= 0 and states[t - 1] >= 0:
            lo = max(0, t - buffer)
            hi = min(n, t + buffer)
            buffered[lo:hi] = -1
    return buffered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interaction_alpha", type=float, default=INTERACTION_ALPHA)
    args = parser.parse_args()
    alpha = args.interaction_alpha

    if alpha > 0:
        suffix = f"_alpha{alpha:.1f}".replace(".", "p")
        raw_dir = PROJ / f"data/synthetic_well_v2{suffix}/raw"
        src_dir = PROJ / f"data/synthetic_well_v2{suffix}/source"
    else:
        raw_dir = PROJ / "data/synthetic_well_v2/raw"
        src_dir = PROJ / "data/synthetic_well_v2/source"
    src_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Preparing x Well-Occupancy Regime Labels (Synthetic v2)")
    if alpha > 0:
        print(f"  Interaction alpha = {alpha:.2f}")
    print("=" * 70)

    raw = np.load(raw_dir / "trajectories.npz", allow_pickle=True)
    x_trajs = raw['x_trajs']
    state_labels = raw['state_labels']

    seq_len = LAG + 1
    all_windows_0, all_windows_1 = [], []

    for idx, traj_cfg in enumerate(TRAJECTORIES):
        x_traj = x_trajs[idx]
        states = state_labels[idx]

        states_buf = apply_boundary_buffer(states)

        n_frames = len(x_traj)
        n_r0, n_r1 = 0, 0

        for t in range(seq_len - 1, n_frames):
            s = states_buf[t]
            if s < 0:
                continue
            window_states = states_buf[t - seq_len + 1: t + 1]
            if not np.all(window_states == s):
                continue

            window = x_traj[t - seq_len + 1: t + 1]
            if s == 0:
                all_windows_0.append(window)
                n_r0 += 1
            else:
                all_windows_1.append(window)
                n_r1 += 1

        print(f"  {traj_cfg['name']}: regime_0(A)={n_r0}, regime_1(B)={n_r1}, "
              f"discarded={n_frames - n_r0 - n_r1}")

    all_windows_0 = np.array(all_windows_0, dtype=np.float32)
    all_windows_1 = np.array(all_windows_1, dtype=np.float32)

    print(f"\nTotal: regime_0={len(all_windows_0)}, regime_1={len(all_windows_1)}")

    # Balance
    rng = np.random.RandomState(SEED)
    n_min = min(len(all_windows_0), len(all_windows_1))
    n_target = int(n_min * SUBSAMPLE_RATIO)

    if len(all_windows_0) > n_target:
        idx = rng.choice(len(all_windows_0), size=n_target, replace=False)
        all_windows_0 = all_windows_0[idx]
    if len(all_windows_1) > n_target:
        idx = rng.choice(len(all_windows_1), size=n_target, replace=False)
        all_windows_1 = all_windows_1[idx]

    print(f"After balancing: regime_0={len(all_windows_0)}, regime_1={len(all_windows_1)}")

    # Combine
    xt = np.concatenate([all_windows_0, all_windows_1], axis=0)
    ct = np.concatenate([
        np.zeros((len(all_windows_0), 1)),
        np.ones((len(all_windows_1), 1)),
    ], axis=0).astype(np.float32)

    # Shuffle
    perm = rng.permutation(len(xt))
    xt = xt[perm]
    ct = ct[perm]
    yt = xt.copy()

    input_dim = xt.shape[-1]

    print(f"\nFinal dataset:")
    print(f"  xt shape: {xt.shape}")
    print(f"  ct shape: {ct.shape}")
    print(f"  Input dim: {input_dim}")
    print(f"  Regime 0 (A): {int((ct == 0).sum())}")
    print(f"  Regime 1 (B): {int((ct == 1).sum())}")

    np.savez_compressed(src_dir / "data.npz", xt=xt, yt=yt, ct=ct)
    np.savez_compressed(src_dir / "metadata.npz",
                        input_dim=input_dim, obs_dim=OBS_DIM,
                        lag=LAG, n_latent=Z_DIM)
    print(f"\n[SAVED] {src_dir / 'data.npz'}")
    print(f"[SAVED] {src_dir / 'metadata.npz'}")


if __name__ == "__main__":
    main()
