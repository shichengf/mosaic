#!/usr/bin/env python3
"""
71_prepare_tep.py -- Prepare Tennessee Eastman Process (TEP) dataset for MOSAIC

The TEP is a realistic chemical process simulation with 52 measured variables
(41 process measurements + 11 manipulated variables) and 21 pre-programmed
fault scenarios.  This script loads normal operation data and selected fault
files, assigns regime labels, standardises, creates sliding windows, balances
classes, and saves in the standard TDRL format.

Input layout (Braatz / Downs & Vogel format):
  d00.dat      Normal training run.  Shape on disk: (52, 500) -- TRANSPOSED.
               Each row is a variable, each column is a timestep.  Must be
               transposed to (500, 52) before use.
  d00_te.dat   Normal test run.  Shape: (960, 52).
  dXX.dat      Fault XX training run.  Shape: (480, 52).  Fault present from
               the first timestep (no pre-fault window).
  dXX_te.dat   Fault XX test run.  Shape: (960, 52).  First 160 timesteps are
               normal; fault introduced at step 161.

Variables (columns 0-51):
  Columns 0-21:   XMEAS 1-22   (continuous process measurements)
  Columns 22-40:  XMEAS 23-41  (composition measurements, sampled less often)
  Columns 41-51:  XMV 1-11     (manipulated variables)

Ground-truth modules (from Downs & Vogel 1993 process description):
  Reactor:      XMEAS 6-9,21 + XMEAS 23-28 + XMV 3,4,10
  Condenser:    XMEAS 22 + XMV 11
  Separator:    XMEAS 10-14 + XMV 6,7
  Stripper:     XMEAS 4,15-19 + XMEAS 29-36 + XMV 8,9
  Compressor:   XMEAS 20 + XMV 5
  Feed_Global:  XMEAS 1-3,5 + XMV 1,2

Regime: 0 = normal operation, 1 = faulty operation.

Usage:
    python scripts/data_prep/prepare_tep.py
    python scripts/data_prep/prepare_tep.py --raw_dir data/tep/raw --faults 1 2 4 5
    python scripts/data_prep/prepare_tep.py --lag 3 --fault_onset_test 160
"""

import numpy as np
import argparse
from pathlib import Path
from collections import Counter


# ---------------------------------------------------------------------------
# Variable names: XMEAS 1..41, XMV 1..11 (52 total, 0-indexed)
# ---------------------------------------------------------------------------
VARIABLE_NAMES = (
    [f"XMEAS_{i}" for i in range(1, 42)]
    + [f"XMV_{i}" for i in range(1, 12)]
)

# ---------------------------------------------------------------------------
# Ground-truth module assignment (0-based column indices)
#   Columns 0-21  = XMEAS 1-22
#   Columns 22-40 = XMEAS 23-41
#   Columns 41-51 = XMV 1-11
# ---------------------------------------------------------------------------
TEP_MODULES = {
    "Reactor": [5, 6, 7, 8, 20, 22, 23, 24, 25, 26, 27, 43, 44, 50],
    "Condenser": [21, 51],
    "Separator": [9, 10, 11, 12, 13, 46, 47],
    "Stripper": [3, 14, 15, 16, 17, 18, 28, 29, 30, 31, 32, 33, 34, 35, 48, 49],
    "Compressor": [19, 45],
    "Feed_Global": [0, 1, 2, 4, 41, 42],
}


def _col_to_group():
    """Return an array of length 52 mapping each column index to its group name."""
    groups = ["Unassigned"] * 52
    for group_name, indices in TEP_MODULES.items():
        for idx in indices:
            groups[idx] = group_name
    return groups


def load_dat(path):
    """Load a TEP .dat file, auto-detecting the transposed d00.dat format.

    d00.dat on disk is (52, T) -- rows are variables, columns are timesteps.
    All other files are (T, 52) -- rows are timesteps, columns are variables.
    This function always returns (T, 52).
    """
    data = np.loadtxt(path)

    if data.shape[0] == 52 and data.shape[1] != 52:
        # Transposed format (d00.dat): (52, T) -> (T, 52)
        data = data.T
        transposed = True
    elif data.shape[1] == 52:
        transposed = False
    else:
        raise ValueError(
            f"Unexpected shape {data.shape} in {path}. "
            f"Expected 52 columns (or 52 rows for d00.dat transposed format)."
        )

    print(f"  {path.name:12s}: {data.shape}"
          f"{'  (transposed from disk)' if transposed else ''}")
    return data.astype(np.float64)


def make_sliding_windows(data, regime, seq_len):
    """Create sliding windows (stride=1) within a single continuous segment.

    Parameters
    ----------
    data : ndarray, shape (T, D)
    regime : ndarray, shape (T,)
    seq_len : int

    Returns
    -------
    windows : ndarray, shape (N, seq_len, D)
    labels  : ndarray, shape (N,) -- regime label of the last frame
    """
    T, D = data.shape
    n_windows = T - seq_len + 1
    if n_windows <= 0:
        return (np.empty((0, seq_len, D), dtype=np.float32),
                np.empty((0,), dtype=np.float32))

    windows = np.lib.stride_tricks.sliding_window_view(data, seq_len, axis=0)
    # shape: (n_windows, D, seq_len) -> (n_windows, seq_len, D)
    windows = windows.transpose(0, 2, 1).copy().astype(np.float32)

    labels = regime[seq_len - 1: seq_len - 1 + n_windows].astype(np.float32)
    return windows, labels


def main(args):
    print("=" * 70)
    print("TEP Data Preparation: Tennessee Eastman Process -> TDRL format")
    print("=" * 70)

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seq_len = args.lag + 1
    fault_onset_test = args.fault_onset_test   # step index where fault begins in test

    # ------------------------------------------------------------------
    # 1. Load normal operation data
    # ------------------------------------------------------------------
    print("\nLoading normal operation data ...")
    d00_path = raw_dir / "d00.dat"
    d00_te_path = raw_dir / "d00_te.dat"

    segments = []   # list of (data_array, regime_array)

    if d00_path.exists():
        d00 = load_dat(d00_path)
        segments.append((d00, np.zeros(d00.shape[0], dtype=np.float32)))
    else:
        print(f"  WARNING: {d00_path} not found, skipping normal training")

    if d00_te_path.exists():
        d00_te = load_dat(d00_te_path)
        segments.append((d00_te, np.zeros(d00_te.shape[0], dtype=np.float32)))
    else:
        print(f"  WARNING: {d00_te_path} not found, skipping normal test")

    # ------------------------------------------------------------------
    # 2. Load fault data
    # ------------------------------------------------------------------
    faults = args.faults
    print(f"\nLoading fault data for faults: {faults} ...")

    for fault_id in faults:
        tag = f"d{fault_id:02d}"

        # --- Training file (fault present from timestep 0) ---
        train_path = raw_dir / f"{tag}.dat"
        if train_path.exists():
            d_train = load_dat(train_path)
            regime_train = np.ones(d_train.shape[0], dtype=np.float32)
            segments.append((d_train, regime_train))
        else:
            print(f"  WARNING: {train_path} not found, skipping")

        # --- Test file (first `fault_onset_test` steps normal, rest faulty) ---
        test_path = raw_dir / f"{tag}_te.dat"
        if test_path.exists():
            d_test = load_dat(test_path)
            T_test = d_test.shape[0]
            regime_test = np.zeros(T_test, dtype=np.float32)
            regime_test[fault_onset_test:] = 1.0
            segments.append((d_test, regime_test))
        else:
            print(f"  WARNING: {test_path} not found, skipping")

    if not segments:
        raise FileNotFoundError(
            f"No .dat files found in {raw_dir}. Check the --raw_dir path."
        )

    # ------------------------------------------------------------------
    # 3. Standardise per-variable (fit on all data combined)
    # ------------------------------------------------------------------
    print("\nStandardising per-variable ...")
    all_data = np.concatenate([s[0] for s in segments], axis=0)
    mean = all_data.mean(axis=0, keepdims=True)
    std = all_data.std(axis=0, keepdims=True) + 1e-8

    for i in range(len(segments)):
        data_i, regime_i = segments[i]
        data_i = ((data_i - mean) / std).astype(np.float32)
        data_i = np.clip(data_i, -5, 5)
        segments[i] = (data_i, regime_i)

    D = all_data.shape[1]
    print(f"  Variables: {D}, Total timesteps: {all_data.shape[0]}")

    # ------------------------------------------------------------------
    # 4. Create sliding windows per segment (no cross-segment leakage)
    # ------------------------------------------------------------------
    print(f"\nCreating sliding windows (lag={args.lag}, seq_len={seq_len}) ...")
    all_windows = []
    all_labels = []

    for data_i, regime_i in segments:
        win, lab = make_sliding_windows(data_i, regime_i, seq_len)
        if win.shape[0] > 0:
            all_windows.append(win)
            all_labels.append(lab)

    xt = np.concatenate(all_windows, axis=0)
    ct = np.concatenate(all_labels, axis=0).reshape(-1, 1)

    print(f"  Total windows: {xt.shape}")
    print(f"  Regime distribution: {dict(Counter(ct.squeeze().astype(int)))}")

    # ------------------------------------------------------------------
    # 5. Balance classes (1:1 subsample, seed=42)
    # ------------------------------------------------------------------
    n0 = int((ct == 0).sum())
    n1 = int((ct == 1).sum())
    print(f"\nBefore balancing: normal={n0}, faulty={n1}")

    n_min = min(n0, n1)
    rng = np.random.RandomState(42)
    idx0 = rng.choice(np.where(ct.squeeze() == 0)[0], n_min, replace=(n0 < n_min))
    idx1 = rng.choice(np.where(ct.squeeze() == 1)[0], n_min, replace=(n1 < n_min))
    idx = rng.permutation(np.concatenate([idx0, idx1]))

    xt = xt[idx]
    ct = ct[idx]
    yt = xt.copy()

    print(f"After balancing: normal={int((ct == 0).sum())}, faulty={int((ct == 1).sum())}")

    # ------------------------------------------------------------------
    # 6. Diagnostic checks
    # ------------------------------------------------------------------
    corr_vals = []
    for c in range(D):
        r = np.corrcoef(xt[:, 0, c], xt[:, 1, c])[0, 1]
        if not np.isnan(r):
            corr_vals.append(r)
    corr = np.mean(corr_vals) if corr_vals else float("nan")
    print(f"corr(frame0, frame1): {corr:.3f}")

    # ------------------------------------------------------------------
    # 7. Variable names and ground-truth groups
    # ------------------------------------------------------------------
    var_names = np.array(VARIABLE_NAMES[:D])
    var_groups_list = _col_to_group()[:D]
    var_groups = np.array(var_groups_list)

    print(f"\nFinal: xt={xt.shape}, N/D={xt.shape[0] / D:.0f}")
    print(f"  regime 0={int((ct == 0).sum())}, regime 1={int((ct == 1).sum())}")
    print(f"  Groups: {dict(Counter(var_groups))}")

    # Verify that all columns are assigned
    n_unassigned = int((var_groups == "Unassigned").sum())
    if n_unassigned:
        unassigned_cols = [i for i, g in enumerate(var_groups_list) if g == "Unassigned"]
        print(f"  WARNING: {n_unassigned} columns have no group assignment: {unassigned_cols}")

    # ------------------------------------------------------------------
    # 8. Save
    # ------------------------------------------------------------------
    np.savez_compressed(
        output_dir / "data.npz",
        xt=xt, yt=yt, ct=ct,
    )
    np.savez_compressed(
        output_dir / "metadata.npz",
        input_dim=D,
        n_variables=len(var_names),
        variable_names=var_names,
        variable_groups=var_groups,
        lag=args.lag,
        seq_len=seq_len,
        dataset_name="TEP",
        fault_ids=np.array(faults),
        fault_onset_test=fault_onset_test,
        regime_description="Normal operation (0) vs Faulty operation (1)",
    )

    print(f"\n[SAVED] {output_dir / 'data.npz'}")
    print(f"[SAVED] {output_dir / 'metadata.npz'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare Tennessee Eastman Process dataset for MOSAIC")
    parser.add_argument(
        "--raw_dir", default="data/tep/raw",
        help="Directory containing dXX.dat and dXX_te.dat files")
    parser.add_argument(
        "--output_dir", default="data/tep/prepared",
        help="Output directory for data.npz and metadata.npz")
    parser.add_argument(
        "--lag", type=int, default=2,
        help="Lag order (seq_len = lag + 1, default: 2)")
    parser.add_argument(
        "--faults", type=int, nargs="+", default=[1, 2, 4, 5],
        help="Fault IDs to include (default: 1 2 4 5)")
    parser.add_argument(
        "--fault_onset_test", type=int, default=160,
        help="Timestep at which fault begins in test files (default: 160)")
    args = parser.parse_args()
    main(args)
