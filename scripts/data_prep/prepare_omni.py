#!/usr/bin/env python3
"""
70_prepare_omni.py -- Prepare NASA OMNI solar wind dataset for MOSAIC

Hourly OMNI data (2000-2023): 21 variables spanning interplanetary magnetic
field, solar wind plasma, coupling functions, and geomagnetic indices.

Regime definition (geomagnetic storm severity via Dst index):
  Quiet  (0): Dst > -30 nT
  Storm  (1): Dst < -50 nT
  Discard:    -50 <= Dst <= -30  (transition zone)

Ground-truth modules (4 groups from solar-terrestrial physics):
  - IMF:          ABS_B, BX_GSE, BY_GSE, BZ_GSE, BY_GSM, BZ_GSM
  - Plasma:       V, N, T, Pressure
  - Coupling:     E, Beta, Mach
  - Geomagnetic:  AE, AL, AU, SYM_H, SYM_D, ASY_H, ASY_D, DST

Expected drivers: IMF (BZ_GSE southward turning), Plasma (V, Pressure),
  Coupling (E, Beta) -> geomagnetic response
Reference: King & Papitashvili (2005) JGR 110:A02104

Usage:
    python scripts/data_prep/prepare_omni.py
    python scripts/data_prep/prepare_omni.py --raw_dir data/omni/raw --output_dir data/omni/prepared
"""

import numpy as np
import argparse
from pathlib import Path
from collections import Counter

# ------------------------------------------------------------------ #
# Variable catalogue
# ------------------------------------------------------------------ #
# Names in omni_raw.npz have a "1800" suffix from CDAWeb (1-hour resolution).
# We strip that for display.
OMNI_VAR_MAP = {
    'BX_GSE1800':  'BX_GSE',
    'BY_GSE1800':  'BY_GSE',
    'BZ_GSE1800':  'BZ_GSE',
    'BY_GSM1800':  'BY_GSM',
    'BZ_GSM1800':  'BZ_GSM',
    'ABS_B1800':   'ABS_B',
    'V1800':       'V',
    'N1800':       'N',
    'T1800':       'T',
    'Pressure1800':'Pressure',
    'E1800':       'E',
    'Beta1800':    'Beta',
    'Mach1800':    'Mach',
    'AE1800':      'AE',
    'AL1800':      'AL',
    'AU1800':      'AU',
    'SYM_H1800':   'SYM_H',
    'SYM_D1800':   'SYM_D',
    'ASY_H1800':   'ASY_H',
    'ASY_D1800':   'ASY_D',
    'DST1800':     'DST',
}

# Ordered list of display names (canonical order)
OMNI_VARS = [
    'ABS_B', 'BX_GSE', 'BY_GSE', 'BZ_GSE', 'BY_GSM', 'BZ_GSM',  # IMF
    'V', 'N', 'T', 'Pressure',                                      # Plasma
    'E', 'Beta', 'Mach',                                             # Coupling
    'AE', 'AL', 'AU', 'SYM_H', 'SYM_D', 'ASY_H', 'ASY_D', 'DST', # Geomagnetic
]

# Variable-to-module mapping
OMNI_MODULES = {
    'IMF':         ['ABS_B', 'BX_GSE', 'BY_GSE', 'BZ_GSE', 'BY_GSM', 'BZ_GSM'],
    'Plasma':      ['V', 'N', 'T', 'Pressure'],
    'Coupling':    ['E', 'Beta', 'Mach'],
    'Geomagnetic': ['AE', 'AL', 'AU', 'SYM_H', 'SYM_D', 'ASY_H', 'ASY_D', 'DST'],
}

# Inverted: variable -> group
OMNI_GROUPS = {}
for group, members in OMNI_MODULES.items():
    for var in members:
        OMNI_GROUPS[var] = group

# ------------------------------------------------------------------ #
# Fill-value thresholds for OMNI
# ------------------------------------------------------------------ #
# OMNI uses large sentinel values for missing data.
# Per-variable thresholds (conservative); anything above is fill.
FILL_THRESHOLDS = {
    'ABS_B':  999.0,   'BX_GSE': 999.0,  'BY_GSE': 999.0,
    'BZ_GSE': 999.0,   'BY_GSM': 999.0,  'BZ_GSM': 999.0,
    'V':      9999.0,  'N':      999.0,   'T':      999999.0,
    'Pressure': 99.0,
    'E':      999.0,   'Beta':   999.0,   'Mach':   999.0,
    'AE':     9999.0,  'AL':     9999.0,  'AU':     9999.0,
    'SYM_H':  9999.0,  'SYM_D':  9999.0, 'ASY_H':  9999.0,
    'ASY_D':  9999.0,  'DST':    9999.0,
}


def load_omni_npz(raw_dir):
    """Load omni_raw.npz and return (T, D) array + variable names."""
    npz_path = Path(raw_dir) / 'omni_raw.npz'
    if not npz_path.exists():
        raise FileNotFoundError(f"Not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    available_keys = list(data.keys())
    print(f"  NPZ keys ({len(available_keys)}): {available_keys[:10]}...")

    # Build reverse map: display_name -> npz_key
    reverse_map = {v: k for k, v in OMNI_VAR_MAP.items()}

    arrays = []
    found_vars = []
    for var_name in OMNI_VARS:
        npz_key = reverse_map.get(var_name)
        if npz_key and npz_key in data:
            arr = data[npz_key].astype(np.float64).flatten()
            arrays.append(arr)
            found_vars.append(var_name)
        elif var_name in data:
            arr = data[var_name].astype(np.float64).flatten()
            arrays.append(arr)
            found_vars.append(var_name)
        else:
            print(f"  WARNING: variable '{var_name}' not found in NPZ")

    if not arrays:
        raise ValueError("No matching variables found in NPZ")

    # All arrays must have the same length
    lengths = [len(a) for a in arrays]
    if len(set(lengths)) > 1:
        min_len = min(lengths)
        print(f"  WARNING: variable lengths vary {set(lengths)}, truncating to {min_len}")
        arrays = [a[:min_len] for a in arrays]

    matrix = np.column_stack(arrays)  # (T, D)
    print(f"  Loaded {matrix.shape[0]} hours x {matrix.shape[1]} variables from NPZ")
    return matrix, found_vars


def download_omni_fallback(raw_dir):
    """Download OMNI2 all-years ASCII file as fallback.

    Source: https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_all_years.dat

    Returns (T, D) array and variable names.
    """
    import urllib.request
    import io

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_path = raw_dir / 'omni2_all_years.dat'

    url = 'https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_all_years.dat'

    if cache_path.exists():
        print(f"  Using cached file: {cache_path}")
    else:
        print(f"  Downloading from {url} ...")
        try:
            urllib.request.urlretrieve(url, str(cache_path))
            print(f"  Downloaded -> {cache_path}")
        except Exception as e:
            raise RuntimeError(f"Download failed: {e}") from e

    # Parse the OMNI2 fixed-width format.
    # Reference: https://omniweb.gsfc.nasa.gov/html/ow_data.html
    # Column layout (1-indexed, selected columns):
    #   1: Year, 2: Day, 3: Hour
    #   8: |B| (ABS_B)
    #   9: BX_GSE, 10: BY_GSE, 11: BZ_GSE
    #   12: BY_GSM, 13: BZ_GSM
    #   24: V (speed)
    #   25: Vx (not used here)
    #   26: Vy (not used here)
    #   27: Vz (not used here)
    #   28: N (proton density)
    #   29: T (proton temperature)
    #   30: Pressure (flow pressure)
    #   31: E (electric field)
    #   32: Beta (plasma beta)
    #   33: Mach (Alfven Mach)
    #   37: AE, 38: AL, 39: AU
    #   40: SYM/D, 41: SYM/H, 42: ASY/D, 43: ASY/H
    #   40: DST
    # Note: exact column indices depend on the file version.
    # We use a robust approach: load all columns and select by index.

    print(f"  Parsing OMNI2 ASCII file ...")
    all_data = np.loadtxt(str(cache_path), dtype=np.float64)
    print(f"  Raw shape: {all_data.shape}")

    n_cols = all_data.shape[1]
    print(f"  Number of columns: {n_cols}")

    # Column indices (0-based) for the standard OMNI2 55-column format
    # Ref: https://omniweb.gsfc.nasa.gov/html/ow_data.html#1
    col_map = {
        'ABS_B':    7,   # Col 8: Scalar B
        'BX_GSE':   8,   # Col 9
        'BY_GSE':   9,   # Col 10
        'BZ_GSE':  10,   # Col 11
        'BY_GSM':  11,   # Col 12
        'BZ_GSM':  12,   # Col 13
        'V':       23,   # Col 24: Flow speed
        'N':       24,   # Col 25: Proton density -- NOTE: in some versions
        'T':       25,   # Col 26: Proton temperature
        'Pressure':26,   # Col 27: Flow pressure
        'E':       27,   # Col 28: Electric field
        'Beta':    28,   # Col 29: Plasma beta
        'Mach':    29,   # Col 30: Alfven Mach number
        'AE':      32,   # Col 33
        'AL':      33,   # Col 34
        'AU':      34,   # Col 35
        'SYM_D':   36,   # Col 37
        'SYM_H':   37,   # Col 38
        'ASY_D':   38,   # Col 39
        'ASY_H':   39,   # Col 40
        'DST':     40,   # Col 41
    }

    # Filter to years 2000-2023
    year_col = all_data[:, 0]
    mask_year = (year_col >= 2000) & (year_col <= 2023)
    all_data = all_data[mask_year]
    print(f"  After year filter (2000-2023): {all_data.shape[0]} hours")

    arrays = []
    found_vars = []
    for var_name in OMNI_VARS:
        idx = col_map.get(var_name)
        if idx is not None and idx < n_cols:
            arrays.append(all_data[:, idx])
            found_vars.append(var_name)
        else:
            print(f"  WARNING: column for '{var_name}' not available (idx={idx}, ncols={n_cols})")

    matrix = np.column_stack(arrays)
    print(f"  Parsed {matrix.shape[0]} hours x {matrix.shape[1]} variables")
    return matrix, found_vars


def replace_fill_values(data, var_names):
    """Replace OMNI fill values with NaN."""
    n_replaced = 0
    for j, var in enumerate(var_names):
        thresh = FILL_THRESHOLDS.get(var, 9999.0)
        fill_mask = np.abs(data[:, j]) >= thresh
        n_fill = int(fill_mask.sum())
        if n_fill > 0:
            data[fill_mask, j] = np.nan
            n_replaced += n_fill
    pct = 100.0 * n_replaced / data.size
    print(f"  Replaced {n_replaced} fill values ({pct:.2f}% of all entries)")
    return data


def clean_nans(data, max_nan_frac=0.30, max_gap_hours=24):
    """Clean NaN values: drop bad rows, forward/backward fill short gaps.

    1. Drop rows where > max_nan_frac of variables are NaN
    2. Forward-fill then backward-fill remaining NaN (gaps <= max_gap_hours)
    3. Drop any rows still containing NaN (long gaps)
    """
    T, D = data.shape

    # Step 1: drop rows with too many NaN
    nan_frac = np.isnan(data).sum(axis=1) / D
    keep_row = nan_frac <= max_nan_frac
    n_dropped = int((~keep_row).sum())
    data = data[keep_row].copy()
    print(f"  Dropped {n_dropped} rows with >{max_nan_frac*100:.0f}% NaN "
          f"({data.shape[0]} remaining)")

    # Step 2: forward-fill then backward-fill, column by column
    for j in range(D):
        col = data[:, j]
        nan_mask = np.isnan(col)
        if not nan_mask.any():
            continue

        # Identify NaN gap lengths
        # For each NaN position, count how many consecutive NaN precede it
        filled = col.copy()

        # Forward fill
        last_valid = np.nan
        gap_count = 0
        for i in range(len(filled)):
            if np.isnan(filled[i]):
                gap_count += 1
                if not np.isnan(last_valid) and gap_count <= max_gap_hours:
                    filled[i] = last_valid
            else:
                last_valid = filled[i]
                gap_count = 0

        # Backward fill remaining NaN (for leading NaN or gaps not reached by ffill)
        last_valid = np.nan
        gap_count = 0
        for i in range(len(filled) - 1, -1, -1):
            if np.isnan(filled[i]):
                gap_count += 1
                if not np.isnan(last_valid) and gap_count <= max_gap_hours:
                    filled[i] = last_valid
            else:
                last_valid = filled[i]
                gap_count = 0

        data[:, j] = filled

    # Step 3: drop any rows that still have NaN (long gaps)
    still_nan = np.isnan(data).any(axis=1)
    n_still = int(still_nan.sum())
    data = data[~still_nan].copy()
    print(f"  Dropped {n_still} rows with unfillable gaps "
          f"({data.shape[0]} remaining)")

    return data


def assign_regime(data, var_names, dst_quiet=-30, dst_storm=-50):
    """Assign regime labels based on Dst index.

    Quiet (0): Dst > dst_quiet
    Storm (1): Dst < dst_storm
    Transition: dst_storm <= Dst <= dst_quiet -> marked as -1 (discard)
    """
    dst_idx = var_names.index('DST')
    dst = data[:, dst_idx]

    regime = np.full(len(dst), -1, dtype=np.float32)
    regime[dst > dst_quiet] = 0.0    # quiet
    regime[dst < dst_storm] = 1.0    # storm

    n_quiet = int((regime == 0).sum())
    n_storm = int((regime == 1).sum())
    n_trans = int((regime == -1).sum())
    print(f"  Regime assignment: quiet={n_quiet}, storm={n_storm}, "
          f"transition={n_trans} (discarded)")
    return regime


def apply_boundary_buffer(regime, buffer_hours=24):
    """Mask samples near regime transitions to avoid label noise.

    Any sample within buffer_hours of a transition (regime change) is
    set to -1 (discard).
    """
    n = len(regime)
    buffered = regime.copy()

    # Find transition points (where regime changes, ignoring already-discarded)
    for i in range(1, n):
        if regime[i] != regime[i - 1]:
            lo = max(0, i - buffer_hours)
            hi = min(n, i + buffer_hours)
            buffered[lo:hi] = -1.0

    n_before = int((regime >= 0).sum())
    n_after = int((buffered >= 0).sum())
    print(f"  Boundary buffer ({buffer_hours}h): {n_before - n_after} additional "
          f"samples discarded ({n_after} remaining)")
    return buffered


def main(args):
    print("=" * 70)
    print("OMNI Solar Wind Data Preparation -> TDRL format")
    print("=" * 70)

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seq_len = args.lag + 1

    # ----------------------------------------------------------------
    # 1. Load data
    # ----------------------------------------------------------------
    print("\n[1] Loading OMNI data ...")
    try:
        data, var_names = load_omni_npz(raw_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"  NPZ load failed: {e}")
        print("  Falling back to OMNI2 ASCII download ...")
        data, var_names = download_omni_fallback(raw_dir)

    D = len(var_names)
    print(f"  Variables ({D}): {var_names}")

    # ----------------------------------------------------------------
    # 2. Replace fill values with NaN
    # ----------------------------------------------------------------
    print("\n[2] Replacing fill values ...")
    data = replace_fill_values(data, var_names)

    # ----------------------------------------------------------------
    # 3. Clean NaN: drop bad rows, fill short gaps
    # ----------------------------------------------------------------
    print("\n[3] Cleaning NaN values ...")
    data = clean_nans(data, max_nan_frac=0.30, max_gap_hours=24)

    # ----------------------------------------------------------------
    # 4. Assign regime labels (Dst-based)
    # ----------------------------------------------------------------
    print("\n[4] Assigning geomagnetic regime ...")
    regime = assign_regime(data, var_names,
                           dst_quiet=args.dst_quiet,
                           dst_storm=args.dst_storm)

    # ----------------------------------------------------------------
    # 5. Apply boundary buffer near transitions
    # ----------------------------------------------------------------
    print("\n[5] Applying boundary buffer ...")
    regime = apply_boundary_buffer(regime, buffer_hours=args.buffer_hours)

    # ----------------------------------------------------------------
    # 6. Standardize per-variable (z-score over full series)
    # ----------------------------------------------------------------
    print("\n[6] Standardizing ...")
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True) + 1e-8
    data_norm = ((data - mean) / std).astype(np.float32)
    print(f"  Data shape after standardization: {data_norm.shape}")

    # ----------------------------------------------------------------
    # 7. Create sliding windows
    # ----------------------------------------------------------------
    print(f"\n[7] Creating sliding windows (lag={args.lag}, seq_len={seq_len}) ...")
    T = data_norm.shape[0]
    n_windows = T - seq_len + 1

    windows = np.lib.stride_tricks.sliding_window_view(data_norm, seq_len, axis=0)
    # shape: (n_windows, D, seq_len) -> (n_windows, seq_len, D)
    windows = windows.transpose(0, 2, 1).copy().astype(np.float32)

    # Regime label from last frame of each window
    ct = regime[seq_len - 1: seq_len - 1 + n_windows]
    print(f"  Windows: {windows.shape}, ct: {ct.shape}")

    # ----------------------------------------------------------------
    # 8. Discard transition-zone windows
    # ----------------------------------------------------------------
    print("\n[8] Discarding transition-zone windows ...")
    valid = ct >= 0
    windows = windows[valid]
    ct = ct[valid].reshape(-1, 1).astype(np.float32)
    print(f"  After discarding: {windows.shape[0]} windows")

    # ----------------------------------------------------------------
    # 9. Balance classes
    # ----------------------------------------------------------------
    print("\n[9] Balancing classes ...")
    n0 = int((ct == 0).sum())
    n1 = int((ct == 1).sum())
    print(f"  Before balancing: quiet={n0}, storm={n1}")

    n_min = min(n0, n1)
    rng = np.random.RandomState(42)
    idx0 = rng.choice(np.where(ct.squeeze() == 0)[0], n_min, replace=False)
    idx1 = rng.choice(np.where(ct.squeeze() == 1)[0], n_min, replace=n1 < n_min)
    idx = rng.permutation(np.concatenate([idx0, idx1]))

    xt = windows[idx]
    ct = ct[idx]
    yt = xt.copy()

    # ----------------------------------------------------------------
    # 10. Diagnostics
    # ----------------------------------------------------------------
    print("\n[10] Diagnostics ...")
    corr_vals = []
    for c in range(xt.shape[-1]):
        r = np.corrcoef(xt[:, 0, c], xt[:, 1, c])[0, 1]
        if not np.isnan(r):
            corr_vals.append(r)
    corr = np.mean(corr_vals) if corr_vals else 0.0
    print(f"  corr(frame0, frame1): {corr:.3f}")

    var_names_arr = np.array(var_names)
    var_groups_arr = np.array([OMNI_GROUPS[v] for v in var_names])

    print(f"\n  Final: xt={xt.shape}, N/D={xt.shape[0] / D:.0f}")
    print(f"  regime 0 (quiet) = {int((ct == 0).sum())}")
    print(f"  regime 1 (storm) = {int((ct == 1).sum())}")
    print(f"  Groups: {dict(Counter(var_groups_arr))}")

    # ----------------------------------------------------------------
    # 11. Save
    # ----------------------------------------------------------------
    print("\n[11] Saving ...")
    regime_desc = (f"Quiet (Dst>{args.dst_quiet}) vs "
                   f"Storm (Dst<{args.dst_storm})")

    np.savez_compressed(output_dir / 'data.npz', xt=xt, yt=yt, ct=ct)
    np.savez_compressed(output_dir / 'metadata.npz',
                        input_dim=D,
                        n_variables=D,
                        variable_names=var_names_arr,
                        variable_groups=var_groups_arr,
                        lag=args.lag,
                        seq_len=seq_len,
                        dataset_name='omni',
                        regime_description=regime_desc)

    print(f"\n[SAVED] {output_dir / 'data.npz'}")
    print(f"[SAVED] {output_dir / 'metadata.npz'}")
    print("=" * 70)
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare NASA OMNI solar wind dataset for MOSAIC')
    parser.add_argument('--raw_dir', default='data/omni/raw',
                        help='Directory containing omni_raw.npz (or download target)')
    parser.add_argument('--output_dir', default='data/omni/prepared',
                        help='Output directory for data.npz and metadata.npz')
    parser.add_argument('--lag', type=int, default=2,
                        help='Number of lag frames (seq_len = lag + 1, default: 2)')
    parser.add_argument('--dst_quiet', type=float, default=-30,
                        help='Dst threshold for quiet regime (default: -30)')
    parser.add_argument('--dst_storm', type=float, default=-50,
                        help='Dst threshold for storm regime (default: -50)')
    parser.add_argument('--buffer_hours', type=int, default=24,
                        help='Boundary buffer near regime transitions (default: 24)')
    args = parser.parse_args()
    main(args)
