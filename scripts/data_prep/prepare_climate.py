#!/usr/bin/env python3
"""
60_prepare_climate.py — Prepare NOAA ERSSTv5 tropical Pacific SST for MOSAIC

Extracts sea surface temperature anomalies from the tropical Pacific,
pairs with ENSO regime labels from the ONI index, and outputs data.npz.

Observed variables: ~1500-2000 ocean grid cells (after masking land)
Regime: El Nino (ONI >= 0.5) vs Non-El Nino
Ground truth modules: Nino 1+2, Nino 3, Nino 3.4, Nino 4, Indian Ocean, etc.

Usage:
    python scripts/data_prep/prepare_climate.py
    python scripts/data_prep/prepare_climate.py --sst_path data/climate/raw/sst.mnmean.nc --oni_path data/climate/raw/oni_raw.txt --output_dir data/climate/prepared
"""

import numpy as np
import argparse
from pathlib import Path


def load_sst(sst_path):
    """Load ERSSTv5 NetCDF, subset to tropical Pacific."""
    import xarray as xr

    ds = xr.open_dataset(sst_path)

    # Subset: tropical Pacific + Indian Ocean
    # Latitude: 30S to 30N
    # Longitude: 40E to 80W (= 40 to 280 in 0-360 coords)
    sst = ds['sst'].sel(
        lat=slice(30, -30),
        lon=slice(40, 280)
    )

    # Select modern era with good data coverage
    sst = sst.sel(time=slice('1950-01', '2023-12'))

    # Get coordinate arrays for metadata
    lats = sst.lat.values
    lons = sst.lon.values
    times = sst.time.values

    # Flatten spatial dims: (T, lat, lon) -> (T, N)
    T = len(times)
    sst_vals = sst.values.reshape(T, -1)

    # Create ocean mask (non-NaN in first timestep)
    ocean_mask = ~np.isnan(sst_vals[0])
    sst_ocean = sst_vals[:, ocean_mask]  # (T, N_ocean)

    # Fill any remaining NaN with 0 (rare, at ice edges)
    sst_ocean = np.nan_to_num(sst_ocean, nan=0.0)

    # Standardize per grid cell
    mean = sst_ocean.mean(axis=0, keepdims=True)
    std = sst_ocean.std(axis=0, keepdims=True) + 1e-8
    sst_norm = ((sst_ocean - mean) / std).astype(np.float32)

    # Build variable names and ground-truth groups
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    lat_flat = lat_grid.reshape(-1)[ocean_mask]
    lon_flat = lon_grid.reshape(-1)[ocean_mask]

    var_names = [f"SST_{lat:.1f}_{lon:.1f}" for lat, lon in zip(lat_flat, lon_flat)]

    # Assign ground-truth Nino region groups
    groups = []
    for lat, lon in zip(lat_flat, lon_flat):
        if -10 <= lat <= 0 and 270 <= lon <= 280:
            groups.append('Nino1+2')
        elif -5 <= lat <= 5 and 210 <= lon <= 270:
            groups.append('Nino3')
        elif -5 <= lat <= 5 and 190 <= lon <= 240:
            groups.append('Nino3.4')
        elif -5 <= lat <= 5 and 160 <= lon <= 210:
            groups.append('Nino4')
        elif -20 <= lat <= 20 and 40 <= lon <= 100:
            groups.append('IndianOcean')
        elif -30 <= lat <= -10 or 10 <= lat <= 30:
            groups.append('Subtropical')
        else:
            groups.append('TropicalOther')

    print(f"SST data: {T} months, {sst_norm.shape[1]} ocean grid cells")
    for g in sorted(set(groups)):
        print(f"  {g}: {groups.count(g)} cells")

    return sst_norm, times, np.array(var_names), np.array(groups)


def load_oni(oni_path, times):
    """Load ONI index and create regime labels aligned with SST times."""
    import pandas as pd

    # Try different ONI file formats
    try:
        # Format: columns separated by whitespace
        # Year, Jan, Feb, ..., Dec
        oni_df = pd.read_csv(oni_path, sep=r'\s+', header=0)

        # Melt to long format
        oni_long = oni_df.melt(id_vars=['YR'], var_name='month', value_name='oni')
        month_map = {'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,
                     'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12}
        oni_long['month_num'] = oni_long['month'].map(month_map)
        oni_long['date'] = pd.to_datetime(
            oni_long['YR'].astype(str) + '-' + oni_long['month_num'].astype(str) + '-01'
        )
        oni_series = oni_long.set_index('date')['oni'].sort_index()
    except Exception as e:
        print(f"ONI parsing failed ({e}), using simple threshold on Nino3.4 SST")
        return None

    # Align with SST times
    sst_dates = pd.DatetimeIndex(times)
    oni_aligned = []
    for d in sst_dates:
        # Find closest ONI month
        key = pd.Timestamp(year=d.year, month=d.month, day=1)
        if key in oni_series.index:
            oni_aligned.append(float(oni_series.loc[key]))
        else:
            oni_aligned.append(0.0)

    oni_array = np.array(oni_aligned)

    # Regime: El Nino (ONI >= 0.5) = 1, else = 0
    regime = (oni_array >= 0.5).astype(np.float32)

    n_nino = int(regime.sum())
    n_other = len(regime) - n_nino
    print(f"ONI regimes: El Nino={n_nino} months, Non-El Nino={n_other} months")

    return regime


def create_windows(data, regime, lag=2):
    """Create sliding windows with regime label from last frame."""
    seq_len = lag + 1
    T, D = data.shape
    n_windows = T - seq_len + 1

    # Sliding window
    windows = np.lib.stride_tricks.sliding_window_view(data, seq_len, axis=0)
    # windows shape: (n_windows, D, seq_len) -> transpose to (n_windows, seq_len, D)
    windows = windows.transpose(0, 2, 1).copy()

    # Regime label from last frame of each window
    ct = regime[seq_len - 1: seq_len - 1 + n_windows]

    return windows.astype(np.float32), ct.reshape(-1, 1).astype(np.float32)


def main(args):
    print("=" * 70)
    print("Climate Data Preparation: ERSSTv5 -> TDRL format")
    print("=" * 70)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load SST
    sst_norm, times, var_names, groups = load_sst(args.sst_path)

    # Load ONI regime labels
    regime = load_oni(args.oni_path, times)

    if regime is None:
        # Fallback: use Nino 3.4 region mean SST as proxy
        nino34_mask = groups == 'Nino3.4'
        nino34_mean = sst_norm[:, nino34_mask].mean(axis=1)
        regime = (nino34_mean > 0.5).astype(np.float32)
        print(f"Fallback regime from Nino 3.4 mean: {int(regime.sum())} El Nino months")

    # Variable screening (when input dim >> sample count)
    if args.screen:
        from tdrl.variable_screening import screen_variables
        print(f"\n--- Variable Screening ---")
        screen_mask, screen_info = screen_variables(
            sst_norm, regime,
            d_threshold=args.screen_d,
            r_threshold=args.screen_r,
            target_dim=args.screen_target_dim,
            variable_names=var_names,
            variable_groups=groups,
        )
        sst_norm = sst_norm[:, screen_mask]
        var_names = var_names[screen_mask]
        groups = groups[screen_mask]
        print(f"Post-screening: {sst_norm.shape[1]} variables, {sst_norm.shape[0]} timepoints")

    # Create sliding windows
    lag = args.lag
    xt, ct = create_windows(sst_norm, regime, lag=lag)
    yt = xt.copy()

    # Balance classes (subsample majority)
    n_regime1 = int((ct == 1).sum())
    n_regime0 = int((ct == 0).sum())
    print(f"\nBefore balancing: regime0={n_regime0}, regime1={n_regime1}")

    if args.balance:
        n_min = min(n_regime0, n_regime1)
        rng = np.random.RandomState(42)
        idx0 = rng.choice(np.where(ct.squeeze() == 0)[0], n_min, replace=False)
        idx1 = rng.choice(np.where(ct.squeeze() == 1)[0], n_min, replace=False)
        idx = np.sort(np.concatenate([idx0, idx1]))
        xt, yt, ct = xt[idx], yt[idx], ct[idx]
        print(f"After balancing: {len(xt)} samples ({n_min} per class)")

    # Shuffle
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(xt))
    xt, yt, ct = xt[perm], yt[perm], ct[perm]

    # Save
    print(f"\nFinal: xt={xt.shape}, ct={ct.shape}")
    np.savez_compressed(output_dir / 'data.npz', xt=xt, yt=yt, ct=ct)
    np.savez_compressed(output_dir / 'metadata.npz',
                        input_dim=xt.shape[-1],
                        n_variables=len(var_names),
                        variable_names=var_names,
                        variable_groups=groups,
                        lag=lag,
                        seq_len=lag + 1,
                        dataset_name='ERSSTv5_tropical_pacific',
                        regime_description='El Nino (ONI>=0.5) vs Non-El Nino')

    print(f"[SAVED] {output_dir / 'data.npz'}")
    print(f"[SAVED] {output_dir / 'metadata.npz'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sst_path', default='data/climate/raw/sst.mnmean.nc')
    parser.add_argument('--oni_path', default='data/climate/raw/oni_raw.txt')
    parser.add_argument('--output_dir', default='data/climate/prepared')
    parser.add_argument('--lag', type=int, default=2)
    parser.add_argument('--balance', action='store_true', default=True)
    # Variable screening options
    parser.add_argument('--screen', action='store_true', default=False,
                        help='Enable variable pre-screening (recommended for high-dim data)')
    parser.add_argument('--screen_d', type=float, default=0.5,
                        help='Cohen d threshold for discriminability filter (default: 0.5)')
    parser.add_argument('--screen_r', type=float, default=0.9,
                        help='Correlation threshold for deduplication (default: 0.9)')
    parser.add_argument('--screen_target_dim', type=int, default=None,
                        help='Target dimensionality after screening (default: auto)')
    args = parser.parse_args()
    main(args)
