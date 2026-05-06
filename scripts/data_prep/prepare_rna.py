#!/usr/bin/env python3
"""
67c_prepare_rna_v3.py — RNA data preparation with module-specific regime + multiple features

Key changes from 67b:
  - Regime defined by loop-specific Q (not global Q), so 400K partial
    unfolding events are correctly labeled as disrupted.
  - Multiple feature modes: chi+delta sin/cos (D=56), per-residue dist (D=28),
    or combined (D=84).
  - Diagnostic output for regime verification.

Usage:
    python scripts/data_prep/prepare_rna.py --features chi_delta_sincos --output_dir data/RNA/prepared_chi_delta_v3
    python scripts/data_prep/prepare_rna.py --features per_residue_dist --output_dir data/RNA/prepared_dist_v3
"""

import numpy as np
import mdtraj as md
import argparse
from pathlib import Path
from itertools import combinations
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

RNA_SEQUENCE = "GGCACUUCGGUGCC"

# Ground-truth structural modules (0-indexed residue IDs)
MODULES = {
    "OuterStem":   [0, 1, 12, 13],
    "InnerStem":   [2, 3, 10, 11],
    "ClosingPair": [4, 9],
    "Loop":        [5, 6, 7, 8],
}

RES_TO_MOD = {}
for mod, residues in MODULES.items():
    for r in residues:
        RES_TO_MOD[r] = mod

# Loop + closing pair residues for module-specific regime
LOOP_RESIDUES = {4, 5, 6, 7, 8, 9}  # C5, U6, U7, C8, G9, G10

# Outer stem residues for stem-based regime (avoids circular reasoning with loop)
STEM_RESIDUES = {0, 1, 12, 13}  # G1, G2, C13, C14

# Closing pair residues for CP regime (tests what drives CP opening)
CP_RESIDUES = {4, 9}  # C5, G10

PURINES = {'G', 'A'}
PYRIMIDINES = {'C', 'U'}


# ============================================================
# Feature extraction
# ============================================================
def compute_chi_angles(traj):
    """Manual chi angle: O4'-C1'-N9-C4 (purines) or O4'-C1'-N1-C2 (pyrimidines)."""
    quads = []
    for res in traj.topology.residues:
        atoms = {a.name: a.index for a in res.atoms}
        rn = res.name.strip()
        if rn in PURINES:
            quads.append([atoms["O4'"], atoms["C1'"], atoms["N9"], atoms["C4"]])
        elif rn in PYRIMIDINES:
            quads.append([atoms["O4'"], atoms["C1'"], atoms["N1"], atoms["C2"]])
        else:
            raise ValueError(f"Unknown residue: {rn}")
    return md.compute_dihedrals(traj, np.array(quads))  # (n_frames, 14)


def compute_delta_angles(traj):
    """Manual delta angle: C5'-C4'-C3'-O3' for each residue."""
    quads = []
    for res in traj.topology.residues:
        atoms = {a.name: a.index for a in res.atoms}
        quads.append([atoms["C5'"], atoms["C4'"], atoms["C3'"], atoms["O3'"]])
    return md.compute_dihedrals(traj, np.array(quads))  # (n_frames, 14)


def compute_chi_delta_sincos(traj):
    """Chi + delta with sin/cos encoding. D=56."""
    chi = compute_chi_angles(traj)    # (n_frames, 14)
    delta = compute_delta_angles(traj)  # (n_frames, 14)
    return np.concatenate([
        np.sin(chi), np.cos(chi), np.sin(delta), np.cos(delta)
    ], axis=1)  # (n_frames, 56)


# Full backbone torsion definitions (from 67_prepare_rna.py)
BACKBONE_TORSIONS = {
    "alpha":   [("O3'", -1), ("P",   0), ("O5'", 0), ("C5'", 0)],
    "beta":    [("P",    0), ("O5'", 0), ("C5'", 0), ("C4'", 0)],
    "gamma":   [("O5'",  0), ("C5'", 0), ("C4'", 0), ("C3'", 0)],
    "delta":   [("C5'",  0), ("C4'", 0), ("C3'", 0), ("O3'", 0)],
    "epsilon": [("C4'",  0), ("C3'", 0), ("O3'", 0), ("P",  +1)],
    "zeta":    [("C3'",  0), ("O3'", 0), ("P",  +1), ("O5'",+1)],
}


def _find_atom_index(topology, res_idx, atom_name):
    """Return the atom index for a given residue index and atom name, or None."""
    residue = list(topology.residues)[res_idx]
    for atom in residue.atoms:
        if atom.name == atom_name:
            return atom.index
    return None


def build_torsion_indices(topology):
    """Build arrays of 4-atom index tuples for all RNA torsion angles.

    Returns
    -------
    indices : np.ndarray, shape (n_angles, 4)
    angle_names : list of str, e.g. "G1_delta"
    angle_residues : list of int (1-indexed residue number)
    """
    residues = list(topology.residues)
    n_res = len(residues)
    indices, angle_names, angle_residues = [], [], []

    for i in range(n_res):
        res = residues[i]
        res_label = f"{res.name}{i + 1}"

        for torsion_name, atom_specs in BACKBONE_TORSIONS.items():
            atom_indices = []
            ok = True
            for atom_name, offset in atom_specs:
                neighbor = i + offset
                if neighbor < 0 or neighbor >= n_res:
                    ok = False
                    break
                idx = _find_atom_index(topology, neighbor, atom_name)
                if idx is None:
                    ok = False
                    break
                atom_indices.append(idx)
            if ok and len(atom_indices) == 4:
                indices.append(atom_indices)
                angle_names.append(f"{res_label}_{torsion_name}")
                angle_residues.append(i + 1)

        # Chi torsion
        if res.name in PURINES:
            chi_atoms = ["O4'", "C1'", "N9", "C4"]
        else:
            chi_atoms = ["O4'", "C1'", "N1", "C2"]
        atom_indices = []
        ok = True
        for aname in chi_atoms:
            idx = _find_atom_index(topology, i, aname)
            if idx is None:
                ok = False
                break
            atom_indices.append(idx)
        if ok and len(atom_indices) == 4:
            indices.append(atom_indices)
            angle_names.append(f"{res_label}_chi")
            angle_residues.append(i + 1)

    return np.array(indices, dtype=np.int32), angle_names, angle_residues


def compute_all_dihedral_sincos(traj):
    """All 7 backbone angles with sin/cos encoding. D=2*n_angles (~188)."""
    indices, angle_names, angle_residues = build_torsion_indices(traj.topology)
    angles = md.compute_dihedrals(traj, indices)  # (n_frames, n_angles)
    features = np.concatenate([np.sin(angles), np.cos(angles)], axis=1)
    return features, angle_names, angle_residues


def compute_per_residue_distances(traj):
    """Per-residue distance features: mean + std of distances. D=28."""
    n_res = traj.n_residues
    res_pairs = list(combinations(range(n_res), 2))
    contacts, _ = md.compute_contacts(traj, contacts=res_pairs, scheme='closest-heavy')

    res_dists = np.zeros((traj.n_frames, n_res, n_res - 1))
    for idx, (i, j) in enumerate(res_pairs):
        others_i = [x for x in range(n_res) if x != i]
        others_j = [x for x in range(n_res) if x != j]
        col_i = others_i.index(j)
        col_j = others_j.index(i)
        res_dists[:, i, col_i] = contacts[:, idx]
        res_dists[:, j, col_j] = contacts[:, idx]

    res_mean = res_dists.mean(axis=2)  # (n_frames, 14)
    res_std = res_dists.std(axis=2)    # (n_frames, 14)
    return np.concatenate([res_mean, res_std], axis=1)  # (n_frames, 28)


def compute_frozen_centroid_distances(traj, ref_frame=0):
    """Per-residue distance to a frozen reference centroid. D=28.

    Uses the centroid position from ref_frame (typically a stable folded frame)
    for all frames, eliminating artificial global coupling from per-frame centroid.
    Features: mean_dist_to_ref_centroid + std_dist_to_others (using frozen ref). D=28.
    """
    n_res = traj.n_residues
    res_pairs = list(combinations(range(n_res), 2))
    contacts, _ = md.compute_contacts(traj, contacts=res_pairs, scheme='closest-heavy')

    # Build full pairwise distance matrix per frame
    res_dists = np.zeros((traj.n_frames, n_res, n_res - 1))
    for idx, (i, j) in enumerate(res_pairs):
        others_i = [x for x in range(n_res) if x != i]
        others_j = [x for x in range(n_res) if x != j]
        col_i = others_i.index(j)
        col_j = others_j.index(i)
        res_dists[:, i, col_i] = contacts[:, idx]
        res_dists[:, j, col_j] = contacts[:, idx]

    # Frozen centroid: use ref_frame's mean distances as the reference
    ref_mean = res_dists[ref_frame].mean(axis=1)  # (n_res,) frozen centroid distances

    # Per-frame features: distance of each residue to all others, summarized
    # relative to frozen reference
    res_mean = res_dists.mean(axis=2)  # (n_frames, 14) per-frame mean dist to others
    res_std = res_dists.std(axis=2)    # (n_frames, 14) per-frame std dist to others

    # Subtract frozen centroid to remove global coupling
    res_mean_centered = res_mean - ref_mean[np.newaxis, :]  # deviation from ref

    return np.concatenate([res_mean_centered, res_std], axis=1)  # (n_frames, 28)


# ============================================================
# Q computation
# ============================================================
def compute_Q(traj, cutoff_nm=0.45, tolerance=1.2, min_seq_sep=3, module_residues=None):
    """Native contact fraction, optionally restricted to a residue subset.

    If module_residues is given, only contacts where at least one residue
    is in the set are considered.
    """
    n_res = traj.n_residues
    all_pairs = [(i, j) for i, j in combinations(range(n_res), 2) if j - i >= min_seq_sep]

    if module_residues is not None:
        all_pairs = [(i, j) for i, j in all_pairs
                     if i in module_residues or j in module_residues]

    if not all_pairs:
        return np.ones(traj.n_frames)

    contacts, _ = md.compute_contacts(traj, contacts=all_pairs, scheme='closest-heavy')
    native_dists = contacts[0]
    native_mask = native_dists < cutoff_nm
    if native_mask.sum() == 0:
        return np.ones(traj.n_frames)

    native_contacts = contacts[:, native_mask]
    native_thresholds = native_dists[native_mask] * tolerance
    return (native_contacts < native_thresholds).mean(axis=1)


# ============================================================
# Regime assignment
# ============================================================
def _smooth_Q(Q, window=51):
    """Rolling mean to denoise Q before thresholding."""
    if len(Q) < window:
        return Q
    kernel = np.ones(window) / window
    # Pad edges to avoid shrinkage
    pad = window // 2
    Q_padded = np.pad(Q, pad, mode='edge')
    return np.convolve(Q_padded, kernel, mode='valid')


def assign_regime_global(Q, q_folded=0.6, q_unfolded=0.3, buffer_frames=50,
                         smooth_window=51):
    """Original global-Q regime (same as 67b)."""
    Q_s = _smooth_Q(Q, smooth_window)
    labels = np.full(len(Q_s), -1, dtype=int)
    labels[Q_s > q_folded] = 0
    labels[Q_s < q_unfolded] = 1
    _apply_buffer(labels, buffer_frames)
    return labels


def assign_regime_loop(loop_Q, q_intact=0.8, q_disrupted=0.5, buffer_frames=50,
                       smooth_window=51):
    """Module-specific regime based on loop/closing-pair Q.

    Smooths loop_Q with a rolling mean before thresholding to avoid
    spurious transitions from thermal noise wiping out all data via buffer.
    """
    Q_s = _smooth_Q(loop_Q, smooth_window)
    labels = np.full(len(Q_s), -1, dtype=int)
    labels[Q_s > q_intact] = 0    # intact
    labels[Q_s < q_disrupted] = 1  # disrupted
    _apply_buffer(labels, buffer_frames)
    return labels


def assign_regime_cp(cp_Q, q_intact=0.8, q_disrupted=0.65, buffer_frames=50,
                     smooth_window=51):
    """Regime based on closing pair Q (C5, G10).

    Tests what drives CP opening. Regime 0 = CP intact, regime 1 = CP disrupted.
    Expected driver = Loop (especially U6/G9), not the CP itself.
    """
    Q_s = _smooth_Q(cp_Q, smooth_window)
    labels = np.full(len(Q_s), -1, dtype=int)
    labels[Q_s > q_intact] = 0
    labels[Q_s < q_disrupted] = 1
    _apply_buffer(labels, buffer_frames)
    return labels


def assign_regime_stem(stem_Q, q_intact=0.8, q_disrupted=0.5, buffer_frames=50,
                       smooth_window=51):
    """Regime based on outer stem Q (G1, G2, C13, C14).

    Uses the same logic as assign_regime_loop but on stem residues,
    eliminating circular reasoning when analyzing loop-driven transitions.
    """
    Q_s = _smooth_Q(stem_Q, smooth_window)
    labels = np.full(len(Q_s), -1, dtype=int)
    labels[Q_s > q_intact] = 0    # intact
    labels[Q_s < q_disrupted] = 1  # disrupted
    _apply_buffer(labels, buffer_frames)
    return labels


def _apply_buffer(labels, buffer_frames):
    transitions = np.where(np.diff(labels) != 0)[0]
    for t in transitions:
        lo = max(0, t - buffer_frames + 1)
        hi = min(len(labels), t + buffer_frames + 1)
        labels[lo:hi] = -1


# ============================================================
# Windowing
# ============================================================
def create_windows(features, labels, lag=2):
    seq_len = lag + 1
    windows, window_labels = [], []
    for t in range(seq_len - 1, len(features)):
        r = labels[t]
        if r < 0:
            continue
        if np.all(labels[t - seq_len + 1: t + 1] == r):
            windows.append(features[t - seq_len + 1: t + 1])
            window_labels.append(r)
    return windows, window_labels


# ============================================================
# Separability analysis
# ============================================================
def compute_separability(xt, ct, variable_groups):
    """Compute within-group vs between-group correlation and Cohen's d."""
    x = xt[:, -1, :]  # last frame of each window
    corr = np.abs(np.corrcoef(x.T))

    groups = np.array(variable_groups)
    within, between = [], []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if groups[i] == groups[j]:
                within.append(corr[i, j])
            else:
                between.append(corr[i, j])

    within_arr = np.array(within)
    between_arr = np.array(between)
    within_mean = within_arr.mean()
    between_mean = between_arr.mean()

    pooled_std = np.sqrt((within_arr.var() * len(within_arr) +
                          between_arr.var() * len(between_arr)) /
                         (len(within_arr) + len(between_arr)))
    cohens_d = (within_mean - between_mean) / (pooled_std + 1e-8)

    return within_mean, between_mean, cohens_d


# ============================================================
# Variable names and groups for each feature mode
# ============================================================
def get_var_info(feature_mode, angle_names=None, angle_residues=None):
    var_names, var_groups = [], []
    if feature_mode == 'all_dihedral_sincos':
        assert angle_names is not None, "angle_names required for all_dihedral_sincos"
        # sin block then cos block
        for prefix in ['sin', 'cos']:
            for name, res_1idx in zip(angle_names, angle_residues):
                var_names.append(f"{prefix}_{name}")
                var_groups.append(RES_TO_MOD[res_1idx - 1])  # 1-indexed → 0-indexed
    elif feature_mode in ('chi_delta_sincos', 'chi_delta_dist'):
        for feat_type in ['sin_chi', 'cos_chi', 'sin_delta', 'cos_delta']:
            for i in range(14):
                var_names.append(f"{RNA_SEQUENCE[i]}{i+1}_{feat_type}")
                var_groups.append(RES_TO_MOD[i])
    if feature_mode in ('per_residue_dist', 'chi_delta_dist', 'frozen_centroid_dist'):
        stat_names = ['mean_dist', 'std_dist'] if feature_mode != 'frozen_centroid_dist' else ['dev_dist', 'std_dist']
        for stat in stat_names:
            for i in range(14):
                var_names.append(f"{RNA_SEQUENCE[i]}{i+1}_{stat}")
                var_groups.append(RES_TO_MOD[i])
    return var_names, var_groups


# ============================================================
# Main
# ============================================================
def main(args):
    print("=" * 70)
    print("  RNA v3: Module-Specific Regime + Feature Evaluation")
    print("=" * 70)

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exclude_temps = set(args.exclude_temps) if args.exclude_temps else set()

    # Gather trajectories
    trajs_info = []

    # Main trajectory
    main_pdb = raw_dir / 'rna_only' / 'hairpin_rna_only.pdb'
    main_dcd = raw_dir / 'rna_only' / 'unfold_rna.dcd'
    if main_pdb.exists() and main_dcd.exists():
        if '500K' not in exclude_temps:
            trajs_info.append((main_pdb, main_dcd, 'main_500K', '500K'))
        else:
            print(f"  [SKIP] main_500K (excluded temp)")

    # Replicas (original short trajectories)
    rep_dir = raw_dir / 'RNA_simulations_replicas'
    if rep_dir.exists():
        for rep in sorted(rep_dir.iterdir()):
            if not rep.is_dir():
                continue
            pdb = rep / 'topology_rna.pdb'
            dcds = list(rep.glob('*.dcd'))
            if pdb.exists() and dcds:
                temp = dcds[0].stem.split('_')[-1]
                if temp in exclude_temps:
                    print(f"  [SKIP] {rep.name}_{temp} (excluded temp)")
                    continue
                trajs_info.append((pdb, dcds[0], f'{rep.name}_{temp}', temp))

    # Long 400K trajectories (rep12, rep15-17)
    long_dir = raw_dir / 'RNA_simulations_traj_400K'
    if long_dir.exists():
        for rep in sorted(long_dir.iterdir()):
            if not rep.is_dir():
                continue
            pdb = rep / 'topology_rna.pdb'
            dcd = rep / 'rna.dcd'
            if pdb.exists() and dcd.exists():
                if '400K' in exclude_temps:
                    print(f"  [SKIP] {rep.name}_400K (excluded temp)")
                    continue
                trajs_info.append((pdb, dcd, f'{rep.name}_400K', '400K'))

    print(f"\n[1] Processing {len(trajs_info)} trajectories "
          f"(features={args.features}, regime={args.regime})...")

    all_windows, all_labels = [], []
    diag_labels, diag_global_Q, diag_loop_Q, diag_stem_Q, diag_regime = [], [], [], [], []
    cached_angle_names, cached_angle_residues = None, None

    header = f"{'Trajectory':<16s} {'Temp':>5s} {'Frames':>7s} {'GlobQ':>7s} {'LoopQ':>7s} {'StemQ':>7s} {'Intact':>7s} {'Disrupt':>7s} {'Discard':>7s} {'Windows':>8s}"
    print(f"\n{header}")
    print("-" * len(header))

    for pdb, dcd, label, temp in trajs_info:
        traj = md.load(str(dcd), top=str(pdb))

        # Features
        if args.features == 'chi_delta_sincos':
            features = compute_chi_delta_sincos(traj)
        elif args.features == 'per_residue_dist':
            features = compute_per_residue_distances(traj)
        elif args.features == 'chi_delta_dist':
            features = np.concatenate([
                compute_chi_delta_sincos(traj),
                compute_per_residue_distances(traj)
            ], axis=1)
        elif args.features == 'frozen_centroid_dist':
            features = compute_frozen_centroid_distances(traj, ref_frame=0)
        elif args.features == 'all_dihedral_sincos':
            features, angle_names, angle_residues = compute_all_dihedral_sincos(traj)
            if cached_angle_names is None:
                cached_angle_names = angle_names
                cached_angle_residues = angle_residues
            else:
                assert angle_names == cached_angle_names, \
                    f"Angle mismatch on {label}: got {len(angle_names)}, expected {len(cached_angle_names)}"

        # Q values (always compute all three for diagnostics)
        global_Q = compute_Q(traj, cutoff_nm=args.contact_cutoff,
                             tolerance=args.contact_tolerance,
                             min_seq_sep=args.min_seq_sep)
        loop_Q = compute_Q(traj, cutoff_nm=args.contact_cutoff,
                           tolerance=args.contact_tolerance,
                           min_seq_sep=args.min_seq_sep,
                           module_residues=LOOP_RESIDUES)
        stem_Q = compute_Q(traj, cutoff_nm=args.contact_cutoff,
                           tolerance=args.contact_tolerance,
                           min_seq_sep=args.min_seq_sep,
                           module_residues=STEM_RESIDUES)
        cp_Q = compute_Q(traj, cutoff_nm=args.contact_cutoff,
                         tolerance=args.contact_tolerance,
                         min_seq_sep=args.min_seq_sep,
                         module_residues=CP_RESIDUES)

        # Regime
        if args.regime == 'global_Q':
            labels = assign_regime_global(global_Q,
                                          q_folded=args.q_intact,
                                          q_unfolded=args.q_disrupted,
                                          buffer_frames=args.buffer_frames,
                                          smooth_window=args.smooth_window)
        elif args.regime == 'stem_Q':
            labels = assign_regime_stem(stem_Q,
                                        q_intact=args.q_intact,
                                        q_disrupted=args.q_disrupted,
                                        buffer_frames=args.buffer_frames,
                                        smooth_window=args.smooth_window)
        elif args.regime == 'cp_Q':
            labels = assign_regime_cp(cp_Q,
                                      q_intact=args.q_intact,
                                      q_disrupted=args.q_disrupted,
                                      buffer_frames=args.buffer_frames,
                                      smooth_window=args.smooth_window)
        else:
            labels = assign_regime_loop(loop_Q,
                                        q_intact=args.q_intact,
                                        q_disrupted=args.q_disrupted,
                                        buffer_frames=args.buffer_frames,
                                        smooth_window=args.smooth_window)

        n0 = (labels == 0).sum()
        n1 = (labels == 1).sum()
        n_dis = (labels == -1).sum()

        # Save sequential (pre-windowed) data for SPIB baseline
        if args.save_sequential:
            seq_dir = output_dir / 'sequential'
            seq_dir.mkdir(parents=True, exist_ok=True)
            np.save(seq_dir / f'{label}_features.npy', features.astype(np.float32))
            np.save(seq_dir / f'{label}_labels.npy', labels.astype(np.int8))

        wins, labs = create_windows(features, labels, lag=args.lag)

        print(f"{label:<16s} {temp:>5s} {traj.n_frames:>7d} "
              f"{global_Q.mean():>7.3f} {loop_Q.mean():>7.3f} {stem_Q.mean():>7.3f} "
              f"{n0:>7d} {n1:>7d} {n_dis:>7d} {len(wins):>8d}")

        all_windows.extend(wins)
        all_labels.extend(labs)

        # Diagnostics
        diag_labels.extend([label] * traj.n_frames)
        diag_global_Q.extend(global_Q.tolist())
        diag_loop_Q.extend(loop_Q.tolist())
        diag_stem_Q.extend(stem_Q.tolist())
        diag_regime.extend(labels.tolist())

    print("-" * len(header))

    if not all_windows:
        print("\nERROR: No windows created! Check regime thresholds.")
        return

    # Assemble dataset
    print(f"\n[2] Assembling dataset...")
    xt = np.array(all_windows, dtype=np.float32)
    ct = np.array(all_labels, dtype=np.float32).reshape(-1, 1)
    D = xt.shape[-1]

    n0_raw = int((ct == 0).sum())
    n1_raw = int((ct == 1).sum())
    print(f"    Raw: xt={xt.shape}, D={D}")
    print(f"    Regime 0 (intact)={n0_raw}, Regime 1 (disrupted)={n1_raw}")

    # Standardize
    xt_flat = xt.reshape(-1, D)
    mean = xt_flat.mean(axis=0, keepdims=True)
    std = xt_flat.std(axis=0, keepdims=True) + 1e-8
    xt = ((xt.reshape(-1, D) - mean) / std).reshape(xt.shape).astype(np.float32)

    # Balance
    n_min = min(n0_raw, n1_raw)
    if n_min > 0:
        rng = np.random.RandomState(args.seed)
        idx0 = rng.choice(np.where(ct.squeeze() == 0)[0], n_min, replace=False)
        idx1 = rng.choice(np.where(ct.squeeze() == 1)[0], n_min, replace=False)
        idx = rng.permutation(np.concatenate([idx0, idx1]))
        xt, ct = xt[idx], ct[idx]

    yt = xt.copy()
    print(f"    Balanced: xt={xt.shape}")
    print(f"    Regime 0={int((ct==0).sum())}, Regime 1={int((ct==1).sum())}")

    # Variable info
    var_names, var_groups = get_var_info(args.features,
                                         angle_names=cached_angle_names,
                                         angle_residues=cached_angle_residues)
    print(f"    Groups: {dict(Counter(var_groups))}")

    # Separability
    print(f"\n[3] Separability analysis...")
    within_r, between_r, cohens_d = compute_separability(xt, ct, var_groups)
    print(f"    Within-group  |r|: {within_r:.3f}")
    print(f"    Between-group |r|: {between_r:.3f}")
    print(f"    Cohen's d:         {cohens_d:.3f}")

    # Save
    np.savez_compressed(output_dir / 'data.npz', xt=xt, yt=yt, ct=ct)
    np.savez_compressed(output_dir / 'metadata.npz',
        input_dim=D, n_variables=D,
        variable_names=np.array(var_names),
        variable_groups=np.array(var_groups),
        lag=args.lag, seq_len=args.lag + 1,
        dataset_name=f'rna_{args.features}',
        regime_description=f'Regime via {args.regime}: intact (>{args.q_intact}) vs disrupted (<{args.q_disrupted})',
        feature_mode=args.features,
        regime_mode=args.regime)
    np.savez_compressed(output_dir / 'diagnostics.npz',
        trajectory_labels=np.array(diag_labels),
        global_Q=np.array(diag_global_Q, dtype=np.float32),
        loop_Q=np.array(diag_loop_Q, dtype=np.float32),
        stem_Q=np.array(diag_stem_Q, dtype=np.float32),
        regime_labels=np.array(diag_regime, dtype=np.int8))

    print(f"\n  [SAVED] {output_dir}/data.npz")
    print(f"  [SAVED] {output_dir}/metadata.npz")
    print(f"  [SAVED] {output_dir}/diagnostics.npz")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RNA v3 data preparation")
    parser.add_argument("--raw_dir", default="data/RNA")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--features", choices=['chi_delta_sincos', 'per_residue_dist', 'chi_delta_dist', 'all_dihedral_sincos', 'frozen_centroid_dist'],
                        default='chi_delta_sincos')
    parser.add_argument("--regime", choices=['global_Q', 'loop_Q', 'stem_Q', 'cp_Q'], default='loop_Q')
    parser.add_argument("--q_intact", type=float, default=0.8)
    parser.add_argument("--q_disrupted", type=float, default=0.5)
    parser.add_argument("--lag", type=int, default=2)
    parser.add_argument("--buffer_frames", type=int, default=50)
    parser.add_argument("--contact_cutoff", type=float, default=0.45)
    parser.add_argument("--contact_tolerance", type=float, default=1.2)
    parser.add_argument("--min_seq_sep", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smooth_window", type=int, default=51,
                        help="Rolling mean window for Q smoothing before thresholding (default: 51)")
    parser.add_argument("--exclude_temps", nargs='*', default=[])
    parser.add_argument("--save_sequential", action='store_true',
                        help="Save per-trajectory sequential features + labels (for SPIB)")
    args = parser.parse_args()
    main(args)
