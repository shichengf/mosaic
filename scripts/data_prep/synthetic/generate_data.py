#!/usr/bin/env python3
"""
generate_data.py — Parametric-coupling synthetic benchmark (v2).

Uses JAX autodiff for force computation from the shared potential U_total.
No manual gradient derivation needed — forces come from jax.grad automatically.

Potential (from double_well_potential notebook):
  U_total(x, z1, z2, z3, u1, u2) =
    U_x(x; depth1=z1, depth2=z2, barrier=z3)  [Gaussian double-well]
    + U_z1(z1) + U_z2(z2) + U_z3(z3)          [Gaussian double-wells]
    + U_u1(u1)                                  [double-well, uncoupled]
    + U_u2(u2)                                  [triple-well, uncoupled]

Integrator: Euler-Maruyama Langevin dynamics (JIT-compiled via JAX).

Usage:
    python scripts/data_prep/synthetic/generate_data.py
    python scripts/data_prep/synthetic/generate_data.py --interaction_alpha 0.5
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from config import *

import jax
import jax.numpy as jnp

PROJ = Path(__file__).resolve().parent.parent.parent.parent
RAW_DIR = PROJ / "data/synthetic_well_v2/raw"
SRC_DIR = PROJ / "data/synthetic_well_v2/source"
SANITY_DIR = PROJ / "results/sanity"
for d in [RAW_DIR, SRC_DIR, SANITY_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# JAX potential (autodiff-ready)
# ══════════════════════════════════════════════════════════════

def _gauss(q, center, sigma):
    return jnp.exp(-(q - center)**2 / (2 * sigma**2))


def _U_x(x, depth1, depth2, barrier_height):
    """Gaussian double-well for x, parameterized by z values."""
    return (-depth1 * _gauss(x, X_LEFT, SIGMA_WELL)
            - depth2 * _gauss(x, X_RIGHT, SIGMA_WELL)
            + barrier_height * _gauss(x, 0.0, SIGMA_BARRIER)
            + CONFINEMENT_K * x**4)


def _U_dw(q, q_left, q_right, depth, barrier, sw, sb, ck):
    """Generic Gaussian double-well."""
    q_mid = (q_left + q_right) / 2
    return (-depth * _gauss(q, q_left, sw)
            - depth * _gauss(q, q_right, sw)
            + barrier * _gauss(q, q_mid, sb)
            + ck * (q - q_mid)**4)


def _U_tw(q, qL, qM, qR, depth, barrier, sw, sb, ck):
    """Triple well: three Gaussian wells + two barriers."""
    q_center = (qL + qR) / 2
    return (-depth * _gauss(q, qL, sw)
            - depth * _gauss(q, qM, sw)
            - depth * _gauss(q, qR, sw)
            + barrier * _gauss(q, (qL + qM) / 2, sb)
            + barrier * _gauss(q, (qM + qR) / 2, sb)
            + ck * (q - q_center)**4)


def U_total_jax(q):
    """Total potential on 6D vector q = (x, z1, z2, z3, u1, u2).
    z1, z2, z3 are clipped to valid parameter ranges before entering U_x.
    """
    x, z1, z2, z3, u1, u2 = q[0], q[1], q[2], q[3], q[4], q[5]

    # z values ARE the x-potential parameters (clipped for safety)
    d1 = jnp.clip(z1, 0.1, 3.0)
    d2 = jnp.clip(z2, 0.1, 3.0)
    bh = jnp.clip(z3, 0.05, 3.0)

    Ux = _U_x(x, d1, d2, bh)
    Uz1 = _U_dw(z1, Z1_LEFT, Z1_RIGHT, Z_DEPTH, Z_BARRIER_HEIGHT,
                 Z_SIGMA_WELL, Z_SIGMA_BARRIER, Z_CONFINEMENT_K)
    Uz2 = _U_dw(z2, Z2_LEFT, Z2_RIGHT, Z_DEPTH, Z_BARRIER_HEIGHT,
                 Z_SIGMA_WELL, Z_SIGMA_BARRIER, Z_CONFINEMENT_K)
    Uz3 = _U_dw(z3, Z3_LEFT, Z3_RIGHT, Z_DEPTH, Z_BARRIER_HEIGHT,
                 Z_SIGMA_WELL, Z_SIGMA_BARRIER, Z_CONFINEMENT_K)
    Uu1 = _U_dw(u1, U1_LEFT, U1_RIGHT, U1_DEPTH, U1_BARRIER,
                 U1_SIGMA_WELL, U1_SIGMA_BARRIER, U1_CONFINEMENT_K)
    Uu2 = _U_tw(u2, U2_LEFT, U2_MID, U2_RIGHT, U2_DEPTH, U2_BARRIER,
                 U2_SIGMA_WELL, U2_SIGMA_BARRIER, U2_CONFINEMENT_K)

    return Ux + Uz1 + Uz2 + Uz3 + Uu1 + Uu2


# Force = -grad(U_total) — computed automatically by JAX!
_grad_U = jax.grad(U_total_jax)


# ══════════════════════════════════════════════════════════════
# Langevin dynamics (Euler-Maruyama, JIT-compiled)
# ══════════════════════════════════════════════════════════════

_drift = DT / GAMMA
_noise_scale = jnp.sqrt(2 * KT * DT / GAMMA)


@jax.jit
def _langevin_step(q, key):
    """One Euler-Maruyama Langevin step."""
    grad_U = _grad_U(q)
    noise = jax.random.normal(key, shape=(Z_DIM,))
    q_new = q - _drift * grad_U + _noise_scale * noise
    return q_new


def simulate_trajectory_jax(n_steps, init_q, seed):
    """Run Langevin dynamics and return subsampled trajectory."""
    key = jax.random.PRNGKey(seed)

    def scan_body(carry, _):
        q, key = carry
        key, subkey = jax.random.split(key)
        q_new = _langevin_step(q, subkey)
        return (q_new, key), q_new

    init_q_jax = jnp.array(init_q, dtype=jnp.float32)
    (_, _), trajectory = jax.lax.scan(scan_body, (init_q_jax, key), None,
                                       length=n_steps)
    # trajectory: (n_steps, Z_DIM)
    return np.array(trajectory)


def assign_state_frame(x_val):
    if x_val < -X_THRESHOLD:
        return 0
    elif x_val > X_THRESHOLD:
        return 1
    return -1


# ══════════════════════════════════════════════════════════════
# Mixing (same as v1)
# ══════════════════════════════════════════════════════════════

def build_mixing(rng):
    func_lib = {
        'sin':      lambda z, w, phi: w * np.sin(z + phi),
        'tanh':     lambda z, w, phi: w * np.tanh(2 * (z + phi)),
        'cubic':    lambda z, w, phi: w * (z + phi)**3 / 5,
        'sigmoid':  lambda z, w, phi: w / (1 + np.exp(-3 * (z + phi))),
        'softplus': lambda z, w, phi: w * np.log(1 + np.exp(2 * (z + phi))) / 2,
    }
    inv_support = {i: [] for i in range(OBS_DIM)}
    for j, x_indices in SUPPORT_SETS.items():
        for i in x_indices:
            fname = rng.choice(MIXING_FUNC_NAMES)
            w = rng.uniform(0.5, 2.0)
            phi = rng.uniform(-1.0, 1.0)
            inv_support[i].append((j, fname, w, phi))
    for i in NOISE_ONLY_OBS:
        inv_support[i] = []
    return func_lib, inv_support


def apply_mixing(z_traj, func_lib, inv_support, rng, interaction_alpha=0.0):
    n_frames = z_traj.shape[0]
    x_traj = np.zeros((n_frames, OBS_DIM))
    for i in range(OBS_DIM):
        for (j, fname, w, phi) in inv_support[i]:
            x_traj[:, i] += func_lib[fname](z_traj[:, j], w, phi)
    if interaction_alpha > 0:
        for x_i, (z_a, z_b) in INTERACTION_PAIRS.items():
            x_traj[:, x_i] += interaction_alpha * np.tanh(
                z_traj[:, z_a] * z_traj[:, z_b])
        for x_i, (z_a, z_b) in PURE_INTERACTION_OBS.items():
            x_traj[:, x_i] = interaction_alpha * np.tanh(
                z_traj[:, z_a] * z_traj[:, z_b])
    x_traj += rng.randn(n_frames, OBS_DIM) * OBS_NOISE_STD
    return x_traj


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def get_init_position(init_state, rng):
    """Initial 6D position consistent with a given state."""
    q = np.zeros(Z_DIM)
    if init_state == 'A':
        q[0] = -1.0       # x in left well
        q[1] = 1.5        # z1 high → deep left well
        q[2] = 0.5        # z2 low  → shallow right well
    else:
        q[0] = 1.0        # x in right well
        q[1] = 0.5        # z1 low  → shallow left well
        q[2] = 1.5        # z2 high → deep right well
    q[3] = 0.7            # z3 near middle of [0.2, 1.2]
    q[4] = -1.0 if rng.rand() < 0.5 else 1.0   # u1: random well
    q[5] = rng.choice([-1.0, 0.0, 1.0])         # u2: random triple-well state
    return q


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--interaction_alpha", type=float, default=INTERACTION_ALPHA)
    args = parser.parse_args()
    alpha = args.interaction_alpha

    if alpha > 0:
        suffix = f"_alpha{alpha:.1f}".replace(".", "p")
        raw_dir = PROJ / f"data/synthetic_well_v2{suffix}/raw"
        src_dir = PROJ / f"data/synthetic_well_v2{suffix}/source"
        sanity_dir = PROJ / f"results/interaction_alpha_{alpha:.1f}/sanity"
    else:
        raw_dir = RAW_DIR
        src_dir = SRC_DIR
        sanity_dir = SANITY_DIR
    for d in [raw_dir, src_dir, sanity_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Synthetic v2: Parametric Coupling (allosteric, JAX autodiff)")
    print(f"  dt={DT}, kT={KT}, gamma={GAMMA}")
    print(f"  n_steps={N_STEPS}, subsample={SUBSAMPLE_STEP}")
    if alpha > 0:
        print(f"  Interaction alpha = {alpha:.2f}")
    print("=" * 70)

    # Warm up JIT
    print("\nJIT-compiling Langevin step...")
    _test_q = jnp.zeros(Z_DIM)
    _test_key = jax.random.PRNGKey(0)
    _ = _langevin_step(_test_q, _test_key)
    print("  Done.")

    rng = np.random.RandomState(SEED)
    func_lib, inv_support = build_mixing(rng)

    mix_spec = []
    for i in range(OBS_DIM):
        for (j, fname, w, phi) in inv_support[i]:
            mix_spec.append((i, j, fname, w, phi))

    effective_noise = [] if alpha > 0 else NOISE_ONLY_OBS
    print(f"\nMixing: {len(mix_spec)} connections, noise-only: {effective_noise}")

    all_z, all_x, all_states = [], [], []

    for idx, traj_cfg in enumerate(TRAJECTORIES):
        name = traj_cfg['name']
        n_steps = traj_cfg['n_steps']
        traj_rng = np.random.RandomState(traj_cfg['seed'])

        init_q = get_init_position(traj_cfg['init_state'], traj_rng)
        print(f"\n  {name}: simulating {n_steps} steps...", end=" ", flush=True)

        z_traj_full = simulate_trajectory_jax(n_steps, init_q, traj_cfg['seed'])
        print(f"done. Subsampling 1/{SUBSAMPLE_STEP}...", end=" ", flush=True)

        # Subsample
        z_traj = z_traj_full[::SUBSAMPLE_STEP]
        n_frames = len(z_traj)

        # Assign states
        states = np.array([assign_state_frame(z_traj[t, 0]) for t in range(n_frames)])

        # Apply mixing
        x_traj = apply_mixing(z_traj, func_lib, inv_support, traj_rng,
                              interaction_alpha=alpha)

        n_a = (states == 0).sum()
        n_b = (states == 1).sum()
        n_int = (states == -1).sum()
        definite = states[states >= 0]
        n_trans = (np.diff(definite) != 0).sum() if len(definite) > 1 else 0

        print(f"frames={n_frames}, A={n_a}, B={n_b}, inter={n_int}, trans~{n_trans}")

        all_z.append(z_traj)
        all_x.append(x_traj)
        all_states.append(states)

    total_a = sum((s == 0).sum() for s in all_states)
    total_b = sum((s == 1).sum() for s in all_states)
    print(f"\n  Total: A={total_a}, B={total_b}, ratio={total_a/(total_b+1e-10):.2f}")

    # Save raw
    z_arr = np.empty(len(all_z), dtype=object)
    x_arr = np.empty(len(all_x), dtype=object)
    s_arr = np.empty(len(all_states), dtype=object)
    for i in range(len(all_z)):
        z_arr[i] = all_z[i]
        x_arr[i] = all_x[i]
        s_arr[i] = all_states[i]

    np.savez_compressed(raw_dir / "trajectories.npz",
                        z_trajs=z_arr, x_trajs=x_arr, state_labels=s_arr,
                        n_trajs=len(all_z))
    print(f"\n[SAVED] {raw_dir / 'trajectories.npz'}")

    # Save ground truth
    support_matrix = np.zeros((OBS_DIM, Z_DIM), dtype=bool)
    for j, x_indices in SUPPORT_SETS.items():
        for i in x_indices:
            support_matrix[i, j] = True
    if alpha > 0:
        for x_i, (z_a, z_b) in PURE_INTERACTION_OBS.items():
            support_matrix[x_i, z_a] = True
            support_matrix[x_i, z_b] = True

    np.savez_compressed(src_dir / "ground_truth.npz",
                        support_matrix=support_matrix,
                        regime_driving_factors=np.array(REGIME_DRIVING_FACTORS),
                        regime_invariant_factors=np.array(REGIME_INVARIANT_FACTORS),
                        noise_only_obs=np.array(effective_noise),
                        mix_spec=np.array(mix_spec, dtype=object),
                        inv_support=inv_support,
                        interaction_alpha=alpha)
    print(f"[SAVED] {src_dir / 'ground_truth.npz'}")

    # ── Sanity plots ──
    print("\nGenerating sanity check plots...")
    z0 = all_z[0]
    s0 = all_states[0]

    # Plot 1: z trajectories colored by state
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    t = np.arange(len(z0))
    for j in range(Z_DIM):
        ax = axes[j // 2, j % 2]
        ax.plot(t, z0[:, j], linewidth=0.3, alpha=0.4, color='gray')
        if j in REGIME_DRIVING_FACTORS:
            mask_a = s0 == 0
            mask_b = s0 == 1
            if mask_a.any():
                ax.scatter(t[mask_a], z0[mask_a, j], s=0.1, c='blue', alpha=0.3, label='A')
            if mask_b.any():
                ax.scatter(t[mask_b], z0[mask_b, j], s=0.1, c='red', alpha=0.3, label='B')
            ax.legend(fontsize=7, markerscale=10)
        ax.set_title(f'dim {j}: {DIM_NAMES[j]}')
        ax.set_ylabel(f'q_{j}')
    axes[-1, 0].set_xlabel('Frame')
    axes[-1, 1].set_xlabel('Frame')
    fig.suptitle(f'Latent trajectories (traj_1) — kT={KT}, dt={DT}', fontsize=13)
    fig.tight_layout()
    fig.savefig(sanity_dir / "sanity_z_trajectories.png", dpi=150)
    plt.close()

    # Plot 2: x potential at different z configs (using numpy versions)
    x_grid = np.linspace(-2.5, 2.5, 500)
    z_configs = [
        (1.5, 0.5, 0.7, "z1=1.5, z2=0.5 (deep L)"),
        (0.5, 1.5, 0.7, "z1=0.5, z2=1.5 (deep R)"),
        (1.0, 1.0, 0.7, "z1=z2=1.0 (symmetric)"),
        (1.5, 1.5, 0.7, "z1=z2=1.5 (both deep)"),
        (1.0, 1.0, 0.2, "z3=0.2 (low barrier)"),
        (1.0, 1.0, 1.2, "z3=1.2 (high barrier)"),
    ]
    fig, axes_p = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (z1v, z2v, z3v, label) in zip(axes_p.flatten(), z_configs):
        d1, d2 = np.clip(z1v, 0.1, 3.0), np.clip(z2v, 0.1, 3.0)
        bh = np.clip(z3v, 0.05, 3.0)
        U = (-d1 * np.exp(-(x_grid - X_LEFT)**2 / (2 * SIGMA_WELL**2))
             - d2 * np.exp(-(x_grid - X_RIGHT)**2 / (2 * SIGMA_WELL**2))
             + bh * np.exp(-x_grid**2 / (2 * SIGMA_BARRIER**2))
             + CONFINEMENT_K * x_grid**4)
        ax.plot(x_grid, U, 'b-', linewidth=2)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel('x'); ax.set_ylabel('U_x')
        ax.grid(True, alpha=0.3); ax.axhline(0, color='k', linewidth=0.5)
    fig.suptitle('x potential under different z configurations', fontsize=13)
    fig.tight_layout()
    fig.savefig(sanity_dir / "sanity_x_potential_configs.png", dpi=150)
    plt.close()

    # Plot 3: allosteric correlation scatter
    fig, axes_s = plt.subplots(1, 3, figsize=(15, 4))
    n_show = min(10000, len(z0))
    c_arr = s0[:n_show].astype(float)
    c_arr[s0[:n_show] == -1] = 0.5
    axes_s[0].scatter(z0[:n_show, 0], z0[:n_show, 1], s=0.5, c=c_arr,
                      cmap='coolwarm', alpha=0.3)
    axes_s[0].set_xlabel('x'); axes_s[0].set_ylabel('z1')
    axes_s[0].set_title('x vs z1 (allosteric)')
    axes_s[1].scatter(z0[:n_show, 0], z0[:n_show, 2], s=0.5, c=c_arr,
                      cmap='coolwarm', alpha=0.3)
    axes_s[1].set_xlabel('x'); axes_s[1].set_ylabel('z2')
    axes_s[1].set_title('x vs z2 (allosteric)')
    axes_s[2].scatter(z0[:n_show, 1], z0[:n_show, 2], s=0.5, c=c_arr,
                      cmap='coolwarm', alpha=0.3)
    axes_s[2].set_xlabel('z1'); axes_s[2].set_ylabel('z2')
    axes_s[2].set_title('z1 vs z2')
    fig.tight_layout()
    fig.savefig(sanity_dir / "sanity_allosteric_scatter.png", dpi=150)
    plt.close()

    # Plot 4: support matrix
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(support_matrix.astype(float), aspect='auto', cmap='YlOrRd',
              interpolation='nearest')
    ax.set_xlabel('Latent dim'); ax.set_ylabel('Observed x_i')
    ax.set_xticks(range(Z_DIM))
    ax.set_xticklabels([f'{j}\n{DIM_NAMES[j][:8]}' for j in range(Z_DIM)], fontsize=7)
    ax.set_title('Ground Truth Support Matrix')
    for i in effective_noise:
        ax.text(Z_DIM + 0.3, i, '<- noise', va='center', fontsize=8, color='gray')
    for i in range(OBS_DIM):
        if support_matrix[i].sum() > 1:
            ax.text(Z_DIM + 0.3, i, '<- shared', va='center', fontsize=8, color='orange')
    fig.savefig(sanity_dir / "sanity_support_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 5: state distributions + Cohen's d
    fig, axes_h = plt.subplots(2, 3, figsize=(15, 8))
    mask_a = s0 == 0
    mask_b = s0 == 1
    for j in range(Z_DIM):
        ax = axes_h[j // 3, j % 3]
        if mask_a.sum() > 0:
            ax.hist(z0[mask_a, j], bins=80, alpha=0.5, label='A', color='blue', density=True)
        if mask_b.sum() > 0:
            ax.hist(z0[mask_b, j], bins=80, alpha=0.5, label='B', color='red', density=True)
        ax.set_title(f'{DIM_NAMES[j]}')
        ax.legend(fontsize=7)
        if mask_a.sum() > 10 and mask_b.sum() > 10:
            mu_a, mu_b = z0[mask_a, j].mean(), z0[mask_b, j].mean()
            std_pool = np.sqrt((z0[mask_a, j].var() + z0[mask_b, j].var()) / 2 + 1e-10)
            d = abs(mu_b - mu_a) / std_pool
            ax.set_xlabel(f"Cohen's d = {d:.2f}")
    fig.suptitle('Distribution by state (traj_1)', fontsize=13)
    fig.tight_layout()
    fig.savefig(sanity_dir / "sanity_state_distributions.png", dpi=150)
    plt.close()

    # Plot 6: z potential landscapes (for reference)
    fig, axes_pot = plt.subplots(2, 3, figsize=(15, 8))
    q_ranges = [
        np.linspace(-2.5, 2.5, 300),  # x
        np.linspace(0, 2.5, 300),      # z1
        np.linspace(0, 2.5, 300),      # z2
        np.linspace(-0.5, 2.0, 300),   # z3
        np.linspace(-2.5, 2.5, 300),   # u1
        np.linspace(-2.5, 2.5, 300),   # u2
    ]
    for j in range(Z_DIM):
        ax = axes_pot[j // 3, j % 3]
        q_vals = q_ranges[j]
        U_vals = np.array([float(U_total_jax(
            jnp.array([q if j == 0 else (-1.0 if j > 3 else 1.0),
                        q if j == 1 else 1.0,
                        q if j == 2 else 1.0,
                        q if j == 3 else 0.7,
                        q if j == 4 else 0.0,
                        q if j == 5 else 0.0])))
                           for q in q_vals])
        # Subtract the constant part to show only this dim's contribution
        ax.plot(q_vals, U_vals - U_vals.min(), linewidth=2)
        ax.set_title(f'{DIM_NAMES[j]}')
        ax.set_xlabel(f'q_{j}'); ax.set_ylabel('U (shifted)')
        ax.grid(True, alpha=0.3)
    fig.suptitle('Potential landscape per dimension (others fixed)', fontsize=13)
    fig.tight_layout()
    fig.savefig(sanity_dir / "sanity_potentials.png", dpi=150)
    plt.close()

    print(f"[SAVED] Sanity plots to {sanity_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
