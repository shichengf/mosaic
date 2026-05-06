"""config.py — Parametric-coupling synthetic benchmark (v2).

Key innovation vs v1: z1, z2, z3 parametrically control x's double-well
potential shape (depth1, depth2, barrier_height), modeling allosteric coupling.

Physical picture (from double_well_potential notebook):
  - x: reaction coordinate (protein open/close), Gaussian double-well
  - z1 → depth1: controls left well depth, Gaussian double-well at [0.5, 1.5]
  - z2 → depth2: controls right well depth, Gaussian double-well at [0.5, 1.5]
  - z3 → barrier: controls barrier height, Gaussian double-well at [0.2, 1.2]
  - u1: uncoupled double-well at [-1, +1] (noise/invariant)
  - u2: uncoupled TRIPLE well at [-1, 0, +1] (noise/invariant, 3-state)

Coupling: U_total = U_x(x; z1, z2, z3) + U_z1 + U_z2 + U_z3 + U_u1 + U_u2
  Forces from dU_total/dq give natural bidirectional allosteric feedback.
  z values ARE the x-potential parameters directly (no linear mapping).

Regime: State A = x < 0 (left well), State B = x > 0 (right well).
"""

import numpy as np

SEED = 42

# ── Latent space ──
Z_DIM = 6       # dim 0=x, 1=z1, 2=z2, 3=z3, 4=u1, 5=u2
OBS_DIM = 30
OBS_DIM_ENC = 60

DIM_NAMES = ['x (reaction)', 'z1 (depth1)', 'z2 (depth2)',
             'z3 (barrier)', 'u1 (dw uncoupled)', 'u2 (tw uncoupled)']

# ── Langevin dynamics (Euler-Maruyama) ──
# dq = -(dt/gamma)*grad_U + sqrt(2*kT*dt/gamma)*noise
DT = 0.002
KT = 0.3
GAMMA = 1.0

# ── x potential: Gaussian double-well ──
# U_x = -depth1*G(x, x_L) - depth2*G(x, x_R) + barrier*G(x, 0) + k*x^4
X_LEFT = -1.0
X_RIGHT = 1.0
SIGMA_WELL = 0.45
SIGMA_BARRIER = 0.5
CONFINEMENT_K = 0.08

# ── z potentials: Gaussian double-wells at positive values ──
# z1, z2: wells at 0.5 and 1.5 (depth parameter range)
# z3: wells at 0.2 and 1.2 (barrier parameter range)
Z1_LEFT, Z1_RIGHT = 0.5, 1.5
Z2_LEFT, Z2_RIGHT = 0.5, 1.5
Z3_LEFT, Z3_RIGHT = 0.2, 1.2
Z_SIGMA_WELL = 0.25
Z_SIGMA_BARRIER = 0.3
Z_CONFINEMENT_K = 0.05
Z_DEPTH = 1.0          # well depth for all z potentials
Z_BARRIER_HEIGHT = 0.4  # barrier height for all z potentials

# ── u1: double-well (uncoupled) ──
U1_LEFT, U1_RIGHT = -1.0, 1.0
U1_DEPTH = 1.0
U1_BARRIER = 0.5
U1_SIGMA_WELL = 0.45
U1_SIGMA_BARRIER = 0.5
U1_CONFINEMENT_K = 0.08

# ── u2: TRIPLE well (uncoupled, 3-state) ──
U2_LEFT, U2_MID, U2_RIGHT = -1.0, 0.0, 1.0
U2_DEPTH = 1.0
U2_BARRIER = 0.5
U2_SIGMA_WELL = 0.35
U2_SIGMA_BARRIER = 0.25
U2_CONFINEMENT_K = 0.06

# ── Regime definition ──
X_THRESHOLD = 0.1
REGIME_DRIVING_FACTORS = [0, 1, 2]      # x, z1, z2
REGIME_INVARIANT_FACTORS = [3, 4, 5]    # z3, u1, u2
TRANSITION_DIMS = REGIME_DRIVING_FACTORS
NONTRANSITION_DIMS = REGIME_INVARIANT_FACTORS

# ── State assignment ──
BOUNDARY_BUFFER = 100   # discard frames near state boundaries

# ── TDRL format ──
LAG = 2
SUBSAMPLE_RATIO = 1.0

# ── Observation model (sparse additive mixing) ──
SUPPORT_SETS = {
    0: [0, 1, 2, 3, 4],           # x  → obs 0-4
    1: [5, 6, 7, 8],              # z1 → obs 5-8
    2: [9, 10, 11, 12, 4],        # z2 → obs 9-12, shared obs 4
    3: [13, 14, 15, 16, 8],       # z3 → obs 13-16, shared obs 8
    4: [17, 18, 19, 20, 21],      # u1 → obs 17-21
    5: [22, 23, 24, 25, 26, 21],  # u2 → obs 22-26, shared obs 21
}
NOISE_ONLY_OBS = [27, 28, 29]
MIXING_FUNC_NAMES = ['sin', 'tanh', 'cubic', 'sigmoid', 'softplus']
OBS_NOISE_STD = 0.05

# ── Interaction terms (for additivity ablation) ──
INTERACTION_ALPHA = 0.0
INTERACTION_PAIRS = {
    4:  (0, 2),
    8:  (1, 3),
    21: (4, 5),
}
PURE_INTERACTION_OBS = {
    27: (0, 1),
    28: (0, 3),
    29: (2, 4),
}

# ── Trajectories ──
# n_steps: raw Langevin steps (subsampled later for TDRL)
# subsample: keep every N-th frame → ~50K output frames per traj
N_STEPS = 2_000_000
SUBSAMPLE_STEP = 40     # 2M / 40 = 50K frames

TRAJECTORIES = [
    {'name': 'traj_1', 'n_steps': N_STEPS, 'init_state': 'A', 'seed': 42},
    {'name': 'traj_2', 'n_steps': N_STEPS, 'init_state': 'A', 'seed': 43},
    {'name': 'traj_3', 'n_steps': N_STEPS, 'init_state': 'A', 'seed': 44},
    {'name': 'traj_4', 'n_steps': N_STEPS, 'init_state': 'B', 'seed': 45},
    {'name': 'traj_5', 'n_steps': N_STEPS, 'init_state': 'A', 'seed': 46},
    {'name': 'traj_6', 'n_steps': N_STEPS, 'init_state': 'B', 'seed': 47},
    {'name': 'traj_7', 'n_steps': N_STEPS, 'init_state': 'A', 'seed': 48},
    {'name': 'traj_8', 'n_steps': N_STEPS, 'init_state': 'A', 'seed': 49},
    {'name': 'traj_9', 'n_steps': N_STEPS, 'init_state': 'B', 'seed': 50},
]
