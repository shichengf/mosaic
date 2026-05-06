"""beta_v2.py — BetaVAE with Additive Decoder for structural sparsity

v2 of beta.py. The encoder is IDENTICAL to v1 (BetaVAE_MLP).
The decoder is replaced with AdditiveDecoder: x = Σ_j f_j(z_j) + bias,
where each f_j is an independent sub-network R^1 → R^output_dim.

This enables:
  - Exact decomposition of which z_j generates which x_i
  - Group-lasso sparsity on output layer → structural zeros
  - Direct module assignment without post-hoc Jacobian analysis

Theory: Lachapelle et al., "Disentanglement via Mechanism Sparsity
Regularization", CLeaR 2022. Sparse mixing g(z) guarantees identifiability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
import math

# Reuse from v1
from .beta import reparametrize, kaiming_init


class AdditiveDecoder(nn.Module):
    """Additive decoder: x = Σ_j f_j(z_j) + bias.

    Each z_j has an independent sub-network:
        z_j (scalar) → Linear(1, hidden_per_z) → LeakyReLU → Linear(hidden_per_z, output_dim)

    When hidden_per_z=0, uses a linear decoder: f_j(z_j) = W_j * z_j (no hidden layer).

    Implemented as vectorized operations (no Python loop over z_dim).

    Parameters
    ----------
    z_dim : int
        Number of latent dimensions.
    hidden_per_z : int
        Hidden units per sub-decoder. 0 = linear decoder.
    output_dim : int
        Output dimension (e.g. 658 for sin/cos encoded angles).
    sparsity_mode : str
        'w2' = entropy on ||W2|| norms (original).
        'functional' = entropy on |f_j(+1) - f_j(-1)| (functional effect).
    """

    def __init__(self, z_dim, hidden_per_z, output_dim, nclass=None,
                 use_slot_gate=False, sparsity_mode='w2'):
        super().__init__()
        self.z_dim = z_dim
        self.hidden_per_z = hidden_per_z
        self.output_dim = output_dim
        self.use_slot_gate = use_slot_gate
        self.sparsity_mode = sparsity_mode
        self.is_linear = (hidden_per_z == 0)

        if self.is_linear:
            # Linear decoder: f_j(z_j) = W_lin[j] * z_j
            # W_lin shape: (z_dim, output_dim)
            self.W_lin = nn.Parameter(torch.empty(z_dim, output_dim))
            nn.init.kaiming_normal_(self.W_lin)
            # Alias W2 for compatibility with influence matrix / legacy code
            self.W1 = None
            self.b1 = None
            self.W2 = None
        else:
            # Layer 1: z_j (scalar) → hidden, independently per z_j
            self.W1 = nn.Parameter(torch.empty(z_dim, hidden_per_z))
            self.b1 = nn.Parameter(torch.zeros(z_dim, hidden_per_z))
            # Layer 2: hidden → output, independently per z_j
            self.W2 = nn.Parameter(torch.empty(z_dim, output_dim, hidden_per_z))
            self.W_lin = None

        # Global output bias (shared across all z_j)
        self.bias = nn.Parameter(torch.zeros(output_dim))

        self.act = nn.LeakyReLU(0.2)

        # Regime-dependent gating
        if nclass is not None:
            self.regime_gate = nn.Parameter(torch.zeros(nclass, z_dim))
        else:
            self.regime_gate = None

        if use_slot_gate:
            self.slot_gate_logits = nn.Parameter(torch.zeros(z_dim))
        else:
            self.slot_gate_logits = None

        if not self.is_linear:
            self._init_weights()

    def _init_weights(self):
        """Kaiming initialization adapted for parallel sub-networks."""
        a = 0.2  # LeakyReLU negative slope
        std1 = math.sqrt(2.0 / (1.0 * (1 + a ** 2)))
        nn.init.normal_(self.W1, 0, std1)
        std2 = math.sqrt(2.0 / (self.hidden_per_z * (1 + a ** 2)))
        nn.init.normal_(self.W2, 0, std2)

    def _forward_per_z(self, z):
        """Compute per-z contributions (before summing).

        Returns
        -------
        out : Tensor (batch, z_dim, output_dim)
        """
        if self.is_linear:
            # (batch, z_dim) → (batch, z_dim, 1) * (z_dim, output_dim) → (batch, z_dim, output_dim)
            out = z.unsqueeze(-1) * self.W_lin.unsqueeze(0)
        else:
            h = z.unsqueeze(-1) * self.W1.unsqueeze(0) + self.b1.unsqueeze(0)
            h = self.act(h)
            out = torch.einsum('bzh,zoh->bzo', h, self.W2)
        return out

    def forward(self, z, regime_id=None):
        """
        Parameters
        ----------
        z : Tensor (batch, z_dim)
        regime_id : Tensor (batch,) int, optional

        Returns
        -------
        x : Tensor (batch, output_dim)
        """
        out = self._forward_per_z(z)

        # Regime-dependent gating
        if regime_id is not None and self.regime_gate is not None:
            gate = torch.sigmoid(self.regime_gate[regime_id])
            out = out * gate.unsqueeze(-1)

        if self.slot_gate_logits is not None:
            slot_gate = torch.sigmoid(self.slot_gate_logits)
            out = out * slot_gate.view(1, self.z_dim, 1)

        x = out.sum(dim=1) + self.bias
        return x

    def slot_gate_values(self):
        if self.slot_gate_logits is None:
            return None
        return torch.sigmoid(self.slot_gate_logits)

    def slot_gate_loss(self):
        gate = self.slot_gate_values()
        if gate is None:
            return torch.tensor(0.0, device=self.bias.device)
        return gate.mean()

    def slot_gate_stats(self, threshold=0.2):
        gate = self.slot_gate_values()
        if gate is None:
            return None
        return {
            'values': gate.detach(),
            'mean': gate.mean(),
            'active': (gate > threshold).float().sum(),
        }

    def _get_influence_vector(self):
        """Get per-(z,x) influence magnitude used for sparsity.

        Returns (z_dim, output_dim) tensor of non-negative influence values.
        """
        if self.sparsity_mode in ('functional', 'func_l1'):
            return self._functional_influence()
        else:
            return self._w2_influence()

    def _w2_influence(self):
        """W2-norm based influence (original)."""
        if self.is_linear:
            return self.W_lin.abs()  # (z_dim, output_dim)
        return self.W2.norm(dim=-1)  # (z_dim, output_dim)

    def _functional_influence(self):
        """Functional influence: |f_j(+1) - f_j(-1)| per (z_j, x_i).

        Two forward passes through each sub-decoder at fixed probe points.
        Captures true functional effect including hidden-layer nonlinearity.
        """
        device = self.bias.device
        # Probe at +1 and -1 for all z_j simultaneously
        z_pos = torch.ones(1, self.z_dim, device=device)   # (1, z_dim)
        z_neg = -torch.ones(1, self.z_dim, device=device)  # (1, z_dim)

        out_pos = self._forward_per_z(z_pos).squeeze(0)  # (z_dim, output_dim)
        out_neg = self._forward_per_z(z_neg).squeeze(0)  # (z_dim, output_dim)

        return (out_pos - out_neg).abs()  # (z_dim, output_dim)

    def sparsity_loss(self):
        """Sparsity loss on decoder support.

        Modes:
        - 'w2' / 'functional': entropy of normalized influence (original)
        - 'func_l1': mean L1 of functional influence (simpler, better scaled)
        - 'group_lasso': sum_j ||A[:, j]||_2 = sum of per-column L2 norms.
          Canonical group-lasso (Yuan & Lin, 2006). No alive-mask, no
          normalization — penalizes magnitude directly, not shape.
        """
        influence = self._get_influence_vector()  # (z_dim, output_dim)
                                                  # rows = j, cols = obs i;
                                                  # so A[:, j] in user notation
                                                  # is influence[j, :].

        if self.sparsity_mode == 'func_l1':
            # L1: simply penalize the mean absolute functional effect.
            # Unlike entropy, this scales linearly with the actual effect magnitude,
            # so it doesn't saturate at 0 or explode at log(D).
            return influence.mean()

        if self.sparsity_mode == 'group_lasso':
            # Sum_j sqrt(sum_i A[i,j]^2). With influence shaped (z_dim, D),
            # the norm-over-D is dim=1; .sum() over j.
            column_norms = torch.norm(influence, p=2, dim=1)   # (z_dim,)
            return column_norms.sum()

        # Entropy mode (w2 or functional)
        z_total = influence.sum(dim=1)
        alive_mask = z_total > 0.01 * z_total.max()
        if alive_mask.sum() == 0:
            return torch.tensor(0.0, device=self.bias.device)

        alive_inf = influence[alive_mask]
        alive_total = z_total[alive_mask].unsqueeze(1)

        p = alive_inf / (alive_total + 1e-8)
        entropy = -(p * torch.log(p + 1e-8)).sum(dim=1)
        return entropy.mean()

    def sparsity_loss_legacy(self):
        """Original group-lasso penalty (mean of all group norms)."""
        influence = self._get_influence_vector()
        return influence.mean()

    def col_exclusivity_loss(self):
        """Column-wise exclusivity: each x_i should be dominated by ONE z.

        For each x_i, normalize influence across z's and compute entropy.
        Low entropy = x_i claimed by one z (good).
        High entropy = x_i spread across many z's (bad).

        Key difference from row-wise entropy (sparsity_loss):
        - Row-wise: "each z controls few x" → pushes z to control 1 variable
        - Column-wise: "each x belongs to one z" → z can claim entire module
        """
        influence = self._functional_influence()  # (z_dim, output_dim)
        # Normalize per column (per x_i)
        p = influence / (influence.sum(dim=0, keepdim=True) + 1e-8)  # (z_dim, output_dim)
        entropy = -(p * torch.log(p + 1e-8)).sum(dim=0)  # (output_dim,)
        return entropy.mean()

    def row_l1_loss(self):
        """Row-wise L1: weak penalty on total influence per z.

        Prevents one z from claiming all variables. Not entropy (which
        pushes toward single-variable), just total magnitude.
        """
        influence = self._functional_influence()  # (z_dim, output_dim)
        return influence.sum(dim=1).mean()  # mean over z of total influence

    def balance_loss(self):
        """Penalize unequal z utilization to prevent z-collapse."""
        influence = self._get_influence_vector()
        z_importance = influence.sum(dim=1)
        z_norm = z_importance / (z_importance.mean() + 1e-8)
        return z_norm.var()

    def regime_gate_loss(self):
        """Binary gates + cross-regime differentiation.

        Pushes gates toward 0 or 1 (binary_loss) and encourages different
        regimes to use different z-dims (diff_loss).

        Returns
        -------
        loss : scalar Tensor
        """
        if self.regime_gate is None:
            return torch.tensor(0.0, device=self.W2.device)
        g = torch.sigmoid(self.regime_gate)  # (nclass, z_dim)
        binary_loss = (g * (1 - g)).mean()
        diff_loss = -torch.abs(g[0] - g[1]).mean()
        return binary_loss + diff_loss

    def column_entropy_loss(self):
        """Column-wise entropy: encourages each x_i to be dominated by few z's."""
        influence = self._get_influence_vector()
        p = influence / (influence.sum(dim=0, keepdim=True) + 1e-8)
        entropy = -(p * torch.log(p + 1e-8)).sum(dim=0)
        return entropy.mean()

    def diversity_loss(self):
        """Penalize z-dims with similar influence profiles (DPP-inspired)."""
        influence = self._get_influence_vector()
        z_profiles = influence / (influence.norm(dim=1, keepdim=True) + 1e-8)
        sim = z_profiles @ z_profiles.T
        eye = torch.eye(self.z_dim, device=sim.device)
        off_diag = sim * (1 - eye)
        return off_diag.pow(2).sum() / (self.z_dim * (self.z_dim - 1))

    def get_influence_matrix(self, mode=None):
        """Return the z→x influence matrix for module assignment.

        Parameters
        ----------
        mode : str or None
            'w2', 'functional', or None (use self.sparsity_mode).

        Returns
        -------
        influence : ndarray (output_dim, z_dim)
        """
        with torch.no_grad():
            if mode == 'functional' or (mode is None and self.sparsity_mode == 'functional'):
                influence = self._functional_influence()
            else:
                influence = self._w2_influence()
            gate = self.slot_gate_values()
            if gate is not None:
                influence = influence * gate.unsqueeze(1)
            return influence.T.cpu().numpy()


class DenseDecoder(nn.Module):
    """Standard MLP decoder (v1-style) for ablation comparison.

    Architecture: z → LeakyReLU → Linear → LeakyReLU → Linear → LeakyReLU → Linear → x

    Provides the same interface as AdditiveDecoder (sparsity_loss, get_influence_matrix)
    so it can be used as a drop-in replacement in TimeVaryingProcess_v2.
    """

    def __init__(self, z_dim, hidden_dim, output_dim):
        super().__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Linear(z_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

        for m in self.net:
            kaiming_init(m)

    def forward(self, z):
        return self.net(z)

    def sparsity_loss(self):
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def get_influence_matrix(self):
        return None


class BetaVAE_MLP_v2(nn.Module):
    """BetaVAE with configurable decoder (additive or dense).

    Encoder: IDENTICAL to v1 BetaVAE_MLP (4-layer MLP → mu, logvar).
    Decoder: AdditiveDecoder (default) or DenseDecoder (for ablation).

    The forward() interface is the same as v1 for drop-in compatibility.
    """

    def __init__(self, input_dim=3, z_dim=10, hidden_dim=128, hidden_per_z=8,
                 decoder_type='additive', nclass=None, use_slot_gate=False,
                 use_ard=False, ard_a=1.0, ard_b=0.1, sparsity_mode='w2'):
        super().__init__()
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.decoder_type = decoder_type
        self.use_ard = use_ard
        self.ard_a = ard_a
        self.ard_b = ard_b

        # ARD: learnable log-precision per z-dim
        if use_ard:
            self.log_alpha = nn.Parameter(torch.zeros(z_dim))  # log(1)=0 → unit var
        else:
            self.log_alpha = None

        # Encoder: IDENTICAL to v1 (same architecture, same parameter names)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 2 * z_dim)
        )

        # Decoder: additive (default) or dense (ablation)
        if decoder_type == 'additive':
            self.decoder = AdditiveDecoder(
                z_dim=z_dim,
                hidden_per_z=hidden_per_z,
                output_dim=input_dim,
                nclass=nclass,
                use_slot_gate=use_slot_gate,
                sparsity_mode=sparsity_mode
            )
        elif decoder_type == 'dense':
            self.decoder = DenseDecoder(
                z_dim=z_dim,
                hidden_dim=hidden_dim,
                output_dim=input_dim
            )
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

        # Init encoder with kaiming
        for m in self.encoder:
            kaiming_init(m)

    def forward(self, x, regime_id=None, return_z=True):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z, regime_id=regime_id)

        if return_z:
            return x_recon, mu, logvar, z
        else:
            return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z, regime_id=None):
        if regime_id is not None and hasattr(self.decoder, 'regime_gate'):
            return self.decoder(z, regime_id=regime_id)
        return self.decoder(z)

    def get_ard_prior_variance(self):
        """Return per-dim prior variance: 1/alpha."""
        if not self.use_ard:
            return torch.ones(self.z_dim)
        alpha = torch.exp(self.log_alpha).clamp(max=1e6)
        return 1.0 / (alpha + 1e-8)

    def get_ard_kl_hyperprior(self):
        """KL between learned alpha and Gamma(a, b) hyperprior (point estimate)."""
        if not self.use_ard:
            return torch.tensor(0.0)
        alpha = torch.exp(self.log_alpha).clamp(max=1e6)
        log_alpha_clamped = torch.log(alpha)
        nll = -(self.ard_a - 1) * log_alpha_clamped + self.ard_b * alpha
        return nll.sum()
