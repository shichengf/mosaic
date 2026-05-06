"""mosaic.py — MOSAIC core LightningModule.

Implements the two-stage MOSAIC training described in the paper:
  Stage 1 — encoder + dense decoder + regime-conditioned temporal prior
            (NPChangeTransitionPrior_v2, ParallelMLP-based).
  Stage 2 — frozen encoder/prior, additive decoder x = Σ_j f_j(z_j) + b
            with column-entropy sparsity over the influence matrix.

ELBO computation matches the standard temporal-CRL ELBO; the parallel transition
prior gives the same log-det-Jacobian as the sequential reference (verified
numerically) but is two orders of magnitude faster on GPU.

The class name TimeVaryingProcess_v2 is kept for checkpoint backward-compatibility; MOSAIC is provided as an alias.
"""

import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch.distributions as D
from torch.nn import functional as F
from ..components.beta_v2 import BetaVAE_MLP_v2
from ..components.transition import MBDTransitionPrior
from ..components.transition_v2 import NPChangeTransitionPrior_v2
from sklearn.linear_model import LogisticRegression


class TimeVaryingProcess_v2(pl.LightningModule):
    """MOSAIC: temporal CRL with additive decoder and mechanism sparsity regularization.

    Identical to v1 TimeVaryingProcess except:
    - Decoder is AdditiveDecoder (x = Σ_j f_j(z_j))
    - training_step adds group-lasso sparsity loss on decoder
    - Supports freeze schedule for fine-tuning from v1 checkpoint
    """

    def __init__(
        self,
        input_dim,
        length,
        z_dim,
        lag,
        nclass,
        hidden_dim=128,
        embedding_dim=8,
        hidden_per_z=8,
        decoder_type='additive',
        trans_prior='NP',
        lr=1e-4,
        infer_mode='F',
        beta=0.0025,
        gamma=0.0075,
        lambda_sparse=1e-3,
        lambda_col_entropy=0.0,
        lambda_balance=0.0,
        lambda_diversity=0.0,
        lambda_slot_gate=0.0,
        lambda_regime_gate=0.0,
        lambda_trans_sparsity=0.0,
        lambda_col=0.0,
        lambda_row=0.0,
        use_slot_gate=False,
        sparsity_mode='w2',
        use_ard=False,
        ard_a=1.0,
        ard_b=0.1,
        warmup_epochs=5,
        rampup_epochs=20,
        freeze_epochs=5,
        decoder_dist='gaussian',
        correlation='Pearson'):
        super().__init__()
        assert trans_prior in ('L', 'NP')
        self.z_dim = z_dim
        self.lag = lag
        self.input_dim = input_dim
        self.lr = lr
        self.length = length
        self.beta = beta
        self.gamma = gamma
        self.lambda_col_entropy = lambda_col_entropy
        self.lambda_balance = lambda_balance
        self.lambda_sparse = lambda_sparse
        self.lambda_diversity = lambda_diversity
        self.lambda_slot_gate = lambda_slot_gate
        self.lambda_regime_gate = lambda_regime_gate
        self.lambda_trans_sparsity = lambda_trans_sparsity
        self.lambda_col = lambda_col
        self.lambda_row = lambda_row
        self.use_slot_gate = use_slot_gate
        self.use_ard = use_ard
        self.warmup_epochs = warmup_epochs
        self.rampup_epochs = rampup_epochs
        self.freeze_epochs = freeze_epochs
        self.correlation = correlation
        self.decoder_dist = decoder_dist
        self.decoder_type = decoder_type
        self.sparsity_mode = sparsity_mode
        self.infer_mode = infer_mode
        self.slot_gate_active_threshold = 0.2

        # Domain embeddings (identical to v1)
        self.embed_func = nn.Embedding(nclass, embedding_dim)

        # Encoder + Decoder
        regime_nclass = nclass if lambda_regime_gate > 0 else None
        if infer_mode == 'F':
            self.net = BetaVAE_MLP_v2(input_dim=input_dim,
                                      z_dim=z_dim,
                                      hidden_dim=hidden_dim,
                                      hidden_per_z=hidden_per_z,
                                      decoder_type=decoder_type,
                                      nclass=regime_nclass,
                                      use_slot_gate=use_slot_gate,
                                      use_ard=use_ard,
                                      ard_a=ard_a,
                                      ard_b=ard_b,
                                      sparsity_mode=sparsity_mode)
        elif infer_mode == 'R':
            raise NotImplementedError("v2 only supports infer_mode='F'")

        # Transition prior (v2: ParallelMLP + autograd.grad, ~10-50x faster)
        if trans_prior == 'L':
            self.transition_prior = MBDTransitionPrior(lags=lag,
                                                       latent_size=z_dim,
                                                       bias=False)
        elif trans_prior == 'NP':
            self.transition_prior = NPChangeTransitionPrior_v2(
                lags=lag,
                latent_size=z_dim,
                embedding_dim=embedding_dim,
                num_layers=3,
                hidden_dim=hidden_dim)

        self.register_buffer('base_dist_mean', torch.zeros(self.z_dim))
        self.register_buffer('base_dist_var', torch.eye(self.z_dim))

        # Save hyperparameters for checkpoint
        self.save_hyperparameters()

    @property
    def base_dist(self):
        return D.MultivariateNormal(self.base_dist_mean, self.base_dist_var)

    def reparameterize(self, mean, logvar, random_sampling=True):
        if random_sampling:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5 * logvar)
            return mean + eps * std
        else:
            return mean

    def reconstruction_loss(self, x, x_recon, distribution):
        batch_size = x.size(0)
        assert batch_size != 0
        if distribution == 'bernoulli':
            return F.binary_cross_entropy_with_logits(
                x_recon, x, size_average=False).div(batch_size)
        elif distribution == 'gaussian':
            return F.mse_loss(x_recon, x, size_average=False).div(batch_size)
        elif distribution == 'sigmoid_gaussian':
            x_recon = F.sigmoid(x_recon)
            return F.mse_loss(x_recon, x, size_average=False).div(batch_size)

    def forward(self, batch):
        x, y, c = batch['xt'], batch['yt'], batch['ct']
        batch_size, length, _ = x.shape
        x_flat = x.view(-1, self.input_dim)
        _, mus, logvars, zs = self.net(x_flat)
        return zs, mus, logvars

    # ------------------------------------------------------------------
    # Lambda schedule: warmup → ramp → full
    # ------------------------------------------------------------------
    def get_current_lambda(self):
        epoch = self.current_epoch
        if epoch < self.warmup_epochs:
            return 0.0
        elif epoch < self.warmup_epochs + self.rampup_epochs:
            progress = (epoch - self.warmup_epochs) / max(self.rampup_epochs, 1)
            return self.lambda_sparse * progress
        else:
            return self.lambda_sparse

    # ------------------------------------------------------------------
    # Freeze schedule: decoder-only → joint
    # ------------------------------------------------------------------
    def on_train_epoch_start(self):
        epoch = self.current_epoch
        if epoch < self.freeze_epochs:
            # Freeze everything except decoder
            for name, param in self.named_parameters():
                if 'net.decoder' in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
        elif epoch == self.freeze_epochs:
            # Unfreeze all
            for param in self.parameters():
                param.requires_grad_(True)
            print(f"[Epoch {epoch}] Unfreezing encoder + transition prior")

    # ------------------------------------------------------------------
    # ELBO computation (identical to v1, factored out for reuse)
    # ------------------------------------------------------------------
    def _compute_elbo(self, batch):
        """Compute ELBO components. Returns dict of losses and intermediate tensors."""
        x, y, c = batch['xt'], batch['yt'], batch['ct']
        c = torch.squeeze(c).to(torch.int64)
        batch_size, length, _ = x.shape
        x_flat = x.view(-1, self.input_dim)
        embeddings = self.embed_func(c)

        # Inference (with regime gating if enabled)
        if self.lambda_regime_gate > 0:
            c_flat = c.unsqueeze(1).expand(-1, length).reshape(-1).long()
            x_recon, mus, logvars, zs = self.net(x_flat, regime_id=c_flat)
        else:
            x_recon, mus, logvars, zs = self.net(x_flat)

        # Reshape
        x_recon = x_recon.view(batch_size, length, self.input_dim)
        mus = mus.reshape(batch_size, length, self.z_dim)
        logvars = logvars.reshape(batch_size, length, self.z_dim)
        zs = zs.reshape(batch_size, length, self.z_dim)

        # Reconstruction loss
        recon_loss = self.reconstruction_loss(
            x[:, :self.lag], x_recon[:, :self.lag], self.decoder_dist
        ) + self.reconstruction_loss(
            x[:, self.lag:], x_recon[:, self.lag:], self.decoder_dist
        ) / (length - self.lag)

        # Past KLD
        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(zs)
        if self.use_ard:
            prior_var = self.net.get_ard_prior_variance()  # (z_dim,)
            prior_std = torch.sqrt(prior_var).unsqueeze(0).unsqueeze(0)
            prior_std = prior_std.expand_as(mus[:, :self.lag])
            p_dist = D.Normal(torch.zeros_like(mus[:, :self.lag]), prior_std)
        else:
            p_dist = D.Normal(torch.zeros_like(mus[:, :self.lag]),
                              torch.ones_like(logvars[:, :self.lag]))
        log_pz_normal = torch.sum(torch.sum(
            p_dist.log_prob(zs[:, :self.lag]), dim=-1), dim=-1)
        log_qz_normal = torch.sum(torch.sum(
            log_qz[:, :self.lag], dim=-1), dim=-1)
        kld_normal = (log_qz_normal - log_pz_normal).mean()

        # Future KLD
        log_qz_laplace = log_qz[:, self.lag:]
        sum_log_abs_det_jacobians = 0
        residuals, logabsdet = self.transition_prior(zs, embeddings)
        sum_log_abs_det_jacobians += logabsdet
        log_pz_laplace = (torch.sum(self.base_dist.log_prob(residuals), dim=1)
                          + sum_log_abs_det_jacobians)
        kld_laplace = (torch.sum(torch.sum(log_qz_laplace, dim=-1), dim=-1)
                       - log_pz_laplace) / (length - self.lag)
        kld_laplace = kld_laplace.mean()

        return {
            'recon_loss': recon_loss,
            'kld_normal': kld_normal,
            'kld_laplace': kld_laplace,
            'mus': mus,
            'c': c,
            'residuals': residuals,  # (batch, length-lag, z_dim)
        }

    # ------------------------------------------------------------------
    # Transition-aware sparsity (Feature 2)
    # ------------------------------------------------------------------
    def _transition_aware_sparsity(self, residuals):
        """Modulate group-lasso per z-dim based on transition residuals.

        z-dims with large residuals = "changing" across regimes → less sparsity
        z-dims with small residuals = "stable" → more sparsity
        """
        with torch.no_grad():
            # residuals: (batch, T-L, z_dim) → per-z change score
            z_change = residuals.detach().abs().mean(dim=(0, 1))  # (z_dim,)
            z_change = z_change / (z_change.max() + 1e-8)
        # stable z (z_change~0) → weight~1.0, changing z (z_change~1) → weight~0.3
        sparsity_weight = 1.0 - 0.7 * z_change  # (z_dim,)
        influence = self.net.decoder._get_influence_vector()  # (z_dim, output_dim)
        weighted = influence * sparsity_weight.unsqueeze(1)
        return weighted.mean()

    # ------------------------------------------------------------------
    # training_step: ELBO + sparsity
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        r = self._compute_elbo(batch)

        elbo_loss = (r['recon_loss']
                     + self.beta * r['kld_normal']
                     + self.gamma * r['kld_laplace'])

        # Decoder sparsity (group-lasso on AdditiveDecoder output layer)
        lam = self.get_current_lambda()
        sparse_loss = self.net.decoder.sparsity_loss()
        loss = elbo_loss + lam * sparse_loss

        # Column entropy: push each x to be dominated by few z's
        if self.lambda_col_entropy > 0 and hasattr(self.net.decoder, 'column_entropy_loss'):
            col_ent_loss = self.net.decoder.column_entropy_loss()
            loss = loss + lam * self.lambda_col_entropy * col_ent_loss
            self.log("train_col_entropy", col_ent_loss)

        # Balance loss: prevent z-collapse (penalize unequal z utilization)
        if self.lambda_balance > 0 and hasattr(self.net.decoder, 'balance_loss'):
            bal_loss = self.net.decoder.balance_loss()
            loss = loss + lam * self.lambda_balance * bal_loss
            self.log("train_balance_loss", bal_loss)

        # Diversity loss: penalize z-dims with similar influence profiles (Feature 3)
        if self.lambda_diversity > 0 and hasattr(self.net.decoder, 'diversity_loss'):
            div_loss = self.net.decoder.diversity_loss()
            loss = loss + lam * self.lambda_diversity * div_loss
            self.log("train_diversity_loss", div_loss)

        if self.lambda_slot_gate > 0 and hasattr(self.net.decoder, 'slot_gate_loss'):
            slot_gate_loss = self.net.decoder.slot_gate_loss()
            loss = loss + lam * self.lambda_slot_gate * slot_gate_loss
            self.log("train_slot_gate_loss", slot_gate_loss)

        # Regime gate sparsity: encourage sparse regime-dependent gating (Feature 1)
        if self.lambda_regime_gate > 0 and hasattr(self.net.decoder, 'regime_gate_loss'):
            rg_loss = self.net.decoder.regime_gate_loss()
            loss = loss + lam * self.lambda_regime_gate * rg_loss
            self.log("train_regime_gate_loss", rg_loss)

        # Column exclusivity: each x_i dominated by one z
        # Uses its own warmup schedule independent of lambda_sparse
        if self.lambda_col > 0 and hasattr(self.net.decoder, 'col_exclusivity_loss'):
            epoch = self.current_epoch
            if epoch < self.warmup_epochs:
                lam_col = 0.0
            elif epoch < self.warmup_epochs + self.rampup_epochs:
                progress = (epoch - self.warmup_epochs) / max(self.rampup_epochs, 1)
                lam_col = self.lambda_col * progress
            else:
                lam_col = self.lambda_col
            col_loss = self.net.decoder.col_exclusivity_loss()
            loss = loss + lam_col * col_loss
            self.log("train_col_excl", col_loss)
            self.log("train_lam_col", lam_col)

        # Row L1: weak penalty on total influence per z
        if self.lambda_row > 0 and hasattr(self.net.decoder, 'row_l1_loss'):
            epoch = self.current_epoch
            if epoch < self.warmup_epochs:
                lam_row = 0.0
            elif epoch < self.warmup_epochs + self.rampup_epochs:
                progress = (epoch - self.warmup_epochs) / max(self.rampup_epochs, 1)
                lam_row = self.lambda_row * progress
            else:
                lam_row = self.lambda_row
            row_loss = self.net.decoder.row_l1_loss()
            loss = loss + lam_row * row_loss
            self.log("train_row_l1", row_loss)

        # Transition-aware sparsity: modulate group-lasso by transition residuals (Feature 2)
        if self.lambda_trans_sparsity > 0:
            trans_sp_loss = self._transition_aware_sparsity(r['residuals'])
            loss = loss + lam * self.lambda_trans_sparsity * trans_sp_loss
            self.log("train_trans_sparsity_loss", trans_sp_loss)

        # ARD hyperprior KL
        if self.use_ard:
            ard_kl = self.net.get_ard_kl_hyperprior()
            loss = loss + self.beta * ard_kl
            self.log("train_ard_kl", ard_kl)
            with torch.no_grad():
                var = self.net.get_ard_prior_variance()
                self.log("train_ard_min_var", var.min())
                self.log("train_ard_max_var", var.max())
                self.log("train_ard_alive", float((var > 0.01).sum()))

        gate_stats = None
        if hasattr(self.net.decoder, 'slot_gate_stats'):
            gate_stats = self.net.decoder.slot_gate_stats(
                threshold=self.slot_gate_active_threshold)
        if gate_stats is not None:
            self.log("train_gate_mean", gate_stats['mean'])
            self.log("train_gate_active", gate_stats['active'])

        self.log("train_elbo_loss", elbo_loss)
        self.log("train_recon_loss", r['recon_loss'])
        self.log("train_kld_normal", r['kld_normal'])
        self.log("train_kld_laplace", r['kld_laplace'])
        self.log("train_sparse_loss", sparse_loss)
        self.log("train_lambda", lam)
        self.log("train_total_loss", loss)
        return loss

    # ------------------------------------------------------------------
    # validation_step: ELBO + regime accuracy + sparsity stats
    # ------------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        r = self._compute_elbo(batch)

        elbo_loss = (r['recon_loss']
                     + self.beta * r['kld_normal']
                     + self.gamma * r['kld_laplace'])

        sparse_loss = self.net.decoder.sparsity_loss()

        # Regime classification accuracy
        zt_mean = r['mus'].mean(dim=1).detach().cpu().numpy()
        ct_true = r['c'].detach().cpu().numpy()
        regime_acc = 0.0
        if len(np.unique(ct_true)) > 1:
            try:
                clf = LogisticRegression(max_iter=200, solver='lbfgs')
                clf.fit(zt_mean, ct_true)
                regime_acc = float(clf.score(zt_mean, ct_true))
            except Exception:
                regime_acc = 0.0

        # Sparsity statistics (only for additive decoder)
        influence = self.net.decoder.get_influence_matrix()
        if influence is not None:
            dominant_z = influence.argmax(axis=1)
            n_active_z = len(np.unique(dominant_z))
            row_max = influence.max(axis=1)
            row_sum = influence.sum(axis=1)
            mean_concentration = float(np.mean(row_max / (row_sum + 1e-10)))
        else:
            n_active_z = self.z_dim
            mean_concentration = 0.0

        self.log("val_regime_acc", regime_acc)
        self.log("val_elbo_loss", elbo_loss)
        self.log("val_recon_loss", r['recon_loss'])
        self.log("val_kld_normal", r['kld_normal'])
        self.log("val_kld_laplace", r['kld_laplace'])
        self.log("val_sparse_loss", sparse_loss)
        self.log("val_active_z", float(n_active_z))
        self.log("val_concentration", mean_concentration)

        if hasattr(self.net.decoder, 'diversity_loss'):
            with torch.no_grad():
                self.log("val_diversity_loss", self.net.decoder.diversity_loss())

        if hasattr(self.net.decoder, 'slot_gate_stats'):
            gate_stats = self.net.decoder.slot_gate_stats(
                threshold=self.slot_gate_active_threshold)
            if gate_stats is not None:
                self.log("val_gate_mean", gate_stats['mean'])
                self.log("val_gate_active", gate_stats['active'])

        return elbo_loss

    @torch.inference_mode(False)
    def test_step(self, batch, batch_idx):
        # transition_prior needs autograd for Jacobian computation
        # Clone tensors to escape inference_mode context from PL test loop
        batch = {k: v.clone().requires_grad_(True) if v.is_floating_point() else v.clone()
                 for k, v in batch.items()}
        r = self._compute_elbo(batch)

        elbo_loss = (r['recon_loss']
                     + self.beta * r['kld_normal']
                     + self.gamma * r['kld_laplace'])

        sparse_loss = self.net.decoder.sparsity_loss()

        zt_mean = r['mus'].mean(dim=1).detach().cpu().numpy()
        ct_true = r['c'].detach().cpu().numpy()
        regime_acc = 0.0
        if len(np.unique(ct_true)) > 1:
            try:
                clf = LogisticRegression(max_iter=200, solver='lbfgs')
                clf.fit(zt_mean, ct_true)
                regime_acc = float(clf.score(zt_mean, ct_true))
            except Exception:
                regime_acc = 0.0

        influence = self.net.decoder.get_influence_matrix()
        if influence is not None:
            dominant_z = influence.argmax(axis=1)
            n_active_z = len(np.unique(dominant_z))
            row_max = influence.max(axis=1)
            row_sum = influence.sum(axis=1)
            mean_concentration = float(np.mean(row_max / (row_sum + 1e-10)))
        else:
            n_active_z = self.z_dim
            mean_concentration = 0.0

        self.log("test_regime_acc", regime_acc)
        self.log("test_elbo_loss", elbo_loss)
        self.log("test_recon_loss", r['recon_loss'])
        self.log("test_kld_normal", r['kld_normal'])
        self.log("test_kld_laplace", r['kld_laplace'])
        self.log("test_sparse_loss", sparse_loss)
        self.log("test_active_z", float(n_active_z))
        self.log("test_concentration", mean_concentration)

        return elbo_loss

    def configure_optimizers(self):
        # Only optimize parameters that require gradients
        trainable = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            trainable,
            lr=self.lr,
            betas=(0.9, 0.999),
            weight_decay=0.0001
        )
        return [opt], []

    # ------------------------------------------------------------------
    # Load v1 weights (encoder + transition_prior + embed_func)
    # ------------------------------------------------------------------
    def load_v1_weights(self, v1_ckpt_path):
        """Load weights from a v1 checkpoint.

        - Encoder: direct copy (identical architecture)
        - Embed_func: direct copy
        - Transition prior: convert 64 separate MLPs → ParallelMLP stacked format
        - Decoder: skipped (architecture changed to AdditiveDecoder)
        """
        v1_ckpt = torch.load(v1_ckpt_path, map_location='cpu', weights_only=False)
        v1_state = v1_ckpt['state_dict']

        # Step 1: Load encoder + embed_func (direct match)
        v2_state = self.state_dict()
        direct_loaded, skipped = [], []
        for key, value in v1_state.items():
            # Skip transition_prior (handled separately) and decoder
            if key.startswith('transition_prior.'):
                continue
            if key in v2_state and v2_state[key].shape == value.shape:
                v2_state[key] = value
                direct_loaded.append(key)
            else:
                skipped.append(key)
        self.load_state_dict(v2_state, strict=False)

        # Step 2: Load transition prior (convert v1 separate MLPs → v2 ParallelMLP)
        tp_loaded = 0
        if hasattr(self.transition_prior, 'load_v1_weights'):
            tp_loaded = self.transition_prior.load_v1_weights(v1_state)

        print(f"\n{'='*60}")
        print(f"v1 → v2 weight loading:")
        print(f"  Direct loaded:      {len(direct_loaded)} params (encoder + embed)")
        print(f"  Transition convert: {tp_loaded} params (64 MLPs → ParallelMLP)")
        print(f"  Skipped:            {len(skipped)} params (decoder changed)")
        print(f"{'='*60}")
        for k in skipped:
            print(f"  SKIP: {k}")
        print()


# Alias used in the paper; checkpoints retain the original class name above.
MOSAIC = TimeVaryingProcess_v2
