#!/usr/bin/env python3
"""Run CtrlNS on our synthetic v2 data."""

import argparse
from pathlib import Path

import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split
from datasets.npz_dataset import NpzWindowDataset

# Import model class only — avoid module-level training code in train_nsctrl.py
# by reimporting the necessary components directly
import torch
from torch import nn
import torch.distributions as tD
import numpy as np
from models.metrics.correlation import compute_mcc
from models.metrics.hmm_metrics import compute_acc
from models.utils import JacobianMLP, MLP, BetaVAE_MLP, NPChangeTransitionPrior

_REPO = Path(__file__).resolve().parents[2]


class SparseX(pl.LightningModule):
    def __init__(self, n_class, x_dim, z_dim, lags, hidden_dim=128,
                 embedding_dim=2, alpha=1.0, beta=0.002, gamma=0.02,
                 lr=1e-3, correlation="Pearson", weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.lags = lags
        self.n_class = n_class
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.correlation = correlation
        self.weight_decay = weight_decay
        self.jacobian_mlps = nn.ModuleList([
            JacobianMLP(jacobian_support=np.eye(z_dim), hid_dim=hidden_dim)
            for _ in range(n_class)
        ])
        self.gating = MLP(z_dim * (lags + 1), hidden_dim, n_class, depth=1)
        self.criteria = nn.MSELoss()
        self.net = BetaVAE_MLP(x_dim, z_dim, hidden_dim, leaky_relu_slope=0.2)
        self.c_embeddings = nn.Embedding(n_class, embedding_dim)
        self.transition_prior = NPChangeTransitionPrior(
            lags=lags, latent_size=z_dim, embedding_dim=embedding_dim,
            num_layers=2, hidden_dim=hidden_dim,
        )
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.register_buffer("base_dist_mean", torch.zeros(z_dim))
        self.register_buffer("base_dist_var", torch.eye(z_dim))
        self.validation_step_outputs = []

    @property
    def base_dist(self):
        return tD.MultivariateNormal(self.base_dist_mean, self.base_dist_var)

    def tdrl_loss(self, mus, logvars, z_est, c_est):
        _, lags_and_length, _ = mus.shape
        # c_est may be (batch, length) — take last step for embedding
        if c_est.dim() > 1:
            c_est = c_est[:, -1]
        embeddings = self.c_embeddings(c_est)
        q_dist = tD.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(z_est)

        p_dist = tD.Normal(
            torch.zeros_like(mus[:, :self.lags]),
            torch.ones_like(logvars[:, :self.lags]),
        )
        log_pz_past = torch.sum(torch.sum(p_dist.log_prob(z_est[:, :self.lags]), dim=-1), dim=-1)
        log_qz_past = torch.sum(torch.sum(log_qz[:, :self.lags], dim=-1), dim=-1)
        past_kld = (log_qz_past - log_pz_past).mean()

        z_est_future = z_est[:, self.lags:]
        log_qz_future = log_qz[:, self.lags:]
        residuals, logabsdet = self.transition_prior(z_est, embeddings)
        log_pz_future = torch.sum(self.base_dist.log_prob(residuals), dim=1) + logabsdet
        future_kld = (torch.sum(torch.sum(log_qz_future, dim=-1), dim=-1) - log_pz_future).mean()

        return past_kld, future_kld

    def training_step(self, batch, batch_idx):
        x, z, c = batch
        batch_size, lags_and_length, _ = x.shape
        length = lags_and_length - self.lags

        x_recon, mus, logvars, z_est = self.net(x)
        recon_loss = self.criteria(x_recon, x)

        z_hat = mus
        z_est_pair = (
            z_hat.unfold(dimension=1, size=self.lags + 1, step=1)
            .transpose(-2, -1)
            .reshape(batch_size, length, -1)
        )
        C_z_pair = self.gating(z_est_pair)
        C_z_pair = torch.nn.functional.gumbel_softmax(C_z_pair, tau=1, hard=True)
        z_prev = z_hat[:, :-self.lags]
        z_curr = z_hat[:, self.lags:]
        z_hat_curr = torch.zeros_like(z_curr)
        for i in range(self.n_class):
            z_hat_curr += C_z_pair[:, :, i].unsqueeze(-1) * self.jacobian_mlps[i](z_prev)
        transition_loss = self.criteria(z_hat_curr, z_curr)

        c_est = C_z_pair.argmax(dim=-1)
        past_kld, future_kld = self.tdrl_loss(mus, logvars, z_est, c_est)

        loss = (recon_loss + self.alpha * transition_loss
                + self.beta * past_kld + self.gamma * future_kld)

        self.log_dict({
            "train/loss": loss,
            "train/recon_loss": recon_loss,
            "train/transition_loss": transition_loss,
        })
        return loss

    def validation_step(self, batch, batch_idx):
        x, z, c = batch
        batch_size, lags_and_length, _ = x.shape
        length = lags_and_length - self.lags

        x_recon, mus, logvars, z_est = self.net(x)
        recon_loss = self.criteria(x_recon, x)
        z_hat = mus
        z_est_pair = (
            z_hat.unfold(dimension=1, size=self.lags + 1, step=1)
            .transpose(-2, -1)
            .reshape(batch_size, length, -1)
        )
        C_z_pair = self.gating(z_est_pair)
        C_z_pair = torch.nn.functional.gumbel_softmax(C_z_pair, tau=1, hard=True)
        z_prev = z_hat[:, :-self.lags]
        z_curr = z_hat[:, self.lags:]
        z_hat_curr = torch.zeros_like(z_curr)
        for i in range(self.n_class):
            z_hat_curr += C_z_pair[:, :, i].unsqueeze(-1) * self.jacobian_mlps[i](z_prev)
        transition_loss = self.criteria(z_hat_curr, z_curr)
        c_est = C_z_pair.argmax(dim=-1)
        past_kld, future_kld = self.tdrl_loss(mus, logvars, z_est, c_est)

        loss = (recon_loss + self.alpha * transition_loss
                + self.beta * past_kld + self.gamma * future_kld)

        self.validation_step_outputs.append({
            "loss": loss.data,
            "recon_loss": recon_loss.data,
            "c": c.cpu().numpy(),
            "c_est": c_est.cpu().numpy(),
            "z": z.cpu().numpy(),
            "z_est": mus.cpu().numpy(),
        })

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        avg_recon = torch.stack([x["recon_loss"] for x in outputs]).mean()
        self.log("val/recon_loss", avg_recon, prog_bar=True)

        c = np.concatenate([x["c"] for x in outputs], axis=0)
        c_est = np.concatenate([x["c_est"] for x in outputs], axis=0)
        # Flatten for acc
        c_flat = c.reshape(-1)
        c_est_flat = c_est.reshape(-1)
        try:
            acc, _ = compute_acc(
                c_flat.reshape(1, -1), c_est_flat.reshape(1, -1), C=self.n_class)
        except Exception:
            acc = 0.0
        self.log("val/acc", acc, prog_bar=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return [torch.optim.AdamW(self.parameters(), lr=self.lr,
                                   betas=(0.9, 0.999),
                                   weight_decay=self.weight_decay)], []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str,
                        default=str(_REPO / "data/synthetic_well_v2/source/data.npz"))
    parser.add_argument("--output", type=str,
                        default=str(_REPO / "baseline/results/ctrlns_synthetic"))
    parser.add_argument("--seed", type=int, default=770)
    parser.add_argument("--n_class", type=int, default=2)
    parser.add_argument("--z_dim", type=int, default=6)
    parser.add_argument("--x_dim", type=int, default=30)
    parser.add_argument("--lags", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--bs", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    dataset = NpzWindowDataset(args.data, z_dim=args.z_dim)
    n_val = min(int(0.1 * len(dataset)), 4096)
    train_data, val_data = random_split(dataset, [len(dataset) - n_val, n_val])

    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.bs,
                              pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_data, shuffle=False, batch_size=args.bs * 4,
                            pin_memory=True, num_workers=4)

    model = SparseX(
        n_class=args.n_class, x_dim=args.x_dim, z_dim=args.z_dim,
        lags=args.lags, hidden_dim=args.hidden_dim, lr=args.lr,
        alpha=0.02, beta=0.002, gamma=0.02, weight_decay=1e-4,
    )

    trainer = pl.Trainer(
        default_root_dir=args.output,
        max_epochs=args.epochs,
        val_check_interval=0.1,
        accelerator="gpu",
        devices=[0],
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
