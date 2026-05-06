"""MOSAIC: Module discovery via Sparse Additive Identifiable Causal learning.

A sparse temporal VAE for scientific time series. Stage 1 identifies latent
factors via a regime-conditioned temporal prior (NPChangeTransitionPrior_v2).
Stage 2 freezes the encoder/prior and recovers each latent's main-effect
support through an additive decoder with column-entropy sparsity.

Top-level entry points
----------------------
mosaic.models.MOSAIC          : the LightningModule (alias of TimeVaryingProcess_v2)
mosaic.data.MDRegimeDataset   : standard (xt, yt, ct) windowed-tensor loader

See README.md and `scripts/train/` for end-to-end recipes.
"""

from .models.mosaic import TimeVaryingProcess_v2, MOSAIC

__all__ = ["TimeVaryingProcess_v2", "MOSAIC"]
__version__ = "0.1.0"
