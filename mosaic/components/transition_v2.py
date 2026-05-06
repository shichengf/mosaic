"""transition_v2.py — Optimized parallel transition prior for MOSAIC

Optimizations over v1 transition.py:
  1. Replace torch.autograd.functional.jacobian() with torch.autograd.grad()
     — computes only the needed scalar derivative, not the full Jacobian
  2. Replace 64 separate MLPs with a single ParallelMLP using batched matmul
     — eliminates Python loop, massive GPU utilization improvement

The mathematical computation is IDENTICAL to v1. Only the implementation
is more efficient. v1 weights can be loaded via stack/reshape conversion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ParallelMLP(nn.Module):
    """N independent MLPs computed in one batched operation.

    Equivalent to N separate NLayerLeakyMLP(in_features → hidden → ... → 1),
    but using einsum for parallel computation. No Python loop.

    Parameters
    ----------
    n_parallel : int
        Number of parallel MLPs (= latent_size, typically 64).
    in_features : int
        Input dimension for each MLP.
    num_layers : int
        Number of hidden layers.
    hidden_dim : int
        Hidden dimension for each MLP.
    """

    def __init__(self, n_parallel, in_features, num_layers, hidden_dim):
        super().__init__()
        self.n_parallel = n_parallel
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.act = nn.LeakyReLU(0.2)

        # Build parallel weight/bias lists
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for l in range(num_layers):
            fan_in = in_features if l == 0 else hidden_dim
            fan_out = hidden_dim
            # (n_parallel, fan_out, fan_in)
            W = nn.Parameter(torch.empty(n_parallel, fan_out, fan_in))
            b = nn.Parameter(torch.zeros(n_parallel, fan_out))
            nn.init.kaiming_uniform_(W.view(n_parallel * fan_out, fan_in))
            W.data = W.data.view(n_parallel, fan_out, fan_in)
            self.weights.append(W)
            self.biases.append(b)

        # Output layer: hidden → 1
        W_out = nn.Parameter(torch.empty(n_parallel, 1, hidden_dim))
        b_out = nn.Parameter(torch.zeros(n_parallel, 1))
        nn.init.kaiming_uniform_(W_out.view(n_parallel, hidden_dim))
        W_out.data = W_out.data.view(n_parallel, 1, hidden_dim)
        self.weights.append(W_out)
        self.biases.append(b_out)

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor (batch, n_parallel, in_features)

        Returns
        -------
        out : Tensor (batch, n_parallel, 1)
        """
        h = x
        for l in range(self.num_layers):
            # h: (B, N, F_in), W: (N, F_out, F_in) → (B, N, F_out)
            h = torch.einsum('bni,noi->bno', h, self.weights[l]) + self.biases[l]
            h = self.act(h)

        # Output layer (no activation)
        out = torch.einsum('bni,noi->bno', h, self.weights[-1]) + self.biases[-1]
        return out

    @staticmethod
    def from_separate_mlps(gs_list):
        """Convert a list of separate NLayerLeakyMLP into one ParallelMLP.

        This is used to load v1 weights (64 separate gs[i]) into v2 format.

        Parameters
        ----------
        gs_list : list of NLayerLeakyMLP
            Each has .net = Sequential(Linear, LeakyReLU, Linear, LeakyReLU, ..., Linear)

        Returns
        -------
        parallel : ParallelMLP with stacked weights
        """
        n_parallel = len(gs_list)

        # Parse the first MLP to get architecture
        layers = list(gs_list[0].net.children())
        linear_layers = [l for l in layers if isinstance(l, nn.Linear)]
        num_layers = len(linear_layers) - 1  # exclude output layer

        in_features = linear_layers[0].in_features
        hidden_dim = linear_layers[0].out_features

        parallel = ParallelMLP(n_parallel, in_features, num_layers, hidden_dim)

        # Stack weights from individual MLPs
        with torch.no_grad():
            for l, lin_idx in enumerate(range(len(linear_layers))):
                stacked_W = torch.stack([
                    linear_layers_i[lin_idx].weight
                    for gs_i in gs_list
                    for linear_layers_i in [
                        [layer for layer in gs_i.net.children()
                         if isinstance(layer, nn.Linear)]
                    ]
                ])
                stacked_b = torch.stack([
                    linear_layers_i[lin_idx].bias
                    for gs_i in gs_list
                    for linear_layers_i in [
                        [layer for layer in gs_i.net.children()
                         if isinstance(layer, nn.Linear)]
                    ]
                ])
                parallel.weights[l].copy_(stacked_W)
                parallel.biases[l].copy_(stacked_b)

        return parallel


class EmbeddingFC(nn.Module):
    """Small MLP for embedding projection (replaces the fc in v1)."""

    def __init__(self, in_features, out_features, hidden_dim, num_layers=2):
        super().__init__()
        layers = []
        for l in range(num_layers):
            fin = in_features if l == 0 else hidden_dim
            layers.append(nn.Linear(fin, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Linear(hidden_dim, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class NPChangeTransitionPrior_v2(nn.Module):
    """Optimized NPChangeTransitionPrior.

    Mathematically identical to v1, but:
    - Uses ParallelMLP instead of 64 separate MLPs + Python loop
    - Uses autograd.grad instead of full jacobian computation

    Parameters are identical to v1 for weight compatibility.
    """

    def __init__(self, lags, latent_size, embedding_dim,
                 num_layers=3, hidden_dim=64):
        super().__init__()
        self.L = lags
        self.latent_size = latent_size

        # Parallel MLPs: one per latent dimension
        # Each takes (embedding_hidden + lags*latent_size + 1) → 1
        in_features = hidden_dim + lags * latent_size + 1
        self.parallel_gs = ParallelMLP(
            n_parallel=latent_size,
            in_features=in_features,
            num_layers=num_layers,
            hidden_dim=hidden_dim
        )

        # Embedding projection (same as v1)
        self.fc = EmbeddingFC(
            in_features=embedding_dim,
            out_features=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=2
        )

    def forward(self, x, embeddings, masks=None):
        """
        Parameters
        ----------
        x : Tensor (batch, T, latent_size)
        embeddings : Tensor (batch, embedding_dim)

        Returns
        -------
        residuals : Tensor (batch, T-L, latent_size)
        sum_log_abs_det_jacobian : Tensor (batch,)
        """
        batch_size, length, input_dim = x.shape
        embeddings = self.fc(embeddings)

        # Unfold into windows
        x = x.unfold(dimension=1, size=self.L + 1, step=1)
        x = torch.swapaxes(x, 2, 3)

        num_windows = length - self.L
        embeddings = (embeddings.unsqueeze(1)
                      .expand(-1, num_windows, -1)
                      .reshape(-1, embeddings.shape[-1]))

        x = x.reshape(-1, self.L + 1, input_dim)
        xx = x[:, -1:]   # (BW, 1, D) — current time step
        yy = x[:, :-1]   # (BW, L, D) — lagged time steps
        yy = yy.reshape(-1, self.L * input_dim)

        BW = xx.shape[0]  # batch_size * num_windows

        # Build parallel inputs: (BW, D, in_features)
        # For each z_i: input = [embeddings, yy, xx_i]
        # embeddings: (BW, hidden) → expand to (BW, D, hidden)
        # yy: (BW, L*D) → expand to (BW, D, L*D)
        # xx_i: (BW, 1) for each i → (BW, D, 1) via diagonal selection

        emb_exp = embeddings.unsqueeze(1).expand(-1, input_dim, -1)
        yy_exp = yy.unsqueeze(1).expand(-1, input_dim, -1)

        # xx[:, 0, :] has shape (BW, D). For MLP_i, the last input is xx[:, 0, i].
        # Build diagonal selection: xx_diag[b, i, 0] = xx[b, 0, i]
        xx_flat = xx.squeeze(1)  # (BW, D)
        xx_diag = xx_flat.unsqueeze(1) * torch.eye(
            input_dim, device=xx_flat.device
        ).unsqueeze(0)  # (BW, D, D)
        xx_diag = xx_diag.sum(-1, keepdim=True)  # (BW, D, 1)

        parallel_inputs = torch.cat([emb_exp, yy_exp, xx_diag], dim=-1)
        # (BW, D, in_features)

        # Detach and re-enable grad so autograd.grad works in both train/val
        parallel_inputs = parallel_inputs.detach().requires_grad_(True)

        # Forward all D MLPs at once + compute gradient
        with torch.enable_grad():
            residuals = self.parallel_gs(parallel_inputs)  # (BW, D, 1)
            residuals = residuals.squeeze(-1)  # (BW, D)

            # d(residual_i)/d(xx_i) = d(residual_i)/d(inputs[..., -1])
            grad_all = torch.autograd.grad(
                outputs=residuals.sum(),
                inputs=parallel_inputs,
                create_graph=self.training,
            )[0]  # (BW, D, in_features)

            # Extract gradient w.r.t. last input dim (xx_i)
            grad = grad_all[:, :, -1]  # (BW, D)

        logabsdet = torch.log(torch.abs(grad) + 1e-8)  # (BW, D)
        sum_log_abs_det_jacobian = logabsdet.sum(dim=-1)  # (BW,)

        # Reshape
        residuals = residuals.reshape(batch_size, num_windows, input_dim)
        sum_log_abs_det_jacobian = sum_log_abs_det_jacobian.reshape(
            batch_size, num_windows).sum(dim=1)  # (batch_size,)

        return residuals, sum_log_abs_det_jacobian

    def load_v1_weights(self, v1_state_dict, prefix='transition_prior.'):
        """Load weights from v1 NPChangeTransitionPrior state dict.

        Converts 64 separate gs[i] weights into ParallelMLP stacked format,
        and loads the fc (embedding projection) weights.

        Parameters
        ----------
        v1_state_dict : dict
            Full v1 model state dict.
        prefix : str
            Key prefix for transition prior in v1 state dict.
        """
        # --- Load fc weights ---
        # v1 uses NLayerLeakyMLP with self.net, keys like:
        #   transition_prior.fc.net.0.weight
        # v2 uses EmbeddingFC with self.net, keys:
        #   fc.net.0.weight
        # So just strip the prefix
        fc_mapping = {}
        for key in v1_state_dict:
            if key.startswith(prefix + 'fc.'):
                new_key = key[len(prefix):]  # "fc.net.0.weight"
                fc_mapping[new_key] = v1_state_dict[key]

        # --- Load and stack gs weights ---
        # v1 keys: transition_prior.gs.{i}.net.{layer_idx}.weight/bias
        # We need to stack across i for each layer_idx
        n_parallel = self.latent_size

        # Find all linear layer indices from gs.0
        layer_indices = []
        for key in v1_state_dict:
            if key.startswith(prefix + 'gs.0.net.') and 'weight' in key:
                # e.g., "transition_prior.gs.0.net.0.weight"
                parts = key.split('.')
                layer_idx = int(parts[-2])
                layer_indices.append(layer_idx)
        layer_indices.sort()

        # Stack weights for each linear layer
        parallel_state = {}
        for param_l, v1_layer_idx in enumerate(layer_indices):
            ws = []
            bs = []
            for i in range(n_parallel):
                w_key = f'{prefix}gs.{i}.net.{v1_layer_idx}.weight'
                b_key = f'{prefix}gs.{i}.net.{v1_layer_idx}.bias'
                ws.append(v1_state_dict[w_key])
                bs.append(v1_state_dict[b_key])

            parallel_state[f'parallel_gs.weights.{param_l}'] = torch.stack(ws)
            parallel_state[f'parallel_gs.biases.{param_l}'] = torch.stack(bs)

        # Apply to self
        my_state = self.state_dict()
        loaded = 0
        for key, value in {**fc_mapping, **parallel_state}.items():
            if key in my_state and my_state[key].shape == value.shape:
                my_state[key] = value
                loaded += 1
            else:
                print(f"  WARN transition_v2: cannot load {key}, "
                      f"expected {my_state.get(key, 'MISSING')}")

        self.load_state_dict(my_state)
        return loaded
