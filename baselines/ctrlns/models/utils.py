import torch
from torch import nn
import torch.nn.init as init
from torch.func import jacfwd, jacrev, vmap


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = std.data.new(std.size()).normal_()
    return mu + std * eps


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, depth=1):
        super(MLP, self).__init__()
        self.relu = nn.LeakyReLU(0.1)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList()
        for _ in range(depth - 1):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = self.relu(self.fc1(x))
        for hidden_layer in self.hidden_layers:
            h = self.relu(hidden_layer(h))
        out = self.fc2(h)
        return out


class JacobianMLP(nn.Module):
    def __init__(self, jacobian_support, hid_dim):
        super(JacobianMLP, self).__init__()
        jacobian_support = torch.tensor(jacobian_support)
        out_dim, in_dim = jacobian_support.shape
        self.out_dim = out_dim
        self.input_layers = nn.ModuleList()
        self.output_layer = nn.ModuleList()
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.jacobian_support = jacobian_support
        for i in range(out_dim):
            linear_layer = nn.Linear(in_dim, hid_dim)
            # zero out the weights
            linear_layer.weight.data.zero_()
            # Use boolean indexing for efficient weight initialization
            mask = jacobian_support[i] == 1
            assert mask.sum() > 0, "Each output must depend on at least one input"
            normal_weights = torch.randn(hid_dim, in_dim)
            small_values_mask = normal_weights.abs() < 0.01
            adjusted_values = 0.02 * (normal_weights >= 0).float() - 0.01
            normal_weights[small_values_mask] = adjusted_values[small_values_mask]
            normal_weights = normal_weights * mask.float()
            linear_layer.weight.data = normal_weights  # Transpose to match shape

            self.input_layers.append(linear_layer)
            self.output_layer.append(nn.Linear(hid_dim, 1))

    def get_l1(self):
        return sum(
            [
                torch.abs(param[1]).sum()
                for param in self.named_parameters()
                if "weight" in param[0] and "input_layers" in param[0]
            ]
        )

    def forward(self, x):
        outs = []
        for i in range(self.out_dim):
            hidden = self.relu(self.input_layers[i](x))
            out = self.output_layer[i](hidden)
            outs.append(out)
        outs = torch.cat(outs, dim=-1)
        return outs


class BetaVAE_MLP(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, input_dim=3, z_dim=10, hidden_dim=128, leaky_relu_slope=0.2):
        super(BetaVAE_MLP, self).__init__()
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Linear(hidden_dim, 2 * z_dim),
        )
        # Fix the functional form to ground-truth mixing function
        self.decoder = nn.Sequential(
            nn.LeakyReLU(leaky_relu_slope),
            nn.Linear(z_dim, hidden_dim),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Linear(hidden_dim, input_dim),
        )
        # self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x, return_z=True):

        distributions = self._encode(x)
        mu = distributions[..., : self.z_dim]
        logvar = distributions[..., self.z_dim :]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        if return_z:
            return x_recon, mu, logvar, z
        else:
            return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


class ParallelMLP(nn.Module):
    """N independent MLPs computed in one batched operation via einsum."""

    def __init__(self, n_parallel, in_features, num_layers, hidden_dim):
        super().__init__()
        self.n_parallel = n_parallel
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.act = nn.LeakyReLU(0.2)

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for l in range(num_layers):
            fan_in = in_features if l == 0 else hidden_dim
            fan_out = hidden_dim
            W = nn.Parameter(torch.empty(n_parallel, fan_out, fan_in))
            b = nn.Parameter(torch.zeros(n_parallel, fan_out))
            nn.init.kaiming_uniform_(W.view(n_parallel * fan_out, fan_in))
            W.data = W.data.view(n_parallel, fan_out, fan_in)
            self.weights.append(W)
            self.biases.append(b)

        W_out = nn.Parameter(torch.empty(n_parallel, 1, hidden_dim))
        b_out = nn.Parameter(torch.zeros(n_parallel, 1))
        nn.init.kaiming_uniform_(W_out.view(n_parallel, hidden_dim))
        W_out.data = W_out.data.view(n_parallel, 1, hidden_dim)
        self.weights.append(W_out)
        self.biases.append(b_out)

    def forward(self, x):
        h = x
        for l in range(self.num_layers):
            h = torch.einsum('bni,noi->bno', h, self.weights[l]) + self.biases[l]
            h = self.act(h)
        out = torch.einsum('bni,noi->bno', h, self.weights[-1]) + self.biases[-1]
        return out


class NLayerLeakyMLP(nn.Module):
    def __init__(self, in_features, out_features, num_layers, hidden_dim):
        super().__init__()
        layers = [nn.Linear(in_features, hidden_dim), nn.LeakyReLU(0.2)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2)]
        layers.append(nn.Linear(hidden_dim, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class NPChangeTransitionPrior(nn.Module):
    """Optimized: ParallelMLP + autograd.grad (10-50x faster than vmap+jacfwd)."""

    def __init__(self, lags, latent_size, embedding_dim, num_layers=3, hidden_dim=64):
        super().__init__()
        self.L = lags
        self.latent_size = latent_size

        in_features = hidden_dim + lags * latent_size + 1
        self.parallel_gs = ParallelMLP(
            n_parallel=latent_size,
            in_features=in_features,
            num_layers=num_layers,
            hidden_dim=hidden_dim
        )

        self.fc = NLayerLeakyMLP(in_features=embedding_dim,
                                 out_features=hidden_dim,
                                 num_layers=2,
                                 hidden_dim=hidden_dim)

    def forward(self, x, embeddings):
        batch_size, length, input_dim = x.shape
        embeddings = self.fc(embeddings)

        x = x.unfold(dimension=1, size=self.L + 1, step=1)
        x = torch.swapaxes(x, 2, 3)

        num_windows = length - self.L
        embeddings = (embeddings.unsqueeze(1)
                      .expand(-1, num_windows, -1)
                      .reshape(-1, embeddings.shape[-1]))

        x = x.reshape(-1, self.L + 1, input_dim)
        xx = x[:, -1:]
        yy = x[:, :-1]
        yy = yy.reshape(-1, self.L * input_dim)

        emb_exp = embeddings.unsqueeze(1).expand(-1, input_dim, -1)
        yy_exp = yy.unsqueeze(1).expand(-1, input_dim, -1)

        xx_flat = xx.squeeze(1)
        xx_diag = xx_flat.unsqueeze(1) * torch.eye(
            input_dim, device=xx_flat.device
        ).unsqueeze(0)
        xx_diag = xx_diag.sum(-1, keepdim=True)

        parallel_inputs = torch.cat([emb_exp, yy_exp, xx_diag], dim=-1)
        if self.training:
            parallel_inputs = parallel_inputs.requires_grad_(True)
        else:
            parallel_inputs = parallel_inputs.detach().requires_grad_(True)

        with torch.enable_grad():
            residuals = self.parallel_gs(parallel_inputs).squeeze(-1)
            grad_all = torch.autograd.grad(
                outputs=residuals.sum(),
                inputs=parallel_inputs,
                create_graph=self.training,
            )[0]
            grad = grad_all[:, :, -1]

        logabsdet = torch.log(torch.abs(grad) + 1e-8)
        sum_log_abs_det_jacobian = logabsdet.sum(dim=-1)

        residuals = residuals.reshape(batch_size, num_windows, input_dim)
        sum_log_abs_det_jacobian = sum_log_abs_det_jacobian.reshape(
            batch_size, num_windows).sum(dim=1)

        return residuals, sum_log_abs_det_jacobian
