"""Dataset adapter: load our project npz into CtrlNS format."""

import torch
import numpy as np
from torch.utils.data import Dataset


class NpzWindowDataset(Dataset):
    """Wraps windowed npz data (xt, ct) for CtrlNS training.

    CtrlNS expects (x, z, c) per sample:
      x: (lags_and_length, x_dim) — observation window
      z: (lags_and_length, z_dim) — ground truth latents (dummy, only for MCC logging)
      c: (length,) — regime labels, where length = lags_and_length - lags
    """

    def __init__(self, npz_path, z_dim=6):
        super().__init__()
        data = np.load(npz_path)
        self.xt = data['xt'].astype(np.float32)  # (N, lag+1, obs_dim)
        ct = data['ct'].astype(np.int64).squeeze()  # (N,)
        self.ct = ct.reshape(-1, 1)  # (N, 1) — length=1 per window
        seq_len = self.xt.shape[1]
        # Dummy z for MCC logging (not used in training loss)
        self.zt = np.zeros((len(self.xt), seq_len, z_dim), dtype=np.float32)

    def __len__(self):
        return len(self.xt)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.xt[idx])
        z = torch.from_numpy(self.zt[idx])
        c = torch.from_numpy(self.ct[idx])
        return x, z, c
