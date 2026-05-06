"""dataset.py — standard windowed dataset for MOSAIC.

Every prepared dataset (synthetic, RNA, OMNI, ENSO, TEP, disruption) is
serialized to a single ``data.npz`` containing three arrays:

    xt : (N, T, D)  -- observed-variable windows of length T
    yt : (N, T, D)  -- copy of xt (kept for compatibility with the v1 ELBO,
                       which expects a paired tensor for some baselines)
    ct : (N, 1)     -- regime label per window (binary 0/1 by default)

The lag in the temporal prior is ``T - 1`` (the last frame is the target,
the preceding ``T - 1`` frames are context).
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset


class MDRegimeDataset(Dataset):
    """Loads a single ``data.npz`` produced by the data-prep scripts.

    Parameters
    ----------
    data_path : str or Path
        Either the directory containing ``data.npz`` or the file path itself.
    verbose : bool
        Print summary statistics on construction (default True).
    """

    def __init__(self, data_path: Union[str, Path], verbose: bool = True):
        super().__init__()
        p = Path(data_path)
        if p.is_dir():
            p = p / "data.npz"
        if not p.exists():
            raise FileNotFoundError(f"data.npz not found at {p}")

        with np.load(p) as npz:
            self.data = {key: npz[key].copy() for key in ("yt", "xt", "ct")}

        if verbose:
            print(f"Loaded dataset from {p}")
            print(f"  xt: {self.data['xt'].shape}")
            print(
                f"  Regime 0: {int((self.data['ct'] == 0).sum())}, "
                f"Regime 1: {int((self.data['ct'] == 1).sum())}"
            )

    def __len__(self) -> int:
        return len(self.data["yt"])

    def __getitem__(self, idx: int):
        return {
            "yt": torch.from_numpy(self.data["yt"][idx].astype("float32")),
            "xt": torch.from_numpy(self.data["xt"][idx].astype("float32")),
            "ct": torch.from_numpy(
                np.atleast_1d(self.data["ct"][idx]).astype("float32")
            ),
        }
