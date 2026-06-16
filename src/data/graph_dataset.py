"""
Dataset utilities for aligned graph-mode PWV forecasting.
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class GraphPWVDataset(Dataset):
    """
    Sliding-window dataset for graph-mode PWV forecasting.

    Input arrays must have shape [time, num_nodes]. Each sample returns:
        x: [seq_len, num_nodes]
        y: [pred_len, num_nodes]
    """

    def __init__(
        self,
        data,
        seq_len: int = 168,
        pred_len: int = 24,
        stride: int = 1,
        n_samples_per_epoch: int = None,
    ):
        values = np.asarray(data, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"GraphPWVDataset expects [time, nodes], got {values.shape}")

        self.values = values
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride
        self.n_samples_per_epoch = n_samples_per_epoch
        self.total_len = seq_len + pred_len

        self.total_windows = max(0, (len(self.values) - self.total_len) // self.stride + 1)
        if self.total_windows <= 0:
            raise ValueError(
                f"No graph windows available: len={len(self.values)}, "
                f"seq_len={seq_len}, pred_len={pred_len}."
            )

        self._len = self.n_samples_per_epoch if self.n_samples_per_epoch is not None else self.total_windows

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if self.n_samples_per_epoch is not None:
            max_start = len(self.values) - self.total_len
            start = np.random.randint(0, max_start + 1)
        else:
            start = idx * self.stride

        x = self.values[start : start + self.seq_len]
        y = self.values[start + self.seq_len : start + self.total_len]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


def load_graph_processed_data(processed_dir: str, split: str = "train"):
    """Load graph-mode .npy file with shape [time, num_nodes]."""
    path = os.path.join(processed_dir, f"{split}.npy")
    return np.load(path, allow_pickle=False)
