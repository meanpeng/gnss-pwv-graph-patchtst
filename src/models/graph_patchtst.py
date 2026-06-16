"""
Graph-PatchTST for aligned multi-station PWV forecasting.

The model first encodes each station's temporal history with a shared PatchTST
encoder, then mixes station features with a static geographic graph.
"""
import os

import numpy as np
import torch
import torch.nn as nn


def load_adjacency(adjacency_path: str, num_nodes: int) -> torch.Tensor:
    if not adjacency_path:
        raise ValueError("GraphPatchTST requires adjacency_path.")
    if not os.path.exists(adjacency_path):
        raise FileNotFoundError(f"Adjacency file not found: {adjacency_path}")

    adjacency = np.load(adjacency_path).astype(np.float32)
    if adjacency.shape != (num_nodes, num_nodes):
        raise ValueError(
            f"Adjacency shape {adjacency.shape} does not match num_nodes={num_nodes}."
        )
    return torch.from_numpy(adjacency)


def normalize_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    """Symmetric graph normalization."""
    adjacency = adjacency.float()
    adjacency = torch.maximum(adjacency, adjacency.transpose(0, 1))
    adjacency = adjacency.clone()
    adjacency.fill_diagonal_(1.0)
    degree = adjacency.sum(dim=1).clamp_min(1e-6)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    return degree_inv_sqrt[:, None] * adjacency * degree_inv_sqrt[None, :]


class SharedPatchEmbedding(nn.Module):
    """Patchify all stations with a shared projection."""

    def __init__(self, seq_len, patch_len, stride, d_model, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (seq_len - patch_len) // stride + 1
        self.proj = nn.Linear(patch_len, d_model)
        self.position = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, N]
        bsz, _, num_nodes = x.shape
        x = x.permute(0, 2, 1)  # [B, N, L]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [B, N, P, patch_len]
        x = x.reshape(bsz * num_nodes, self.num_patches, self.patch_len)
        x = self.proj(x)
        x = x + self.position
        return self.dropout(x)


class GraphConv(nn.Module):
    """Static graph convolution over the station dimension."""

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x, adjacency):
        # x: [B, N, D], adjacency: [N, N]
        residual = x
        x = torch.einsum("ij,bjd->bid", adjacency, x)
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.norm(residual + x)


class GraphPatchTST(nn.Module):
    """
    PatchTST temporal encoder with static graph spatial mixing.

    Input:
        x: [B, seq_len, num_nodes]
    Output:
        y: [B, pred_len, num_nodes]
    """

    def __init__(
        self,
        seq_len=168,
        pred_len=24,
        patch_len=24,
        stride=6,
        d_model=64,
        n_heads=2,
        e_layers=2,
        d_ff=64,
        dropout=0.1,
        num_nodes=None,
        adjacency_path=None,
        graph_layers=1,
        graph_dropout=None,
        graph_alpha_init=0.1,
        use_node_embedding=True,
        **_,
    ):
        super().__init__()
        if num_nodes is None:
            raise ValueError("GraphPatchTST requires num_nodes.")

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_nodes = int(num_nodes)
        self.d_model = d_model
        self.graph_layers = int(graph_layers)

        self.patch_embedding = SharedPatchEmbedding(
            seq_len=seq_len,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=e_layers,
            enable_nested_tensor=False,
        )

        flattened_dim = self.patch_embedding.num_patches * d_model
        self.temporal_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(flattened_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        if use_node_embedding:
            self.node_embedding = nn.Parameter(torch.zeros(1, self.num_nodes, d_model))
            nn.init.trunc_normal_(self.node_embedding, std=0.02)
        else:
            self.node_embedding = None

        graph_dropout = dropout if graph_dropout is None else graph_dropout
        self.graph_blocks = nn.ModuleList(
            [GraphConv(d_model=d_model, dropout=graph_dropout) for _ in range(self.graph_layers)]
        )
        self.graph_scale = nn.Parameter(torch.tensor(float(graph_alpha_init)))

        adjacency = load_adjacency(adjacency_path, self.num_nodes)
        self.register_buffer("adjacency", normalize_adjacency(adjacency), persistent=True)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )

    def forward(self, x):
        if x.dim() == 4 and x.size(-1) == 1:
            x = x.squeeze(-1)
        if x.dim() != 3:
            raise ValueError(f"GraphPatchTST expects [B, L, N], got {tuple(x.shape)}")

        bsz, _, num_nodes = x.shape
        if num_nodes != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {num_nodes}.")

        x = self.patch_embedding(x)  # [B*N, P, D]
        x = self.encoder(x)
        x = x.reshape(bsz, num_nodes, -1)
        temporal_features = self.temporal_proj(x)  # [B, N, D]

        if self.node_embedding is not None:
            temporal_features = temporal_features + self.node_embedding

        graph_features = temporal_features
        for block in self.graph_blocks:
            graph_features = block(graph_features, self.adjacency)

        if self.graph_layers > 0:
            features = temporal_features + self.graph_scale * (graph_features - temporal_features)
        else:
            features = temporal_features

        y = self.head(features)  # [B, N, pred_len]
        return y.permute(0, 2, 1).contiguous()

    def get_adjacency(self):
        return self.adjacency.detach().cpu()
