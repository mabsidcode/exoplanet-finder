"""Compact dual-branch 1D CNN (AstroNet-style, reduced in size).

Inputs
  global view: (batch, 2, 2001)  channels = [normalised flux, coverage mask]
  local view:  (batch, 2, 201)

Each branch is a stack of blocks: Conv1d(k=5) -> ReLU -> Conv1d(k=5) -> ReLU ->
MaxPool1d(5, stride 2). The flattened branch outputs are concatenated and fed
to a small fully connected head that returns one logit. ``score =
sigmoid(logit)`` is a model score in [0, 1], not a calibrated probability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
from torch import nn


@dataclass
class ModelConfig:
    global_channels: list[int] = field(default_factory=lambda: [16, 32, 64, 64, 64])
    local_channels: list[int] = field(default_factory=lambda: [16, 32])
    kernel_size: int = 5
    head_hidden: list[int] = field(default_factory=lambda: [256, 256])
    dropout: float = 0.3
    use_local: bool = True
    global_len: int = 2001
    local_len: int = 201
    in_channels: int = 2

    @classmethod
    def from_dict(cls, d: dict) -> ModelConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)


def _pooled_len(length: int, n_blocks: int, pool: int = 5, stride: int = 2) -> int:
    for _ in range(n_blocks):
        length = (length - pool) // stride + 1
    return length


class ConvBranch(nn.Module):
    def __init__(self, in_ch: int, channels: list[int], k: int):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_ch
        for c in channels:
            layers += [
                nn.Conv1d(prev, c, k, padding=k // 2),
                nn.ReLU(),
                nn.Conv1d(c, c, k, padding=k // 2),
                nn.ReLU(),
                nn.MaxPool1d(5, stride=2),
            ]
            prev = c
        self.net = nn.Sequential(*layers, nn.Flatten())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DualBranchCNN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.global_branch = ConvBranch(cfg.in_channels, cfg.global_channels, cfg.kernel_size)
        n_feat = cfg.global_channels[-1] * _pooled_len(cfg.global_len, len(cfg.global_channels))
        if cfg.use_local:
            self.local_branch = ConvBranch(cfg.in_channels, cfg.local_channels, cfg.kernel_size)
            n_feat += cfg.local_channels[-1] * _pooled_len(cfg.local_len, len(cfg.local_channels))
        head: list[nn.Module] = []
        prev = n_feat
        for h in cfg.head_hidden:
            head += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(cfg.dropout)]
            prev = h
        head.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*head)
        self.n_features = n_feat

    def forward(self, global_view: torch.Tensor, local_view: torch.Tensor | None = None) -> torch.Tensor:
        feats = [self.global_branch(global_view)]
        if self.cfg.use_local:
            feats.append(self.local_branch(local_view))
        return self.head(torch.cat(feats, dim=1)).squeeze(1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def select_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
