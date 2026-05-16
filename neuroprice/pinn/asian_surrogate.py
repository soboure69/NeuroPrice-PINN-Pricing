from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class AsianSurrogateDomain:
    K: float = 100.0
    r: float = 0.05
    sigma: float = 0.20
    T: float = 1.0
    S_max: float = 300.0


class AsianArithmeticSurrogate(nn.Module):
    def __init__(self, hidden_dim: int = 128, hidden_layers: int = 5) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, S_norm: torch.Tensor, tau_norm: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([S_norm, tau_norm], dim=1))
