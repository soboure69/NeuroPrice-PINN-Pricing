from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class HestonSurrogateDomain:
    spot_max: float = 150.0
    strike_max: float = 140.0
    rate_min: float = 0.0
    rate_max: float = 0.10
    maturity_max: float = 3.0
    v0_max: float = 0.25
    kappa_max: float = 5.0
    theta_max: float = 0.25
    xi_max: float = 1.0
    rho_min: float = -0.90
    rho_max: float = 0.20

    @property
    def input_dim(self) -> int:
        return 13


class HestonCallSurrogate(nn.Module):
    def __init__(self, input_dim: int = 13, hidden_dim: int = 256, hidden_layers: int = 5) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
