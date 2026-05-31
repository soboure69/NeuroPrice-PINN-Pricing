from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class BasketSurrogateDomain:
    n_assets: int = 5
    spot_max: float = 150.0
    sigma_max: float = 0.60
    strike_max: float = 140.0
    rate_min: float = 0.0
    rate_max: float = 0.10
    maturity_max: float = 3.0
    correlation_min: float = -0.25
    correlation_max: float = 0.75

    @property
    def input_dim(self) -> int:
        return 3 * self.n_assets + 8


class BasketCallSurrogate(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, hidden_layers: int = 5) -> None:
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
