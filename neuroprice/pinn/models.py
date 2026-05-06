from __future__ import annotations

import torch
from torch import nn


class BlackScholesPINN(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        hidden_layers: int = 4,
        output_transform: str = "direct",
        K: float = 100.0,
        r: float = 0.05,
        T: float = 1.0,
        S_max: float = 300.0,
    ) -> None:
        super().__init__()
        if output_transform not in {"direct", "constrained"}:
            raise ValueError("output_transform must be either 'direct' or 'constrained'")
        self.output_transform = output_transform
        self.K = float(K)
        self.r = float(r)
        self.T = float(T)
        self.S_max = float(S_max)
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
        x = torch.cat([S_norm, tau_norm], dim=1)
        raw = self.net(x)
        if self.output_transform == "direct":
            return raw

        S = self.S_max * S_norm
        tau = self.T * tau_norm
        payoff = torch.clamp(S - self.K, min=0.0) / self.S_max
        upper = (self.S_max - self.K * torch.exp(-self.r * tau)) / self.S_max
        upper_payoff = (self.S_max - self.K) / self.S_max
        baseline = payoff + S_norm * (upper - upper_payoff)
        envelope = tau_norm * S_norm * (1.0 - S_norm)
        return baseline + envelope * raw
