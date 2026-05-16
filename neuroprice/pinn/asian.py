from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class AsianOptionDomain:
    K: float = 100.0
    r: float = 0.05
    sigma: float = 0.20
    T: float = 1.0
    S_max: float = 300.0
    A_max: float = 300.0


@dataclass(frozen=True)
class AsianPINNLossWeights:
    pde: float = 1.0
    terminal: float = 10.0
    lower_boundary: float = 1.0
    upper_boundary: float = 1.0
    supervised: float = 0.0


@dataclass(frozen=True)
class AsianCollocationBatch:
    interior_S: torch.Tensor
    interior_A: torch.Tensor
    interior_tau: torch.Tensor
    terminal_S: torch.Tensor
    terminal_A: torch.Tensor
    terminal_tau: torch.Tensor
    lower_S: torch.Tensor
    lower_A: torch.Tensor
    lower_tau: torch.Tensor
    upper_S: torch.Tensor
    upper_A: torch.Tensor
    upper_tau: torch.Tensor
    supervised_S: torch.Tensor | None = None
    supervised_A: torch.Tensor | None = None
    supervised_tau: torch.Tensor | None = None


@dataclass(frozen=True)
class AsianPINNLossBreakdown:
    total: torch.Tensor
    pde: torch.Tensor
    terminal: torch.Tensor
    lower_boundary: torch.Tensor
    upper_boundary: torch.Tensor
    supervised: torch.Tensor


class AsianArithmeticPINN(nn.Module):
    def __init__(self, hidden_dim: int = 96, hidden_layers: int = 5) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(3, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, S_norm: torch.Tensor, A_norm: torch.Tensor, tau_norm: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([S_norm, A_norm, tau_norm], dim=1))


def normalized_to_asian_physical(
    S_norm: torch.Tensor,
    A_norm: torch.Tensor,
    tau_norm: torch.Tensor,
    domain: AsianOptionDomain,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    S = domain.S_max * S_norm
    A = domain.A_max * A_norm
    tau = domain.T * tau_norm
    return S, A, tau


def sample_asian_batch(
    *,
    n_interior: int,
    n_terminal: int,
    n_boundary: int,
    n_supervised: int,
    device: torch.device,
    domain: AsianOptionDomain,
    improved_sampling: bool = True,
) -> AsianCollocationBatch:
    strike_S_norm = domain.K / domain.S_max
    strike_A_norm = domain.K / domain.A_max

    def sample_points(n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not improved_sampling:
            return torch.rand(n, 1, device=device), torch.rand(n, 1, device=device), torch.rand(n, 1, device=device)
        n_strike = n // 3
        n_short_maturity = n // 3
        n_uniform = n - n_strike - n_short_maturity
        S_uniform = torch.rand(n_uniform, 1, device=device)
        A_uniform = torch.rand(n_uniform, 1, device=device)
        tau_uniform = torch.rand(n_uniform, 1, device=device)
        S_strike = torch.clamp(strike_S_norm + 0.10 * torch.randn(n_strike, 1, device=device), 0.0, 1.0)
        A_strike = torch.clamp(strike_A_norm + 0.10 * torch.randn(n_strike, 1, device=device), 0.0, 1.0)
        tau_strike = torch.rand(n_strike, 1, device=device)
        S_short = torch.rand(n_short_maturity, 1, device=device)
        A_short = torch.rand(n_short_maturity, 1, device=device)
        tau_short = torch.rand(n_short_maturity, 1, device=device) ** 2
        S_norm = torch.cat([S_uniform, S_strike, S_short], dim=0)
        A_norm = torch.cat([A_uniform, A_strike, A_short], dim=0)
        tau_norm = torch.cat([tau_uniform, tau_strike, tau_short], dim=0)
        permutation = torch.randperm(n, device=device)
        return S_norm[permutation], A_norm[permutation], tau_norm[permutation]

    interior_S, interior_A, interior_tau = sample_points(n_interior)
    terminal_S = torch.rand(n_terminal, 1, device=device)
    terminal_A = torch.rand(n_terminal, 1, device=device)
    terminal_tau = torch.zeros(n_terminal, 1, device=device)
    lower_S = torch.zeros(n_boundary, 1, device=device)
    lower_A = torch.rand(n_boundary, 1, device=device)
    lower_tau = torch.rand(n_boundary, 1, device=device)
    upper_S = torch.ones(n_boundary, 1, device=device)
    upper_A = torch.rand(n_boundary, 1, device=device)
    upper_tau = torch.rand(n_boundary, 1, device=device)
    if n_supervised > 0:
        supervised_S, supervised_A, supervised_tau = sample_points(n_supervised)
    else:
        supervised_S = None
        supervised_A = None
        supervised_tau = None
    return AsianCollocationBatch(
        interior_S,
        interior_A,
        interior_tau,
        terminal_S,
        terminal_A,
        terminal_tau,
        lower_S,
        lower_A,
        lower_tau,
        upper_S,
        upper_A,
        upper_tau,
        supervised_S,
        supervised_A,
        supervised_tau,
    )


def asian_pde_residual(
    model: nn.Module,
    S_norm: torch.Tensor,
    A_norm: torch.Tensor,
    tau_norm: torch.Tensor,
    domain: AsianOptionDomain,
) -> torch.Tensor:
    S_norm = S_norm.clone().detach().requires_grad_(True)
    A_norm = A_norm.clone().detach().requires_grad_(True)
    tau_norm = tau_norm.clone().detach().requires_grad_(True)
    V_norm = model(S_norm, A_norm, tau_norm)
    V = domain.S_max * V_norm
    dV_dS_norm = torch.autograd.grad(V, S_norm, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    dV_dA_norm = torch.autograd.grad(V, A_norm, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    dV_dtau_norm = torch.autograd.grad(V, tau_norm, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    d2V_dS_norm2 = torch.autograd.grad(dV_dS_norm, S_norm, grad_outputs=torch.ones_like(dV_dS_norm), create_graph=True)[0]
    S, A, tau = normalized_to_asian_physical(S_norm, A_norm, tau_norm, domain)
    dV_dS = dV_dS_norm / domain.S_max
    dV_dA = dV_dA_norm / domain.A_max
    d2V_dS2 = d2V_dS_norm2 / (domain.S_max * domain.S_max)
    dV_dtau = dV_dtau_norm / domain.T
    tau_safe = torch.clamp(tau, min=1e-3)
    average_drift = (S - A) / tau_safe
    residual = dV_dtau - 0.5 * domain.sigma**2 * S**2 * d2V_dS2 - domain.r * S * dV_dS - average_drift * dV_dA + domain.r * V
    return residual / domain.S_max


def asian_pinn_loss(
    model: nn.Module,
    batch: AsianCollocationBatch,
    domain: AsianOptionDomain,
    weights: AsianPINNLossWeights | None = None,
) -> AsianPINNLossBreakdown:
    if weights is None:
        weights = AsianPINNLossWeights()
    residual = asian_pde_residual(model, batch.interior_S, batch.interior_A, batch.interior_tau, domain)
    pde_loss = torch.mean(residual**2)
    _, A_terminal, _ = normalized_to_asian_physical(batch.terminal_S, batch.terminal_A, batch.terminal_tau, domain)
    terminal_target = torch.clamp(A_terminal - domain.K, min=0.0) / domain.S_max
    terminal_loss = torch.mean((model(batch.terminal_S, batch.terminal_A, batch.terminal_tau) - terminal_target) ** 2)
    lower_loss = torch.mean(model(batch.lower_S, batch.lower_A, batch.lower_tau) ** 2)
    S_upper, A_upper, tau_upper = normalized_to_asian_physical(batch.upper_S, batch.upper_A, batch.upper_tau, domain)
    upper_target = torch.clamp((A_upper + tau_upper * S_upper) / torch.clamp(1.0 + tau_upper, min=1e-6) - domain.K * torch.exp(-domain.r * tau_upper), min=0.0) / domain.S_max
    upper_loss = torch.mean((model(batch.upper_S, batch.upper_A, batch.upper_tau) - upper_target) ** 2)
    supervised_loss = torch.zeros((), device=batch.interior_S.device)
    total = weights.pde * pde_loss + weights.terminal * terminal_loss + weights.lower_boundary * lower_loss + weights.upper_boundary * upper_loss + weights.supervised * supervised_loss
    return AsianPINNLossBreakdown(total, pde_loss, terminal_loss, lower_loss, upper_loss, supervised_loss)
