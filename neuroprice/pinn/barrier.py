from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class BarrierOptionDomain:
    K: float = 100.0
    B: float = 70.0
    r: float = 0.05
    sigma: float = 0.20
    T: float = 1.0
    S_max: float = 300.0


@dataclass(frozen=True)
class BarrierPINNLossWeights:
    pde: float = 1.0
    terminal: float = 10.0
    barrier: float = 10.0
    upper_boundary: float = 1.0
    supervised: float = 0.0
    near_barrier_supervised: float = 0.0


@dataclass(frozen=True)
class BarrierCollocationBatch:
    interior_S: torch.Tensor
    interior_tau: torch.Tensor
    terminal_S: torch.Tensor
    terminal_tau: torch.Tensor
    barrier_S: torch.Tensor
    barrier_tau: torch.Tensor
    upper_S: torch.Tensor
    upper_tau: torch.Tensor
    supervised_S: torch.Tensor | None = None
    supervised_tau: torch.Tensor | None = None


@dataclass(frozen=True)
class BarrierPINNLossBreakdown:
    total: torch.Tensor
    pde: torch.Tensor
    terminal: torch.Tensor
    barrier: torch.Tensor
    upper_boundary: torch.Tensor
    supervised: torch.Tensor


class DownAndOutBarrierPINN(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        hidden_layers: int = 4,
        output_transform: str = "direct",
        K: float = 100.0,
        B: float = 70.0,
        r: float = 0.05,
        T: float = 1.0,
        S_max: float = 300.0,
    ) -> None:
        super().__init__()
        if output_transform not in {"direct", "barrier"}:
            raise ValueError("output_transform must be either 'direct' or 'barrier'")
        self.output_transform = output_transform
        self.K = float(K)
        self.B = float(B)
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
        raw = self.net(torch.cat([S_norm, tau_norm], dim=1))
        if self.output_transform == "direct":
            return raw
        S = self.S_max * S_norm
        barrier_factor = torch.clamp((S - self.B) / (self.S_max - self.B), min=0.0)
        return barrier_factor * raw


def normalized_to_barrier_physical(S_norm: torch.Tensor, tau_norm: torch.Tensor, domain: BarrierOptionDomain) -> tuple[torch.Tensor, torch.Tensor]:
    S = domain.S_max * S_norm
    tau = domain.T * tau_norm
    return S, tau


def down_and_out_call_price_torch(S: torch.Tensor, tau: torch.Tensor, domain: BarrierOptionDomain) -> torch.Tensor:
    S_safe = torch.clamp(S, min=1e-12)
    tau_safe = torch.clamp(tau, min=1e-12)
    sqrt_tau = torch.sqrt(tau_safe)
    normal = torch.distributions.Normal(loc=0.0, scale=1.0)
    d1 = (torch.log(S_safe / domain.K) + (domain.r + 0.5 * domain.sigma**2) * tau_safe) / (domain.sigma * sqrt_tau)
    d2 = d1 - domain.sigma * sqrt_tau
    vanilla = S_safe * normal.cdf(d1) - domain.K * torch.exp(-domain.r * tau_safe) * normal.cdf(d2)
    mirrored_S = (domain.B * domain.B) / S_safe
    d1_image = (torch.log(mirrored_S / domain.K) + (domain.r + 0.5 * domain.sigma**2) * tau_safe) / (domain.sigma * sqrt_tau)
    d2_image = d1_image - domain.sigma * sqrt_tau
    vanilla_image = mirrored_S * normal.cdf(d1_image) - domain.K * torch.exp(-domain.r * tau_safe) * normal.cdf(d2_image)
    power = 2.0 * domain.r / (domain.sigma * domain.sigma) - 1.0
    image = (S_safe / domain.B) ** power * vanilla_image
    price = torch.clamp(vanilla - image, min=0.0)
    payoff = torch.where(S > domain.B, torch.clamp(S - domain.K, min=0.0), torch.zeros_like(S))
    price = torch.where(S <= domain.B, torch.zeros_like(price), price)
    return torch.where(tau <= 1e-12, payoff, price)


def sample_barrier_batch(
    *,
    n_interior: int,
    n_terminal: int,
    n_boundary: int,
    n_supervised: int,
    device: torch.device,
    domain: BarrierOptionDomain,
    improved_sampling: bool = True,
) -> BarrierCollocationBatch:
    barrier_norm = domain.B / domain.S_max
    strike_norm = domain.K / domain.S_max
    near_barrier_width = 0.05 * (1.0 - barrier_norm)

    def sample_S_tau(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not improved_sampling:
            S_norm = barrier_norm + (1.0 - barrier_norm) * torch.rand(n, 1, device=device)
            tau_norm = torch.rand(n, 1, device=device)
            return S_norm, tau_norm
        n_barrier = n // 2
        n_strike = n // 4
        n_short_maturity = n // 8
        n_uniform = n - n_barrier - n_strike - n_short_maturity
        S_uniform = barrier_norm + (1.0 - barrier_norm) * torch.rand(n_uniform, 1, device=device)
        tau_uniform = torch.rand(n_uniform, 1, device=device)
        S_barrier = torch.clamp(barrier_norm + near_barrier_width * torch.rand(n_barrier, 1, device=device) ** 2, barrier_norm, 1.0)
        tau_barrier = torch.rand(n_barrier, 1, device=device)
        S_strike = torch.clamp(strike_norm + 0.10 * torch.randn(n_strike, 1, device=device), barrier_norm, 1.0)
        tau_strike = torch.rand(n_strike, 1, device=device)
        S_short_maturity = barrier_norm + (1.0 - barrier_norm) * torch.rand(n_short_maturity, 1, device=device)
        tau_short_maturity = torch.rand(n_short_maturity, 1, device=device) ** 2
        S_norm = torch.cat([S_uniform, S_barrier, S_strike, S_short_maturity], dim=0)
        tau_norm = torch.cat([tau_uniform, tau_barrier, tau_strike, tau_short_maturity], dim=0)
        permutation = torch.randperm(n, device=device)
        return S_norm[permutation], tau_norm[permutation]

    interior_S, interior_tau = sample_S_tau(n_interior)
    terminal_S = barrier_norm + (1.0 - barrier_norm) * torch.rand(n_terminal, 1, device=device)
    terminal_tau = torch.zeros(n_terminal, 1, device=device)
    barrier_S = torch.full((n_boundary, 1), barrier_norm, device=device)
    barrier_tau = torch.rand(n_boundary, 1, device=device)
    upper_S = torch.ones(n_boundary, 1, device=device)
    upper_tau = torch.rand(n_boundary, 1, device=device)
    if n_supervised > 0:
        supervised_S, supervised_tau = sample_S_tau(n_supervised)
    else:
        supervised_S = None
        supervised_tau = None
    return BarrierCollocationBatch(interior_S, interior_tau, terminal_S, terminal_tau, barrier_S, barrier_tau, upper_S, upper_tau, supervised_S, supervised_tau)


def barrier_pde_residual(model: nn.Module, S_norm: torch.Tensor, tau_norm: torch.Tensor, domain: BarrierOptionDomain) -> torch.Tensor:
    S_norm = S_norm.clone().detach().requires_grad_(True)
    tau_norm = tau_norm.clone().detach().requires_grad_(True)
    V_norm = model(S_norm, tau_norm)
    V = domain.S_max * V_norm
    dV_dS_norm = torch.autograd.grad(V, S_norm, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    dV_dtau_norm = torch.autograd.grad(V, tau_norm, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    d2V_dS_norm2 = torch.autograd.grad(dV_dS_norm, S_norm, grad_outputs=torch.ones_like(dV_dS_norm), create_graph=True)[0]
    S, _ = normalized_to_barrier_physical(S_norm, tau_norm, domain)
    dV_dS = dV_dS_norm / domain.S_max
    d2V_dS2 = d2V_dS_norm2 / (domain.S_max * domain.S_max)
    dV_dtau = dV_dtau_norm / domain.T
    residual = dV_dtau - 0.5 * domain.sigma**2 * S**2 * d2V_dS2 - domain.r * S * dV_dS + domain.r * V
    return residual / domain.S_max


def barrier_pinn_loss(
    model: nn.Module,
    batch: BarrierCollocationBatch,
    domain: BarrierOptionDomain,
    weights: BarrierPINNLossWeights | None = None,
) -> BarrierPINNLossBreakdown:
    if weights is None:
        weights = BarrierPINNLossWeights()
    residual = barrier_pde_residual(model, batch.interior_S, batch.interior_tau, domain)
    pde_loss = torch.mean(residual**2)
    S_terminal, _ = normalized_to_barrier_physical(batch.terminal_S, batch.terminal_tau, domain)
    terminal_target = torch.where(S_terminal > domain.B, torch.clamp(S_terminal - domain.K, min=0.0), torch.zeros_like(S_terminal)) / domain.S_max
    terminal_loss = torch.mean((model(batch.terminal_S, batch.terminal_tau) - terminal_target) ** 2)
    barrier_loss = torch.mean(model(batch.barrier_S, batch.barrier_tau) ** 2)
    S_upper, tau_upper = normalized_to_barrier_physical(batch.upper_S, batch.upper_tau, domain)
    upper_target = (S_upper - domain.K * torch.exp(-domain.r * tau_upper)) / domain.S_max
    upper_loss = torch.mean((model(batch.upper_S, batch.upper_tau) - upper_target) ** 2)
    if weights.supervised > 0.0 and batch.supervised_S is not None and batch.supervised_tau is not None:
        S_supervised, tau_supervised = normalized_to_barrier_physical(batch.supervised_S, batch.supervised_tau, domain)
        supervised_target = down_and_out_call_price_torch(S_supervised, tau_supervised, domain) / domain.S_max
        supervised_pred = model(batch.supervised_S, batch.supervised_tau)
        supervised_error = (supervised_pred - supervised_target) ** 2
        distance_to_barrier = torch.clamp((S_supervised - domain.B) / (domain.S_max - domain.B), min=0.0)
        near_barrier_weight = 1.0 + weights.near_barrier_supervised * torch.exp(-distance_to_barrier / 0.03)
        supervised_loss = torch.mean(near_barrier_weight * supervised_error)
    else:
        supervised_loss = torch.zeros((), device=batch.interior_S.device)
    total = weights.pde * pde_loss + weights.terminal * terminal_loss + weights.barrier * barrier_loss + weights.upper_boundary * upper_loss + weights.supervised * supervised_loss
    return BarrierPINNLossBreakdown(total, pde_loss, terminal_loss, barrier_loss, upper_loss, supervised_loss)
