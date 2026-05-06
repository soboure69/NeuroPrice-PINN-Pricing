from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from neuroprice.pinn.collocation import CollocationBatch


@dataclass(frozen=True)
class BlackScholesDomain:
    K: float = 100.0
    r: float = 0.05
    sigma: float = 0.20
    T: float = 1.0
    S_max: float = 300.0


@dataclass(frozen=True)
class PINNLossWeights:
    pde: float = 1.0
    terminal: float = 10.0
    lower_boundary: float = 1.0
    upper_boundary: float = 1.0
    supervised: float = 0.0


@dataclass(frozen=True)
class PINNLossBreakdown:
    total: torch.Tensor
    pde: torch.Tensor
    terminal: torch.Tensor
    lower_boundary: torch.Tensor
    upper_boundary: torch.Tensor
    supervised: torch.Tensor


def normalized_to_physical(S_norm: torch.Tensor, tau_norm: torch.Tensor, domain: BlackScholesDomain) -> tuple[torch.Tensor, torch.Tensor]:
    S = domain.S_max * S_norm
    tau = domain.T * tau_norm
    return S, tau


def black_scholes_call_price_torch(S: torch.Tensor, tau: torch.Tensor, domain: BlackScholesDomain) -> torch.Tensor:
    S_safe = torch.clamp(S, min=1e-12)
    tau_safe = torch.clamp(tau, min=1e-12)
    sqrt_tau = torch.sqrt(tau_safe)
    d1 = (torch.log(S_safe / domain.K) + (domain.r + 0.5 * domain.sigma**2) * tau_safe) / (domain.sigma * sqrt_tau)
    d2 = d1 - domain.sigma * sqrt_tau
    normal = torch.distributions.Normal(loc=0.0, scale=1.0)
    price = S_safe * normal.cdf(d1) - domain.K * torch.exp(-domain.r * tau_safe) * normal.cdf(d2)
    payoff = torch.clamp(S - domain.K, min=0.0)
    return torch.where(tau <= 1e-12, payoff, price)


def black_scholes_pde_residual(model: nn.Module, S_norm: torch.Tensor, tau_norm: torch.Tensor, domain: BlackScholesDomain) -> torch.Tensor:
    S_norm = S_norm.clone().detach().requires_grad_(True)
    tau_norm = tau_norm.clone().detach().requires_grad_(True)

    V_norm = model(S_norm, tau_norm)
    V = domain.S_max * V_norm
    dV_dS_norm = torch.autograd.grad(V, S_norm, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    dV_dtau_norm = torch.autograd.grad(V, tau_norm, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    d2V_dS_norm2 = torch.autograd.grad(dV_dS_norm, S_norm, grad_outputs=torch.ones_like(dV_dS_norm), create_graph=True)[0]

    S, _ = normalized_to_physical(S_norm, tau_norm, domain)
    dV_dS = dV_dS_norm / domain.S_max
    d2V_dS2 = d2V_dS_norm2 / (domain.S_max * domain.S_max)
    dV_dtau = dV_dtau_norm / domain.T

    residual = dV_dtau - 0.5 * domain.sigma**2 * S**2 * d2V_dS2 - domain.r * S * dV_dS + domain.r * V
    return residual / domain.S_max


def black_scholes_pinn_loss(
    model: nn.Module,
    batch: CollocationBatch,
    domain: BlackScholesDomain,
    weights: PINNLossWeights | None = None,
) -> PINNLossBreakdown:
    if weights is None:
        weights = PINNLossWeights()

    residual = black_scholes_pde_residual(model, batch.interior_S, batch.interior_tau, domain)
    pde_loss = torch.mean(residual**2)

    S_terminal, _ = normalized_to_physical(batch.terminal_S, batch.terminal_tau, domain)
    terminal_target = torch.clamp(S_terminal - domain.K, min=0.0) / domain.S_max
    terminal_pred = model(batch.terminal_S, batch.terminal_tau)
    terminal_loss = torch.mean((terminal_pred - terminal_target) ** 2)

    lower_pred = model(batch.lower_S, batch.lower_tau)
    lower_loss = torch.mean(lower_pred**2)

    S_upper, tau_upper = normalized_to_physical(batch.upper_S, batch.upper_tau, domain)
    upper_target = (S_upper - domain.K * torch.exp(-domain.r * tau_upper)) / domain.S_max
    upper_pred = model(batch.upper_S, batch.upper_tau)
    upper_loss = torch.mean((upper_pred - upper_target) ** 2)

    if weights.supervised > 0.0 and batch.supervised_S is not None and batch.supervised_tau is not None:
        S_supervised, tau_supervised = normalized_to_physical(batch.supervised_S, batch.supervised_tau, domain)
        supervised_target = black_scholes_call_price_torch(S_supervised, tau_supervised, domain) / domain.S_max
        supervised_pred = model(batch.supervised_S, batch.supervised_tau)
        supervised_loss = torch.mean((supervised_pred - supervised_target) ** 2)
    else:
        supervised_loss = torch.zeros((), device=batch.interior_S.device)

    total = (
        weights.pde * pde_loss
        + weights.terminal * terminal_loss
        + weights.lower_boundary * lower_loss
        + weights.upper_boundary * upper_loss
        + weights.supervised * supervised_loss
    )

    return PINNLossBreakdown(
        total=total,
        pde=pde_loss,
        terminal=terminal_loss,
        lower_boundary=lower_loss,
        upper_boundary=upper_loss,
        supervised=supervised_loss,
    )
