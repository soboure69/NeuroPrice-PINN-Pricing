from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class LogBlackScholesDomain:
    K: float = 100.0
    r: float = 0.05
    sigma: float = 0.20
    T: float = 1.0
    x_min: float = -5.0
    x_max: float = 2.0


@dataclass(frozen=True)
class LogPINNLossWeights:
    pde: float = 1.0
    terminal: float = 10.0
    lower_boundary: float = 1.0
    upper_boundary: float = 1.0
    supervised: float = 0.0


@dataclass(frozen=True)
class LogPINNLossBreakdown:
    total: torch.Tensor
    pde: torch.Tensor
    terminal: torch.Tensor
    lower_boundary: torch.Tensor
    upper_boundary: torch.Tensor
    supervised: torch.Tensor


class LogBlackScholesPINN(nn.Module):
    def __init__(self, hidden_dim: int = 64, hidden_layers: int = 4) -> None:
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

    def forward(self, x_norm: torch.Tensor, tau_norm: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_norm, tau_norm], dim=1))


def normalized_to_log_physical(x_norm: torch.Tensor, tau_norm: torch.Tensor, domain: LogBlackScholesDomain) -> tuple[torch.Tensor, torch.Tensor]:
    x = domain.x_min + (domain.x_max - domain.x_min) * x_norm
    tau = domain.T * tau_norm
    return x, tau


def log_black_scholes_call_price_torch(x: torch.Tensor, tau: torch.Tensor, domain: LogBlackScholesDomain) -> torch.Tensor:
    S = domain.K * torch.exp(x)
    tau_safe = torch.clamp(tau, min=1e-12)
    sqrt_tau = torch.sqrt(tau_safe)
    d1 = (x + (domain.r + 0.5 * domain.sigma**2) * tau_safe) / (domain.sigma * sqrt_tau)
    d2 = d1 - domain.sigma * sqrt_tau
    normal = torch.distributions.Normal(loc=0.0, scale=1.0)
    price = S * normal.cdf(d1) - domain.K * torch.exp(-domain.r * tau_safe) * normal.cdf(d2)
    payoff = torch.clamp(S - domain.K, min=0.0)
    return torch.where(tau <= 1e-12, payoff, price) / domain.K


def sample_log_black_scholes_batch(
    *,
    n_interior: int,
    n_terminal: int,
    n_boundary: int,
    n_supervised: int,
    device: torch.device,
    improved_sampling: bool = True,
) -> dict[str, torch.Tensor | None]:
    if improved_sampling:
        n_strike = n_interior // 3
        n_maturity = n_interior // 3
        n_uniform = n_interior - n_strike - n_maturity
        x_uniform = torch.rand(n_uniform, 1, device=device)
        tau_uniform = torch.rand(n_uniform, 1, device=device)
        x_strike = torch.clamp(0.5 + 0.12 * torch.randn(n_strike, 1, device=device), 0.0, 1.0)
        tau_strike = torch.rand(n_strike, 1, device=device)
        x_maturity = torch.rand(n_maturity, 1, device=device)
        tau_maturity = torch.rand(n_maturity, 1, device=device) ** 2
        interior_x = torch.cat([x_uniform, x_strike, x_maturity], dim=0)
        interior_tau = torch.cat([tau_uniform, tau_strike, tau_maturity], dim=0)
        permutation = torch.randperm(n_interior, device=device)
        interior_x = interior_x[permutation]
        interior_tau = interior_tau[permutation]
    else:
        interior_x = torch.rand(n_interior, 1, device=device)
        interior_tau = torch.rand(n_interior, 1, device=device)

    supervised_x = torch.rand(n_supervised, 1, device=device) if n_supervised > 0 else None
    supervised_tau = torch.rand(n_supervised, 1, device=device) if n_supervised > 0 else None

    return {
        "interior_x": interior_x,
        "interior_tau": interior_tau,
        "terminal_x": torch.rand(n_terminal, 1, device=device),
        "terminal_tau": torch.zeros(n_terminal, 1, device=device),
        "lower_x": torch.zeros(n_boundary, 1, device=device),
        "lower_tau": torch.rand(n_boundary, 1, device=device),
        "upper_x": torch.ones(n_boundary, 1, device=device),
        "upper_tau": torch.rand(n_boundary, 1, device=device),
        "supervised_x": supervised_x,
        "supervised_tau": supervised_tau,
    }


def log_black_scholes_pde_residual(model: nn.Module, x_norm: torch.Tensor, tau_norm: torch.Tensor, domain: LogBlackScholesDomain) -> torch.Tensor:
    x_norm = x_norm.clone().detach().requires_grad_(True)
    tau_norm = tau_norm.clone().detach().requires_grad_(True)
    u = model(x_norm, tau_norm)
    du_dx_norm = torch.autograd.grad(u, x_norm, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    du_dtau_norm = torch.autograd.grad(u, tau_norm, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    d2u_dx_norm2 = torch.autograd.grad(du_dx_norm, x_norm, grad_outputs=torch.ones_like(du_dx_norm), create_graph=True)[0]

    x_scale = domain.x_max - domain.x_min
    du_dx = du_dx_norm / x_scale
    d2u_dx2 = d2u_dx_norm2 / (x_scale * x_scale)
    du_dtau = du_dtau_norm / domain.T

    residual = du_dtau - 0.5 * domain.sigma**2 * d2u_dx2 - (domain.r - 0.5 * domain.sigma**2) * du_dx + domain.r * u
    return residual


def log_black_scholes_pinn_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor | None],
    domain: LogBlackScholesDomain,
    weights: LogPINNLossWeights | None = None,
) -> LogPINNLossBreakdown:
    if weights is None:
        weights = LogPINNLossWeights()

    residual = log_black_scholes_pde_residual(model, batch["interior_x"], batch["interior_tau"], domain)  # type: ignore[arg-type]
    pde_loss = torch.mean(residual**2)

    x_terminal, tau_terminal = normalized_to_log_physical(batch["terminal_x"], batch["terminal_tau"], domain)  # type: ignore[arg-type]
    terminal_target = log_black_scholes_call_price_torch(x_terminal, tau_terminal, domain)
    terminal_loss = torch.mean((model(batch["terminal_x"], batch["terminal_tau"]) - terminal_target) ** 2)  # type: ignore[arg-type]

    x_lower, tau_lower = normalized_to_log_physical(batch["lower_x"], batch["lower_tau"], domain)  # type: ignore[arg-type]
    lower_target = log_black_scholes_call_price_torch(x_lower, tau_lower, domain)
    lower_loss = torch.mean((model(batch["lower_x"], batch["lower_tau"]) - lower_target) ** 2)  # type: ignore[arg-type]

    x_upper, tau_upper = normalized_to_log_physical(batch["upper_x"], batch["upper_tau"], domain)  # type: ignore[arg-type]
    upper_target = log_black_scholes_call_price_torch(x_upper, tau_upper, domain)
    upper_loss = torch.mean((model(batch["upper_x"], batch["upper_tau"]) - upper_target) ** 2)  # type: ignore[arg-type]

    if weights.supervised > 0.0 and batch["supervised_x"] is not None and batch["supervised_tau"] is not None:
        x_supervised, tau_supervised = normalized_to_log_physical(batch["supervised_x"], batch["supervised_tau"], domain)  # type: ignore[arg-type]
        supervised_target = log_black_scholes_call_price_torch(x_supervised, tau_supervised, domain)
        supervised_loss = torch.mean((model(batch["supervised_x"], batch["supervised_tau"]) - supervised_target) ** 2)  # type: ignore[arg-type]
    else:
        supervised_loss = torch.zeros((), device=residual.device)

    total = weights.pde * pde_loss + weights.terminal * terminal_loss + weights.lower_boundary * lower_loss + weights.upper_boundary * upper_loss + weights.supervised * supervised_loss
    return LogPINNLossBreakdown(total, pde_loss, terminal_loss, lower_loss, upper_loss, supervised_loss)
