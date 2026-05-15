from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ParametricBlackScholesDomain:
    S_min: float = 20.0
    S_max: float = 200.0
    K_min: float = 20.0
    K_max: float = 200.0
    sigma_min: float = 0.05
    sigma_max: float = 0.80
    r_min: float = 0.00
    r_max: float = 0.15
    T_min: float = 0.1
    T_max: float = 5.0
    x_min: float = -2.5
    x_max: float = 2.5


@dataclass(frozen=True)
class ParametricPINNLossWeights:
    pde: float = 1.0
    terminal: float = 10.0
    lower_boundary: float = 1.0
    upper_boundary: float = 1.0
    supervised: float = 0.1
    supervised_relative: float = 0.0
    relative_floor: float = 0.01


@dataclass(frozen=True)
class ParametricPINNLossBreakdown:
    total: torch.Tensor
    pde: torch.Tensor
    terminal: torch.Tensor
    lower_boundary: torch.Tensor
    upper_boundary: torch.Tensor
    supervised: torch.Tensor


class ParametricBlackScholesPINN(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        hidden_layers: int = 4,
        use_strike_input: bool = False,
        output_transform: str = "direct",
        fourier_features: int = 0,
    ) -> None:
        super().__init__()
        if output_transform not in {"direct", "terminal"}:
            raise ValueError("output_transform must be either 'direct' or 'terminal'")
        self.use_strike_input = use_strike_input
        self.output_transform = output_transform
        self.fourier_features = int(fourier_features)
        base_input_dim = 6 if use_strike_input else 5
        input_dim = base_input_dim if self.fourier_features <= 0 else base_input_dim * (1 + 2 * self.fourier_features)
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x_norm: torch.Tensor,
        tau_norm: torch.Tensor,
        sigma_norm: torch.Tensor,
        r_norm: torch.Tensor,
        T_norm: torch.Tensor,
        K_norm: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_strike_input:
            inputs = torch.cat([x_norm, tau_norm, sigma_norm, r_norm, T_norm, K_norm], dim=1)
        else:
            inputs = torch.cat([x_norm, tau_norm, sigma_norm, r_norm, T_norm], dim=1)
        if self.fourier_features > 0:
            frequencies = 2.0 ** torch.arange(self.fourier_features, device=inputs.device, dtype=inputs.dtype)
            angles = torch.pi * inputs.unsqueeze(-1) * frequencies
            inputs = torch.cat([inputs, torch.sin(angles).flatten(1), torch.cos(angles).flatten(1)], dim=1)
        raw = self.net(inputs)
        if self.output_transform == "direct":
            return raw

        x = _scale_unit(x_norm, ParametricBlackScholesDomain.x_min, ParametricBlackScholesDomain.x_max)
        payoff = torch.clamp(torch.exp(x) - 1.0, min=0.0)
        return payoff + tau_norm * raw


def _scale_unit(value_norm: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    return lower + (upper - lower) * value_norm


def normalized_to_parametric_physical(
    x_norm: torch.Tensor,
    tau_norm: torch.Tensor,
    sigma_norm: torch.Tensor,
    r_norm: torch.Tensor,
    T_norm: torch.Tensor,
    K_norm: torch.Tensor,
    domain: ParametricBlackScholesDomain,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = _scale_unit(x_norm, domain.x_min, domain.x_max)
    sigma = _scale_unit(sigma_norm, domain.sigma_min, domain.sigma_max)
    r = _scale_unit(r_norm, domain.r_min, domain.r_max)
    T = _scale_unit(T_norm, domain.T_min, domain.T_max)
    K = _scale_unit(K_norm, domain.K_min, domain.K_max)
    tau = T * tau_norm
    S = K * torch.exp(x)
    return x, tau, sigma, r, T, K, S


def parametric_black_scholes_call_price_torch(
    x: torch.Tensor,
    tau: torch.Tensor,
    sigma: torch.Tensor,
    r: torch.Tensor,
) -> torch.Tensor:
    tau_safe = torch.clamp(tau, min=1e-12)
    sigma_safe = torch.clamp(sigma, min=1e-12)
    sqrt_tau = torch.sqrt(tau_safe)
    d1 = (x + (r + 0.5 * sigma_safe**2) * tau_safe) / (sigma_safe * sqrt_tau)
    d2 = d1 - sigma_safe * sqrt_tau
    normal = torch.distributions.Normal(loc=0.0, scale=1.0)
    price_norm = torch.exp(x) * normal.cdf(d1) - torch.exp(-r * tau_safe) * normal.cdf(d2)
    payoff_norm = torch.clamp(torch.exp(x) - 1.0, min=0.0)
    return torch.where(tau <= 1e-12, payoff_norm, price_norm)


def sample_parametric_black_scholes_batch(
    *,
    n_interior: int,
    n_terminal: int,
    n_boundary: int,
    n_supervised: int,
    device: torch.device,
    improved_sampling: bool = True,
) -> dict[str, torch.Tensor | None]:
    def rand(n: int) -> torch.Tensor:
        return torch.rand(n, 1, device=device)

    def sample_x_tau(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not improved_sampling:
            return rand(n), rand(n)
        n_strike = n // 3
        n_maturity = n // 3
        n_uniform = n - n_strike - n_maturity
        x_uniform = rand(n_uniform)
        tau_uniform = rand(n_uniform)
        x_strike = torch.clamp(0.5 + 0.10 * torch.randn(n_strike, 1, device=device), 0.0, 1.0)
        tau_strike = rand(n_strike)
        x_maturity = rand(n_maturity)
        tau_maturity = rand(n_maturity) ** 2
        x = torch.cat([x_uniform, x_strike, x_maturity], dim=0)
        tau = torch.cat([tau_uniform, tau_strike, tau_maturity], dim=0)
        permutation = torch.randperm(n, device=device)
        return x[permutation], tau[permutation]

    def sample_params(n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return rand(n), rand(n), rand(n), rand(n)

    interior_x, interior_tau = sample_x_tau(n_interior)
    interior_sigma, interior_r, interior_T, interior_K = sample_params(n_interior)
    terminal_sigma, terminal_r, terminal_T, terminal_K = sample_params(n_terminal)
    lower_sigma, lower_r, lower_T, lower_K = sample_params(n_boundary)
    upper_sigma, upper_r, upper_T, upper_K = sample_params(n_boundary)

    if n_supervised > 0:
        supervised_x, supervised_tau = sample_x_tau(n_supervised)
        supervised_sigma, supervised_r, supervised_T, supervised_K = sample_params(n_supervised)
    else:
        supervised_x = supervised_tau = supervised_sigma = supervised_r = supervised_T = supervised_K = None

    return {
        "interior_x": interior_x,
        "interior_tau": interior_tau,
        "interior_sigma": interior_sigma,
        "interior_r": interior_r,
        "interior_T": interior_T,
        "interior_K": interior_K,
        "terminal_x": rand(n_terminal),
        "terminal_tau": torch.zeros(n_terminal, 1, device=device),
        "terminal_sigma": terminal_sigma,
        "terminal_r": terminal_r,
        "terminal_T": terminal_T,
        "terminal_K": terminal_K,
        "lower_x": torch.zeros(n_boundary, 1, device=device),
        "lower_tau": rand(n_boundary),
        "lower_sigma": lower_sigma,
        "lower_r": lower_r,
        "lower_T": lower_T,
        "lower_K": lower_K,
        "upper_x": torch.ones(n_boundary, 1, device=device),
        "upper_tau": rand(n_boundary),
        "upper_sigma": upper_sigma,
        "upper_r": upper_r,
        "upper_T": upper_T,
        "upper_K": upper_K,
        "supervised_x": supervised_x,
        "supervised_tau": supervised_tau,
        "supervised_sigma": supervised_sigma,
        "supervised_r": supervised_r,
        "supervised_T": supervised_T,
        "supervised_K": supervised_K,
    }


def _predict(
    model: nn.Module,
    prefix: str,
    batch: dict[str, torch.Tensor | None],
) -> torch.Tensor:
    return model(
        batch[f"{prefix}_x"],
        batch[f"{prefix}_tau"],
        batch[f"{prefix}_sigma"],
        batch[f"{prefix}_r"],
        batch[f"{prefix}_T"],
        batch[f"{prefix}_K"],
    )


def parametric_black_scholes_pde_residual(
    model: nn.Module,
    x_norm: torch.Tensor,
    tau_norm: torch.Tensor,
    sigma_norm: torch.Tensor,
    r_norm: torch.Tensor,
    T_norm: torch.Tensor,
    K_norm: torch.Tensor,
    domain: ParametricBlackScholesDomain,
) -> torch.Tensor:
    x_norm = x_norm.clone().detach().requires_grad_(True)
    tau_norm = tau_norm.clone().detach().requires_grad_(True)
    sigma_norm = sigma_norm.clone().detach()
    r_norm = r_norm.clone().detach()
    T_norm = T_norm.clone().detach()
    K_norm = K_norm.clone().detach()

    u = model(x_norm, tau_norm, sigma_norm, r_norm, T_norm, K_norm)
    du_dx_norm = torch.autograd.grad(u, x_norm, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    du_dtau_norm = torch.autograd.grad(u, tau_norm, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    d2u_dx_norm2 = torch.autograd.grad(du_dx_norm, x_norm, grad_outputs=torch.ones_like(du_dx_norm), create_graph=True)[0]

    _, _, sigma, r, T, _, _ = normalized_to_parametric_physical(x_norm, tau_norm, sigma_norm, r_norm, T_norm, K_norm, domain)
    x_scale = domain.x_max - domain.x_min
    du_dx = du_dx_norm / x_scale
    d2u_dx2 = d2u_dx_norm2 / (x_scale * x_scale)
    du_dtau = du_dtau_norm / T
    residual = du_dtau - 0.5 * sigma**2 * d2u_dx2 - (r - 0.5 * sigma**2) * du_dx + r * u
    return residual


def _target_from_batch(
    prefix: str,
    batch: dict[str, torch.Tensor | None],
    domain: ParametricBlackScholesDomain,
) -> torch.Tensor:
    x, tau, sigma, r, _, _, _ = normalized_to_parametric_physical(
        batch[f"{prefix}_x"],
        batch[f"{prefix}_tau"],
        batch[f"{prefix}_sigma"],
        batch[f"{prefix}_r"],
        batch[f"{prefix}_T"],
        batch[f"{prefix}_K"],
        domain,
    )
    return parametric_black_scholes_call_price_torch(x, tau, sigma, r)


def _relative_mse(pred: torch.Tensor, target: torch.Tensor, floor: float) -> torch.Tensor:
    scale = torch.clamp(torch.abs(target), min=floor)
    return torch.mean(((pred - target) / scale) ** 2)


def parametric_black_scholes_pinn_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor | None],
    domain: ParametricBlackScholesDomain,
    weights: ParametricPINNLossWeights | None = None,
) -> ParametricPINNLossBreakdown:
    if weights is None:
        weights = ParametricPINNLossWeights()

    residual = parametric_black_scholes_pde_residual(
        model,
        batch["interior_x"],
        batch["interior_tau"],
        batch["interior_sigma"],
        batch["interior_r"],
        batch["interior_T"],
        batch["interior_K"],
        domain,
    )
    pde_loss = torch.mean(residual**2)

    terminal_loss = torch.mean((_predict(model, "terminal", batch) - _target_from_batch("terminal", batch, domain)) ** 2)
    lower_loss = torch.mean((_predict(model, "lower", batch) - _target_from_batch("lower", batch, domain)) ** 2)
    upper_loss = torch.mean((_predict(model, "upper", batch) - _target_from_batch("upper", batch, domain)) ** 2)

    if weights.supervised > 0.0 and batch["supervised_x"] is not None:
        supervised_pred = _predict(model, "supervised", batch)
        supervised_target = _target_from_batch("supervised", batch, domain)
        supervised_loss = torch.mean((supervised_pred - supervised_target) ** 2)
        supervised_relative_loss = _relative_mse(supervised_pred, supervised_target, weights.relative_floor)
    else:
        supervised_loss = torch.zeros((), device=residual.device)
        supervised_relative_loss = torch.zeros((), device=residual.device)

    total = weights.pde * pde_loss + weights.terminal * terminal_loss + weights.lower_boundary * lower_loss + weights.upper_boundary * upper_loss + weights.supervised * supervised_loss + weights.supervised_relative * supervised_relative_loss
    return ParametricPINNLossBreakdown(total, pde_loss, terminal_loss, lower_loss, upper_loss, supervised_loss)
