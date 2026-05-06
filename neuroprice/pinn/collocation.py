from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CollocationBatch:
    interior_S: torch.Tensor
    interior_tau: torch.Tensor
    terminal_S: torch.Tensor
    terminal_tau: torch.Tensor
    lower_S: torch.Tensor
    lower_tau: torch.Tensor
    upper_S: torch.Tensor
    upper_tau: torch.Tensor
    supervised_S: torch.Tensor | None = None
    supervised_tau: torch.Tensor | None = None


def _mixed_interior_samples(
    *,
    n_interior: int,
    strike_norm: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_strike = n_interior // 3
    n_maturity = n_interior // 3
    n_uniform = n_interior - n_strike - n_maturity

    S_uniform = torch.rand(n_uniform, 1, device=device)
    tau_uniform = torch.rand(n_uniform, 1, device=device)

    S_strike = torch.clamp(strike_norm + 0.10 * torch.randn(n_strike, 1, device=device), 0.0, 1.0)
    tau_strike = torch.rand(n_strike, 1, device=device)

    S_maturity = torch.rand(n_maturity, 1, device=device)
    tau_maturity = torch.rand(n_maturity, 1, device=device) ** 2

    interior_S = torch.cat([S_uniform, S_strike, S_maturity], dim=0)
    interior_tau = torch.cat([tau_uniform, tau_strike, tau_maturity], dim=0)
    permutation = torch.randperm(n_interior, device=device)
    return interior_S[permutation], interior_tau[permutation]


def sample_black_scholes_batch(
    *,
    n_interior: int,
    n_terminal: int,
    n_boundary: int,
    n_supervised: int = 0,
    device: torch.device,
    strike_norm: float = 1.0 / 3.0,
    improved_sampling: bool = True,
) -> CollocationBatch:
    if improved_sampling:
        interior_S, interior_tau = _mixed_interior_samples(n_interior=n_interior, strike_norm=strike_norm, device=device)
    else:
        interior_S = torch.rand(n_interior, 1, device=device)
        interior_tau = torch.rand(n_interior, 1, device=device)

    terminal_S = torch.rand(n_terminal, 1, device=device)
    terminal_tau = torch.zeros(n_terminal, 1, device=device)

    lower_S = torch.zeros(n_boundary, 1, device=device)
    lower_tau = torch.rand(n_boundary, 1, device=device)

    upper_S = torch.ones(n_boundary, 1, device=device)
    upper_tau = torch.rand(n_boundary, 1, device=device)

    if n_supervised > 0:
        if improved_sampling:
            supervised_S, supervised_tau = _mixed_interior_samples(n_interior=n_supervised, strike_norm=strike_norm, device=device)
        else:
            supervised_S = torch.rand(n_supervised, 1, device=device)
            supervised_tau = torch.rand(n_supervised, 1, device=device)
    else:
        supervised_S = None
        supervised_tau = None

    return CollocationBatch(
        interior_S=interior_S,
        interior_tau=interior_tau,
        terminal_S=terminal_S,
        terminal_tau=terminal_tau,
        lower_S=lower_S,
        lower_tau=lower_tau,
        upper_S=upper_S,
        upper_tau=upper_tau,
        supervised_S=supervised_S,
        supervised_tau=supervised_tau,
    )
