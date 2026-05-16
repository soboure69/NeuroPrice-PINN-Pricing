from __future__ import annotations

import numpy as np
import torch

from neuroprice.pinn.collocation import sample_black_scholes_batch
from neuroprice.pinn.barrier import BarrierOptionDomain, DownAndOutBarrierPINN, barrier_pinn_loss, sample_barrier_batch
from neuroprice.pinn.asian import AsianArithmeticPINN, AsianOptionDomain, asian_pinn_loss, sample_asian_batch
from neuroprice.pinn.asian_surrogate import AsianArithmeticSurrogate
from neuroprice.pinn.lookback_surrogate import LookbackFloatingCallSurrogate
from neuroprice.pinn.log_bs import (
    LogBlackScholesDomain,
    LogBlackScholesPINN,
    log_black_scholes_call_price_torch,
    log_black_scholes_pinn_loss,
    normalized_to_log_physical,
    sample_log_black_scholes_batch,
)
from neuroprice.pinn.losses import BlackScholesDomain, PINNLossWeights, black_scholes_pinn_loss
from neuroprice.pinn.models import BlackScholesPINN
from neuroprice.pinn.parametric_bs import (
    ParametricBlackScholesDomain,
    ParametricBlackScholesPINN,
    parametric_black_scholes_pinn_loss,
    sample_parametric_black_scholes_batch,
)
from neuroprice.validation.black_scholes_ref import (
    black_scholes_call_delta_np,
    black_scholes_call_gamma_np,
    black_scholes_call_price_np,
    masked_relative_l2_error,
    relative_l2_error,
)
from neuroprice.validation.barrier_ref import down_and_out_call_price_np
from neuroprice.validation.asian_ref import asian_arithmetic_call_mc_np
from neuroprice.validation.lookback_ref import lookback_floating_call_mc_np


def test_pinn_forward_shape() -> None:
    model = BlackScholesPINN(hidden_dim=8, hidden_layers=2)
    S = torch.rand(5, 1)
    tau = torch.rand(5, 1)
    out = model(S, tau)
    assert out.shape == (5, 1)


def test_log_pinn_forward_shape() -> None:
    model = LogBlackScholesPINN(hidden_dim=8, hidden_layers=2)
    x = torch.rand(5, 1)
    tau = torch.rand(5, 1)
    out = model(x, tau)
    assert out.shape == (5, 1)


def test_parametric_pinn_forward_shape() -> None:
    model = ParametricBlackScholesPINN(hidden_dim=8, hidden_layers=2)
    inputs = [torch.rand(5, 1) for _ in range(6)]
    out = model(*inputs)
    assert out.shape == (5, 1)


def test_barrier_pinn_forward_shape() -> None:
    model = DownAndOutBarrierPINN(hidden_dim=8, hidden_layers=2)
    S = torch.rand(5, 1)
    tau = torch.rand(5, 1)
    out = model(S, tau)
    assert out.shape == (5, 1)


def test_asian_pinn_forward_shape() -> None:
    model = AsianArithmeticPINN(hidden_dim=8, hidden_layers=2)
    S = torch.rand(5, 1)
    A = torch.rand(5, 1)
    tau = torch.rand(5, 1)
    out = model(S, A, tau)
    assert out.shape == (5, 1)


def test_asian_surrogate_forward_shape() -> None:
    model = AsianArithmeticSurrogate(hidden_dim=8, hidden_layers=2)
    S = torch.rand(5, 1)
    tau = torch.rand(5, 1)
    out = model(S, tau)
    assert out.shape == (5, 1)


def test_lookback_surrogate_forward_shape() -> None:
    model = LookbackFloatingCallSurrogate(hidden_dim=8, hidden_layers=2)
    S = torch.rand(5, 1)
    tau = torch.rand(5, 1)
    out = model(S, tau)
    assert out.shape == (5, 1)


def test_constrained_pinn_output_structure() -> None:
    model = BlackScholesPINN(hidden_dim=8, hidden_layers=2, output_transform="constrained", K=100.0, r=0.05, T=1.0, S_max=300.0)
    tau = torch.tensor([[0.25], [0.75]])
    lower = model(torch.zeros_like(tau), tau)
    assert torch.allclose(lower, torch.zeros_like(lower), atol=1e-6)

    S_norm = torch.tensor([[0.2], [0.5], [0.8]])
    terminal = model(S_norm, torch.zeros_like(S_norm))
    terminal_target = torch.clamp(300.0 * S_norm - 100.0, min=0.0) / 300.0
    assert torch.allclose(terminal, terminal_target, atol=1e-6)

    upper_tau = torch.tensor([[0.25], [0.75]])
    upper = model(torch.ones_like(upper_tau), upper_tau)
    upper_target = (300.0 - 100.0 * torch.exp(-0.05 * upper_tau)) / 300.0
    assert torch.allclose(upper, upper_target, atol=1e-6)


def test_pinn_loss_is_finite() -> None:
    model = BlackScholesPINN(hidden_dim=8, hidden_layers=2)
    batch = sample_black_scholes_batch(n_interior=8, n_terminal=8, n_boundary=8, device=torch.device("cpu"))
    losses = black_scholes_pinn_loss(model, batch, BlackScholesDomain())
    assert torch.isfinite(losses.total)


def test_supervised_pinn_loss_is_finite() -> None:
    model = BlackScholesPINN(hidden_dim=8, hidden_layers=2)
    batch = sample_black_scholes_batch(n_interior=8, n_terminal=8, n_boundary=8, n_supervised=8, device=torch.device("cpu"))
    losses = black_scholes_pinn_loss(model, batch, BlackScholesDomain(), PINNLossWeights(supervised=1.0))
    assert torch.isfinite(losses.total)
    assert torch.isfinite(losses.supervised)
    assert losses.supervised >= 0.0


def test_log_pinn_loss_is_finite() -> None:
    model = LogBlackScholesPINN(hidden_dim=8, hidden_layers=2)
    batch = sample_log_black_scholes_batch(n_interior=8, n_terminal=8, n_boundary=8, n_supervised=8, device=torch.device("cpu"))
    losses = log_black_scholes_pinn_loss(model, batch, LogBlackScholesDomain())
    assert torch.isfinite(losses.total)


def test_parametric_pinn_loss_is_finite() -> None:
    model = ParametricBlackScholesPINN(hidden_dim=8, hidden_layers=2)
    batch = sample_parametric_black_scholes_batch(n_interior=8, n_terminal=8, n_boundary=8, n_supervised=8, device=torch.device("cpu"))
    losses = parametric_black_scholes_pinn_loss(model, batch, ParametricBlackScholesDomain())
    assert torch.isfinite(losses.total)


def test_barrier_pinn_loss_is_finite() -> None:
    domain = BarrierOptionDomain()
    model = DownAndOutBarrierPINN(hidden_dim=8, hidden_layers=2, B=domain.B, K=domain.K, r=domain.r, T=domain.T, S_max=domain.S_max)
    batch = sample_barrier_batch(n_interior=8, n_terminal=8, n_boundary=8, n_supervised=8, device=torch.device("cpu"), domain=domain)
    losses = barrier_pinn_loss(model, batch, domain)
    assert torch.isfinite(losses.total)


def test_asian_pinn_loss_is_finite() -> None:
    domain = AsianOptionDomain()
    model = AsianArithmeticPINN(hidden_dim=8, hidden_layers=2)
    batch = sample_asian_batch(n_interior=8, n_terminal=8, n_boundary=8, n_supervised=0, device=torch.device("cpu"), domain=domain)
    losses = asian_pinn_loss(model, batch, domain)
    assert torch.isfinite(losses.total)


def test_improved_sampling_shapes_and_ranges() -> None:
    batch = sample_black_scholes_batch(
        n_interior=99,
        n_terminal=16,
        n_boundary=12,
        device=torch.device("cpu"),
        strike_norm=1.0 / 3.0,
        improved_sampling=True,
    )
    assert batch.interior_S.shape == (99, 1)
    assert batch.interior_tau.shape == (99, 1)
    assert torch.all((batch.interior_S >= 0.0) & (batch.interior_S <= 1.0))
    assert torch.all((batch.interior_tau >= 0.0) & (batch.interior_tau <= 1.0))


def test_supervised_sampling_shapes_and_ranges() -> None:
    batch = sample_black_scholes_batch(
        n_interior=8,
        n_terminal=8,
        n_boundary=8,
        n_supervised=16,
        device=torch.device("cpu"),
    )
    assert batch.supervised_S is not None
    assert batch.supervised_tau is not None
    assert batch.supervised_S.shape == (16, 1)
    assert batch.supervised_tau.shape == (16, 1)
    assert torch.all((batch.supervised_S >= 0.0) & (batch.supervised_S <= 1.0))
    assert torch.all((batch.supervised_tau >= 0.0) & (batch.supervised_tau <= 1.0))


def test_black_scholes_reference_vectorized() -> None:
    S = np.array([90.0, 100.0, 110.0])
    tau = np.array([1.0, 1.0, 1.0])
    ref = black_scholes_call_price_np(S, tau, K=100.0, r=0.05, sigma=0.20)
    assert ref.shape == (3,)
    assert np.all(ref >= 0.0)
    assert relative_l2_error(ref, ref) < 1e-12


def test_down_and_out_barrier_reference_vectorized() -> None:
    S = np.array([60.0, 80.0, 120.0])
    tau = np.array([1.0, 1.0, 1.0])
    ref = down_and_out_call_price_np(S, tau, K=100.0, B=70.0, r=0.05, sigma=0.20)
    assert ref.shape == (3,)
    assert ref[0] == 0.0
    assert np.all(ref >= 0.0)


def test_asian_arithmetic_reference_shape_and_terminal() -> None:
    S = np.array([80.0, 100.0, 120.0])
    A = np.array([90.0, 110.0, 130.0])
    tau = np.zeros_like(S)
    prices = asian_arithmetic_call_mc_np(S, A, tau, K=100.0, r=0.05, sigma=0.2, n_paths=200, n_steps=4, seed=123, chunk_size=50)
    assert prices.shape == (3, 1)
    assert np.allclose(prices.reshape(-1), np.maximum(A - 100.0, 0.0))


def test_lookback_floating_reference_shape_and_terminal() -> None:
    S = np.array([80.0, 100.0, 120.0])
    tau = np.zeros_like(S)
    prices = lookback_floating_call_mc_np(S, tau, r=0.05, sigma=0.2, n_paths=200, n_steps=4, seed=123, chunk_size=50)
    assert prices.shape == (3, 1)
    assert np.allclose(prices.reshape(-1), np.zeros_like(S))


def test_log_black_scholes_reference_torch_is_finite() -> None:
    domain = LogBlackScholesDomain()
    x_norm = torch.tensor([[0.4], [0.5], [0.6]])
    tau_norm = torch.ones_like(x_norm)
    x, tau = normalized_to_log_physical(x_norm, tau_norm, domain)
    ref = log_black_scholes_call_price_torch(x, tau, domain)
    assert ref.shape == (3, 1)
    assert torch.all(torch.isfinite(ref))
    assert torch.all(ref >= 0.0)


def test_black_scholes_greek_references_vectorized() -> None:
    S = np.array([90.0, 100.0, 110.0])
    tau = np.array([1.0, 1.0, 1.0])
    delta = black_scholes_call_delta_np(S, tau, K=100.0, r=0.05, sigma=0.20)
    gamma = black_scholes_call_gamma_np(S, tau, K=100.0, r=0.05, sigma=0.20)
    assert delta.shape == (3,)
    assert gamma.shape == (3,)
    assert np.all((delta >= 0.0) & (delta <= 1.0))
    assert np.all(gamma >= 0.0)


def test_masked_relative_l2_error() -> None:
    pred = np.array([1.0, 2.0, 3.0])
    ref = np.array([1.0, 2.0, 4.0])
    mask = np.array([True, True, False])
    assert masked_relative_l2_error(pred, ref, mask) < 1e-12
