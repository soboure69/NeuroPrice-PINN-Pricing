from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from neuroprice.pinn.log_bs import LogBlackScholesDomain, LogBlackScholesPINN
from neuroprice.validation.black_scholes_ref import (
    black_scholes_call_delta_np,
    black_scholes_call_gamma_np,
    black_scholes_call_price_np,
    masked_relative_l2_error,
    relative_l2_error,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="artifacts/phase1_bs_pinn_log/bs_pinn_log.pt")
    parser.add_argument("--out", type=str, default="artifacts/phase1_bs_pinn_log/validation_error.png")
    parser.add_argument("--grid-size", type=int, default=80)
    parser.add_argument("--tau-min-regular", type=float, default=0.02)
    parser.add_argument("--strike-band", type=float, default=0.05)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("coordinate_transform") != "log_moneyness" or checkpoint.get("output_scale") != "normalized_by_K":
        raise ValueError("Checkpoint is not compatible with log-space validation. Retrain with scripts.train_pinn_bs_log.")

    domain = LogBlackScholesDomain(**checkpoint["domain"])
    model = LogBlackScholesPINN(
        hidden_dim=int(checkpoint["hidden_dim"]),
        hidden_layers=int(checkpoint["hidden_layers"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x = np.linspace(domain.x_min, domain.x_max, args.grid_size)
    tau = np.linspace(0.0, domain.T, args.grid_size)
    XX, TT = np.meshgrid(x, tau)
    SS = domain.K * np.exp(XX)

    x_norm_np = (XX - domain.x_min) / (domain.x_max - domain.x_min)
    tau_norm_np = TT / domain.T
    x_norm = torch.tensor(x_norm_np.reshape(-1, 1), dtype=torch.float32, device=device)
    tau_norm = torch.tensor(tau_norm_np.reshape(-1, 1), dtype=torch.float32, device=device)

    with torch.no_grad():
        pred_u = model(x_norm, tau_norm).cpu().numpy().reshape(args.grid_size, args.grid_size)
        pred = domain.K * pred_u

    ref = black_scholes_call_price_np(SS, TT, domain.K, domain.r, domain.sigma)
    rel_l2 = relative_l2_error(pred, ref)
    mask_after_maturity = TT > args.tau_min_regular
    mask_outside_strike = np.abs(SS - domain.K) / domain.K > args.strike_band
    mask_regular = mask_after_maturity & mask_outside_strike
    rel_l2_after_maturity = masked_relative_l2_error(pred, ref, mask_after_maturity)
    rel_l2_outside_strike = masked_relative_l2_error(pred, ref, mask_outside_strike)
    rel_l2_regular = masked_relative_l2_error(pred, ref, mask_regular)
    abs_err = np.abs(pred - ref)

    x_greek = np.linspace(np.log(0.2), np.log(1.8), args.grid_size)
    tau_greek = np.linspace(args.tau_min_regular, domain.T, args.grid_size)
    XX_greek, TT_greek = np.meshgrid(x_greek, tau_greek)
    SS_greek = domain.K * np.exp(XX_greek)
    x_norm_greek = torch.tensor(((XX_greek - domain.x_min) / (domain.x_max - domain.x_min)).reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=True)
    tau_norm_greek = torch.tensor((TT_greek / domain.T).reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=True)
    u_greek = model(x_norm_greek, tau_norm_greek)
    V_greek = domain.K * u_greek
    dV_dx_norm = torch.autograd.grad(V_greek, x_norm_greek, grad_outputs=torch.ones_like(V_greek), create_graph=True)[0]
    d2V_dx_norm2 = torch.autograd.grad(dV_dx_norm, x_norm_greek, grad_outputs=torch.ones_like(dV_dx_norm), create_graph=False)[0]

    x_scale = domain.x_max - domain.x_min
    S_greek_torch = torch.tensor(SS_greek.reshape(-1, 1), dtype=torch.float32, device=device)
    dV_dx = dV_dx_norm / x_scale
    d2V_dx2 = d2V_dx_norm2 / (x_scale * x_scale)
    delta_pred = (dV_dx / S_greek_torch).detach().cpu().numpy().reshape(args.grid_size, args.grid_size)
    gamma_pred = ((d2V_dx2 - dV_dx) / (S_greek_torch * S_greek_torch)).detach().cpu().numpy().reshape(args.grid_size, args.grid_size)
    delta_ref = black_scholes_call_delta_np(SS_greek, TT_greek, domain.K, domain.r, domain.sigma)
    gamma_ref = black_scholes_call_gamma_np(SS_greek, TT_greek, domain.K, domain.r, domain.sigma)
    delta_l2 = relative_l2_error(delta_pred, delta_ref)
    gamma_l2 = relative_l2_error(gamma_pred, gamma_ref)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        abs_err,
        origin="lower",
        aspect="auto",
        extent=[SS.min(), SS.max(), tau.min(), tau.max()],
    )
    ax.set_xlabel("S")
    ax.set_ylabel("tau = T - t")
    ax.set_title(f"Log-space PINN vs Black-Scholes absolute error | relative L2={rel_l2:.4f}")
    fig.colorbar(im, ax=ax, label="absolute error")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)

    print(f"Relative L2 error: {rel_l2:.6f}")
    print(f"Relative L2 error (tau > {args.tau_min_regular:g}): {rel_l2_after_maturity:.6f}")
    print(f"Relative L2 error (outside strike band {args.strike_band:g}): {rel_l2_outside_strike:.6f}")
    print(f"Relative L2 error (regular zone): {rel_l2_regular:.6f}")
    print(f"Delta relative L2 error: {delta_l2:.6f}")
    print(f"Gamma relative L2 error: {gamma_l2:.6f}")
    print(f"Saved error plot to {out_path}")


if __name__ == "__main__":
    main()
