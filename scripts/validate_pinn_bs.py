from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from neuroprice.pinn.losses import BlackScholesDomain
from neuroprice.pinn.models import BlackScholesPINN
from neuroprice.validation.black_scholes_ref import (
    black_scholes_call_delta_np,
    black_scholes_call_gamma_np,
    black_scholes_call_price_np,
    masked_relative_l2_error,
    relative_l2_error,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="artifacts/phase1_bs_pinn/bs_pinn.pt")
    parser.add_argument("--out", type=str, default="artifacts/phase1_bs_pinn/validation_error.png")
    parser.add_argument("--grid-size", type=int, default=60)
    parser.add_argument("--tau-min-regular", type=float, default=0.02)
    parser.add_argument("--strike-band", type=float, default=0.05)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("output_scale") != "normalized_by_S_max":
        raise ValueError("Checkpoint is not compatible with normalized validation. Retrain with scripts.train_pinn_bs.")

    domain = BlackScholesDomain(**checkpoint["domain"])
    model = BlackScholesPINN(
        hidden_dim=int(checkpoint["hidden_dim"]),
        hidden_layers=int(checkpoint["hidden_layers"]),
        output_transform=str(checkpoint.get("output_transform", "direct")),
        K=domain.K,
        r=domain.r,
        T=domain.T,
        S_max=domain.S_max,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    S = np.linspace(1e-6, domain.S_max, args.grid_size)
    tau = np.linspace(0.0, domain.T, args.grid_size)
    SS, TT = np.meshgrid(S, tau)

    S_norm = torch.tensor((SS.reshape(-1, 1) / domain.S_max), dtype=torch.float32, device=device)
    tau_norm = torch.tensor((TT.reshape(-1, 1) / domain.T), dtype=torch.float32, device=device)

    with torch.no_grad():
        pred_norm = model(S_norm, tau_norm).cpu().numpy().reshape(args.grid_size, args.grid_size)
        pred = domain.S_max * pred_norm

    ref = black_scholes_call_price_np(SS, TT, domain.K, domain.r, domain.sigma)
    rel_l2 = relative_l2_error(pred, ref)
    mask_after_maturity = TT > args.tau_min_regular
    mask_outside_strike = np.abs(SS - domain.K) / domain.K > args.strike_band
    mask_regular = mask_after_maturity & mask_outside_strike
    rel_l2_after_maturity = masked_relative_l2_error(pred, ref, mask_after_maturity)
    rel_l2_outside_strike = masked_relative_l2_error(pred, ref, mask_outside_strike)
    rel_l2_regular = masked_relative_l2_error(pred, ref, mask_regular)
    abs_err = np.abs(pred - ref)

    S_greek = np.linspace(max(1e-6, 0.2 * domain.K), min(domain.S_max, 1.8 * domain.K), args.grid_size)
    tau_greek = np.linspace(args.tau_min_regular, domain.T, args.grid_size)
    SS_greek, TT_greek = np.meshgrid(S_greek, tau_greek)
    S_norm_greek = torch.tensor((SS_greek.reshape(-1, 1) / domain.S_max), dtype=torch.float32, device=device, requires_grad=True)
    tau_norm_greek = torch.tensor((TT_greek.reshape(-1, 1) / domain.T), dtype=torch.float32, device=device, requires_grad=True)
    pred_norm_greek = model(S_norm_greek, tau_norm_greek)
    pred_greek = domain.S_max * pred_norm_greek
    dV_dS_norm = torch.autograd.grad(pred_greek, S_norm_greek, grad_outputs=torch.ones_like(pred_greek), create_graph=True)[0]
    d2V_dS_norm2 = torch.autograd.grad(dV_dS_norm, S_norm_greek, grad_outputs=torch.ones_like(dV_dS_norm), create_graph=False)[0]
    delta_pred = (dV_dS_norm / domain.S_max).detach().cpu().numpy().reshape(args.grid_size, args.grid_size)
    gamma_pred = (d2V_dS_norm2 / (domain.S_max * domain.S_max)).detach().cpu().numpy().reshape(args.grid_size, args.grid_size)
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
        extent=[S.min(), S.max(), tau.min(), tau.max()],
    )
    ax.set_xlabel("S")
    ax.set_ylabel("tau = T - t")
    ax.set_title(f"PINN vs Black-Scholes absolute error | relative L2={rel_l2:.4f}")
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
