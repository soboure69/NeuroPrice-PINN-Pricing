from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from neuroprice.pinn.parametric_bs import (
    ParametricBlackScholesDomain,
    ParametricBlackScholesPINN,
    ParametricPINNLossWeights,
    parametric_black_scholes_call_price_torch,
    parametric_black_scholes_pinn_loss,
    normalized_to_parametric_physical,
    sample_parametric_black_scholes_batch,
)


def supervised_parametric_loss(
    model: ParametricBlackScholesPINN,
    domain: ParametricBlackScholesDomain,
    n_samples: int,
    device: torch.device,
    relative_weight: float,
    relative_floor: float,
) -> torch.Tensor:
    x_norm = torch.rand(n_samples, 1, device=device)
    tau_norm = torch.rand(n_samples, 1, device=device)
    sigma_norm = torch.rand(n_samples, 1, device=device)
    r_norm = torch.rand(n_samples, 1, device=device)
    T_norm = torch.rand(n_samples, 1, device=device)
    K_norm = torch.rand(n_samples, 1, device=device)
    x, tau, sigma, r, _, _, _ = normalized_to_parametric_physical(x_norm, tau_norm, sigma_norm, r_norm, T_norm, K_norm, domain)
    target = parametric_black_scholes_call_price_torch(x, tau, sigma, r)
    pred = model(x_norm, tau_norm, sigma_norm, r_norm, T_norm, K_norm)
    absolute_loss = torch.mean((pred - target) ** 2)
    scale = torch.clamp(torch.abs(target), min=relative_floor)
    relative_loss = torch.mean(((pred - target) / scale) ** 2)
    return absolute_loss + relative_weight * relative_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pretrain-epochs", type=int, default=0)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--pretrain-samples", type=int, default=16384)
    parser.add_argument("--pretrain-relative-weight", type=float, default=0.0)
    parser.add_argument("--relative-floor", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--use-strike-input", action="store_true")
    parser.add_argument("--output-transform", choices=["direct", "terminal"], default="direct")
    parser.add_argument("--fourier-features", type=int, default=0)
    parser.add_argument("--n-interior", type=int, default=4096)
    parser.add_argument("--n-terminal", type=int, default=2048)
    parser.add_argument("--n-boundary", type=int, default=2048)
    parser.add_argument("--n-supervised", type=int, default=4096)
    parser.add_argument("--lbfgs-steps", type=int, default=0)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--skip-pinn-finetune", action="store_true")
    parser.add_argument("--uniform-sampling", action="store_true")
    parser.add_argument("--pde-weight", type=float, default=1.0)
    parser.add_argument("--terminal-weight", type=float, default=10.0)
    parser.add_argument("--lower-boundary-weight", type=float, default=1.0)
    parser.add_argument("--upper-boundary-weight", type=float, default=1.0)
    parser.add_argument("--supervised-weight", type=float, default=0.1)
    parser.add_argument("--supervised-relative-weight", type=float, default=0.0)
    parser.add_argument("--out-dir", type=str, default="artifacts/phase2_bs_pinn_parametric")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    domain = ParametricBlackScholesDomain()
    weights = ParametricPINNLossWeights(
        pde=args.pde_weight,
        terminal=args.terminal_weight,
        lower_boundary=args.lower_boundary_weight,
        upper_boundary=args.upper_boundary_weight,
        supervised=args.supervised_weight,
        supervised_relative=args.supervised_relative_weight,
        relative_floor=args.relative_floor,
    )
    model = ParametricBlackScholesPINN(
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        use_strike_input=args.use_strike_input,
        output_transform=args.output_transform,
        fourier_features=args.fourier_features,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history: list[dict[str, float | str]] = []

    if args.pretrain_epochs > 0:
        pretrain_optimizer = torch.optim.Adam(model.parameters(), lr=args.pretrain_lr)
        for epoch in range(1, args.pretrain_epochs + 1):
            loss = supervised_parametric_loss(model, domain, args.pretrain_samples, device, args.pretrain_relative_weight, args.relative_floor)
            pretrain_optimizer.zero_grad()
            loss.backward()
            pretrain_optimizer.step()
            if epoch == 1 or epoch % 100 == 0 or epoch == args.pretrain_epochs:
                row = {"phase": "pretrain", "epoch": float(epoch), "supervised": float(loss.detach().cpu())}
                history.append(row)
                print(f"pretrain_epoch={epoch:05d} supervised={row['supervised']:.6e}")

    if not args.skip_pinn_finetune:
        for epoch in range(1, args.epochs + 1):
            batch = sample_parametric_black_scholes_batch(
                n_interior=args.n_interior,
                n_terminal=args.n_terminal,
                n_boundary=args.n_boundary,
                n_supervised=args.n_supervised,
                device=device,
                improved_sampling=not args.uniform_sampling,
            )
            losses = parametric_black_scholes_pinn_loss(model, batch, domain, weights)
            optimizer.zero_grad()
            losses.total.backward()
            optimizer.step()

            if epoch == 1 or epoch % 100 == 0 or epoch == args.epochs:
                row = {
                    "phase": "adam",
                    "epoch": float(epoch),
                    "total": float(losses.total.detach().cpu()),
                    "pde": float(losses.pde.detach().cpu()),
                    "terminal": float(losses.terminal.detach().cpu()),
                    "lower_boundary": float(losses.lower_boundary.detach().cpu()),
                    "upper_boundary": float(losses.upper_boundary.detach().cpu()),
                    "supervised": float(losses.supervised.detach().cpu()),
                }
                history.append(row)
                print(
                    f"epoch={epoch:05d} total={row['total']:.6e} pde={row['pde']:.6e} "
                    f"terminal={row['terminal']:.6e} lower={row['lower_boundary']:.6e} "
                    f"upper={row['upper_boundary']:.6e} supervised={row['supervised']:.6e}"
                )

    if args.lbfgs_steps > 0 and not args.skip_pinn_finetune:
        lbfgs_batch = sample_parametric_black_scholes_batch(
            n_interior=args.n_interior,
            n_terminal=args.n_terminal,
            n_boundary=args.n_boundary,
            n_supervised=args.n_supervised,
            device=device,
            improved_sampling=not args.uniform_sampling,
        )
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=args.lbfgs_lr,
            max_iter=args.lbfgs_steps,
            max_eval=args.lbfgs_steps * 2,
            history_size=50,
            line_search_fn="strong_wolfe",
        )
        calls = 0

        def closure() -> torch.Tensor:
            nonlocal calls
            lbfgs.zero_grad()
            losses = parametric_black_scholes_pinn_loss(model, lbfgs_batch, domain, weights)
            losses.total.backward()
            calls += 1
            if calls == 1 or calls % 25 == 0:
                row = {
                    "phase": "lbfgs",
                    "epoch": float(args.epochs),
                    "lbfgs_call": float(calls),
                    "total": float(losses.total.detach().cpu()),
                    "pde": float(losses.pde.detach().cpu()),
                    "terminal": float(losses.terminal.detach().cpu()),
                    "lower_boundary": float(losses.lower_boundary.detach().cpu()),
                    "upper_boundary": float(losses.upper_boundary.detach().cpu()),
                    "supervised": float(losses.supervised.detach().cpu()),
                }
                history.append(row)
                print(
                    f"lbfgs_call={calls:05d} total={row['total']:.6e} pde={row['pde']:.6e} "
                    f"terminal={row['terminal']:.6e} lower={row['lower_boundary']:.6e} "
                    f"upper={row['upper_boundary']:.6e} supervised={row['supervised']:.6e}"
                )
            return losses.total

        lbfgs.step(closure)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "domain": domain.__dict__,
        "hidden_dim": args.hidden_dim,
        "hidden_layers": args.hidden_layers,
        "use_strike_input": args.use_strike_input,
        "output_transform": args.output_transform,
        "fourier_features": args.fourier_features,
        "coordinate_transform": "parametric_log_moneyness",
        "output_scale": "normalized_by_K",
        "training": {
            "pretrain_epochs": args.pretrain_epochs,
            "pretrain_lr": args.pretrain_lr,
            "pretrain_samples": args.pretrain_samples,
            "pretrain_relative_weight": args.pretrain_relative_weight,
            "relative_floor": args.relative_floor,
            "adam_epochs": args.epochs,
            "adam_lr": args.lr,
            "skip_pinn_finetune": args.skip_pinn_finetune,
            "lbfgs_steps": args.lbfgs_steps,
            "lbfgs_lr": args.lbfgs_lr,
            "improved_sampling": not args.uniform_sampling,
            "n_supervised": args.n_supervised,
            "loss_weights": weights.__dict__,
        },
    }
    torch.save(checkpoint, out_dir / "bs_pinn_parametric.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved model to {out_dir / 'bs_pinn_parametric.pt'}")


if __name__ == "__main__":
    main()
