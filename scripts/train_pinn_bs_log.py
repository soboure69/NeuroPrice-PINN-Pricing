from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from neuroprice.pinn.log_bs import (
    LogBlackScholesDomain,
    LogBlackScholesPINN,
    LogPINNLossWeights,
    log_black_scholes_call_price_torch,
    log_black_scholes_pinn_loss,
    normalized_to_log_physical,
    sample_log_black_scholes_batch,
)


def supervised_log_loss(model: LogBlackScholesPINN, domain: LogBlackScholesDomain, n_samples: int, device: torch.device) -> torch.Tensor:
    x_norm = torch.rand(n_samples, 1, device=device)
    tau_norm = torch.rand(n_samples, 1, device=device)
    x, tau = normalized_to_log_physical(x_norm, tau_norm, domain)
    target = log_black_scholes_call_price_torch(x, tau, domain)
    pred = model(x_norm, tau_norm)
    return torch.mean((pred - target) ** 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pretrain-epochs", type=int, default=3000)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--pretrain-samples", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--hidden-layers", type=int, default=5)
    parser.add_argument("--n-interior", type=int, default=4096)
    parser.add_argument("--n-terminal", type=int, default=2048)
    parser.add_argument("--n-boundary", type=int, default=2048)
    parser.add_argument("--n-supervised", type=int, default=4096)
    parser.add_argument("--lbfgs-steps", type=int, default=700)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--uniform-sampling", action="store_true")
    parser.add_argument("--K", type=float, default=100.0)
    parser.add_argument("--r", type=float, default=0.05)
    parser.add_argument("--sigma", type=float, default=0.20)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--x-min", type=float, default=-5.0)
    parser.add_argument("--x-max", type=float, default=2.0)
    parser.add_argument("--pde-weight", type=float, default=1.0)
    parser.add_argument("--terminal-weight", type=float, default=10.0)
    parser.add_argument("--lower-boundary-weight", type=float, default=1.0)
    parser.add_argument("--upper-boundary-weight", type=float, default=1.0)
    parser.add_argument("--supervised-weight", type=float, default=0.01)
    parser.add_argument("--out-dir", type=str, default="artifacts/phase1_bs_pinn_log")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    domain = LogBlackScholesDomain(K=args.K, r=args.r, sigma=args.sigma, T=args.T, x_min=args.x_min, x_max=args.x_max)
    weights = LogPINNLossWeights(
        pde=args.pde_weight,
        terminal=args.terminal_weight,
        lower_boundary=args.lower_boundary_weight,
        upper_boundary=args.upper_boundary_weight,
        supervised=args.supervised_weight,
    )
    model = LogBlackScholesPINN(hidden_dim=args.hidden_dim, hidden_layers=args.hidden_layers).to(device)
    history: list[dict[str, float | str]] = []

    if args.pretrain_epochs > 0:
        pretrain_optimizer = torch.optim.Adam(model.parameters(), lr=args.pretrain_lr)
        for epoch in range(1, args.pretrain_epochs + 1):
            loss = supervised_log_loss(model, domain, args.pretrain_samples, device)
            pretrain_optimizer.zero_grad()
            loss.backward()
            pretrain_optimizer.step()
            if epoch == 1 or epoch % 100 == 0 or epoch == args.pretrain_epochs:
                row = {"phase": "pretrain", "epoch": float(epoch), "supervised": float(loss.detach().cpu())}
                history.append(row)
                print(f"pretrain_epoch={epoch:05d} supervised={row['supervised']:.6e}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        batch = sample_log_black_scholes_batch(
            n_interior=args.n_interior,
            n_terminal=args.n_terminal,
            n_boundary=args.n_boundary,
            n_supervised=args.n_supervised,
            device=device,
            improved_sampling=not args.uniform_sampling,
        )
        losses = log_black_scholes_pinn_loss(model, batch, domain, weights)
        optimizer.zero_grad()
        losses.total.backward()
        optimizer.step()
        if epoch == 1 or epoch % 100 == 0 or epoch == args.epochs:
            row = {
                "phase": "pinn",
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

    if args.lbfgs_steps > 0:
        lbfgs_batch = sample_log_black_scholes_batch(
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
            losses = log_black_scholes_pinn_loss(model, lbfgs_batch, domain, weights)
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
        "coordinate_transform": "log_moneyness",
        "output_scale": "normalized_by_K",
        "training": {
            "pretrain_epochs": args.pretrain_epochs,
            "pretrain_lr": args.pretrain_lr,
            "pretrain_samples": args.pretrain_samples,
            "adam_epochs": args.epochs,
            "adam_lr": args.lr,
            "lbfgs_steps": args.lbfgs_steps,
            "lbfgs_lr": args.lbfgs_lr,
            "improved_sampling": not args.uniform_sampling,
            "n_supervised": args.n_supervised,
            "loss_weights": weights.__dict__,
        },
    }
    torch.save(checkpoint, out_dir / "bs_pinn_log.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved model to {out_dir / 'bs_pinn_log.pt'}")


if __name__ == "__main__":
    main()
