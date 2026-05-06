from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from neuroprice.pinn.collocation import sample_black_scholes_batch
from neuroprice.pinn.losses import BlackScholesDomain, PINNLossWeights, black_scholes_call_price_torch, black_scholes_pinn_loss
from neuroprice.pinn.models import BlackScholesPINN


def supervised_black_scholes_loss(model: BlackScholesPINN, domain: BlackScholesDomain, n_samples: int, device: torch.device) -> torch.Tensor:
    S_norm = torch.rand(n_samples, 1, device=device)
    tau_norm = torch.rand(n_samples, 1, device=device)
    S = domain.S_max * S_norm
    tau = domain.T * tau_norm
    target = black_scholes_call_price_torch(S, tau, domain) / domain.S_max
    pred = model(S_norm, tau_norm)
    return torch.mean((pred - target) ** 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pretrain-epochs", type=int, default=0)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--pretrain-samples", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--output-transform", choices=["direct", "constrained"], default="direct")
    parser.add_argument("--n-interior", type=int, default=2048)
    parser.add_argument("--n-terminal", type=int, default=1024)
    parser.add_argument("--n-boundary", type=int, default=1024)
    parser.add_argument("--lbfgs-steps", type=int, default=0)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--uniform-sampling", action="store_true")
    parser.add_argument("--K", type=float, default=100.0)
    parser.add_argument("--r", type=float, default=0.05)
    parser.add_argument("--sigma", type=float, default=0.20)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--S-max", type=float, default=300.0)
    parser.add_argument("--pde-weight", type=float, default=1.0)
    parser.add_argument("--terminal-weight", type=float, default=10.0)
    parser.add_argument("--lower-boundary-weight", type=float, default=1.0)
    parser.add_argument("--upper-boundary-weight", type=float, default=1.0)
    parser.add_argument("--supervised-weight", type=float, default=0.0)
    parser.add_argument("--n-supervised", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="artifacts/phase1_bs_pinn")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    domain = BlackScholesDomain(K=args.K, r=args.r, sigma=args.sigma, T=args.T, S_max=args.S_max)
    weights = PINNLossWeights(
        pde=args.pde_weight,
        terminal=args.terminal_weight,
        lower_boundary=args.lower_boundary_weight,
        upper_boundary=args.upper_boundary_weight,
        supervised=args.supervised_weight,
    )
    model = BlackScholesPINN(
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        output_transform=args.output_transform,
        K=domain.K,
        r=domain.r,
        T=domain.T,
        S_max=domain.S_max,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    strike_norm = domain.K / domain.S_max

    history: list[dict[str, float]] = []

    if args.pretrain_epochs > 0:
        pretrain_optimizer = torch.optim.Adam(model.parameters(), lr=args.pretrain_lr)
        for pretrain_epoch in range(1, args.pretrain_epochs + 1):
            pretrain_loss = supervised_black_scholes_loss(model, domain, args.pretrain_samples, device)
            pretrain_optimizer.zero_grad()
            pretrain_loss.backward()
            pretrain_optimizer.step()

            if pretrain_epoch == 1 or pretrain_epoch % 100 == 0 or pretrain_epoch == args.pretrain_epochs:
                row = {
                    "phase": "pretrain",
                    "epoch": float(pretrain_epoch),
                    "supervised": float(pretrain_loss.detach().cpu()),
                }
                history.append(row)
                print(f"pretrain_epoch={pretrain_epoch:05d} supervised={row['supervised']:.6e}")

    for epoch in range(1, args.epochs + 1):
        batch = sample_black_scholes_batch(
            n_interior=args.n_interior,
            n_terminal=args.n_terminal,
            n_boundary=args.n_boundary,
            n_supervised=args.n_supervised,
            device=device,
            strike_norm=strike_norm,
            improved_sampling=not args.uniform_sampling,
        )
        losses = black_scholes_pinn_loss(model, batch, domain, weights)

        optimizer.zero_grad()
        losses.total.backward()
        optimizer.step()

        if epoch == 1 or epoch % 100 == 0 or epoch == args.epochs:
            row = {
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
        lbfgs_batch = sample_black_scholes_batch(
            n_interior=args.n_interior,
            n_terminal=args.n_terminal,
            n_boundary=args.n_boundary,
            n_supervised=args.n_supervised,
            device=device,
            strike_norm=strike_norm,
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
        lbfgs_calls = 0

        def closure() -> torch.Tensor:
            nonlocal lbfgs_calls
            lbfgs.zero_grad()
            losses = black_scholes_pinn_loss(model, lbfgs_batch, domain, weights)
            losses.total.backward()
            lbfgs_calls += 1
            if lbfgs_calls == 1 or lbfgs_calls % 25 == 0:
                row = {
                    "epoch": float(args.epochs),
                    "lbfgs_call": float(lbfgs_calls),
                    "total": float(losses.total.detach().cpu()),
                    "pde": float(losses.pde.detach().cpu()),
                    "terminal": float(losses.terminal.detach().cpu()),
                    "lower_boundary": float(losses.lower_boundary.detach().cpu()),
                    "upper_boundary": float(losses.upper_boundary.detach().cpu()),
                    "supervised": float(losses.supervised.detach().cpu()),
                }
                history.append(row)
                print(
                    f"lbfgs_call={lbfgs_calls:05d} total={row['total']:.6e} pde={row['pde']:.6e} "
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
        "output_transform": args.output_transform,
        "output_scale": "normalized_by_S_max",
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
    torch.save(checkpoint, out_dir / "bs_pinn.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved model to {out_dir / 'bs_pinn.pt'}")


if __name__ == "__main__":
    main()
