from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from neuroprice.pinn.barrier import (
    BarrierOptionDomain,
    BarrierPINNLossWeights,
    DownAndOutBarrierPINN,
    barrier_pinn_loss,
    down_and_out_call_price_torch,
    normalized_to_barrier_physical,
    sample_barrier_batch,
)


def supervised_barrier_loss(model: DownAndOutBarrierPINN, domain: BarrierOptionDomain, n_samples: int, device: torch.device) -> torch.Tensor:
    barrier_norm = domain.B / domain.S_max
    S_norm = barrier_norm + (1.0 - barrier_norm) * torch.rand(n_samples, 1, device=device)
    tau_norm = torch.rand(n_samples, 1, device=device)
    S, tau = normalized_to_barrier_physical(S_norm, tau_norm, domain)
    target = down_and_out_call_price_torch(S, tau, domain) / domain.S_max
    pred = model(S_norm, tau_norm)
    return torch.mean((pred - target) ** 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pretrain-epochs", type=int, default=2000)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--pretrain-samples", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--output-transform", choices=["direct", "barrier"], default="barrier")
    parser.add_argument("--n-interior", type=int, default=4096)
    parser.add_argument("--n-terminal", type=int, default=2048)
    parser.add_argument("--n-boundary", type=int, default=2048)
    parser.add_argument("--n-supervised", type=int, default=4096)
    parser.add_argument("--lbfgs-steps", type=int, default=0)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--uniform-sampling", action="store_true")
    parser.add_argument("--K", type=float, default=100.0)
    parser.add_argument("--B", type=float, default=70.0)
    parser.add_argument("--r", type=float, default=0.05)
    parser.add_argument("--sigma", type=float, default=0.20)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--S-max", type=float, default=300.0)
    parser.add_argument("--pde-weight", type=float, default=1.0)
    parser.add_argument("--terminal-weight", type=float, default=10.0)
    parser.add_argument("--barrier-weight", type=float, default=10.0)
    parser.add_argument("--upper-boundary-weight", type=float, default=1.0)
    parser.add_argument("--supervised-weight", type=float, default=0.1)
    parser.add_argument("--near-barrier-supervised-weight", type=float, default=2.0)
    parser.add_argument("--out-dir", type=str, default="artifacts/phase3_down_out_barrier")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    domain = BarrierOptionDomain(K=args.K, B=args.B, r=args.r, sigma=args.sigma, T=args.T, S_max=args.S_max)
    weights = BarrierPINNLossWeights(
        pde=args.pde_weight,
        terminal=args.terminal_weight,
        barrier=args.barrier_weight,
        upper_boundary=args.upper_boundary_weight,
        supervised=args.supervised_weight,
        near_barrier_supervised=args.near_barrier_supervised_weight,
    )
    model = DownAndOutBarrierPINN(
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        output_transform=args.output_transform,
        K=domain.K,
        B=domain.B,
        r=domain.r,
        T=domain.T,
        S_max=domain.S_max,
    ).to(device)
    history: list[dict[str, float | str]] = []

    if args.pretrain_epochs > 0:
        pretrain_optimizer = torch.optim.Adam(model.parameters(), lr=args.pretrain_lr)
        for epoch in range(1, args.pretrain_epochs + 1):
            loss = supervised_barrier_loss(model, domain, args.pretrain_samples, device)
            pretrain_optimizer.zero_grad()
            loss.backward()
            pretrain_optimizer.step()
            if epoch == 1 or epoch % 100 == 0 or epoch == args.pretrain_epochs:
                row = {"phase": "pretrain", "epoch": float(epoch), "supervised": float(loss.detach().cpu())}
                history.append(row)
                print(f"pretrain_epoch={epoch:05d} supervised={row['supervised']:.6e}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        batch = sample_barrier_batch(
            n_interior=args.n_interior,
            n_terminal=args.n_terminal,
            n_boundary=args.n_boundary,
            n_supervised=args.n_supervised,
            device=device,
            domain=domain,
            improved_sampling=not args.uniform_sampling,
        )
        losses = barrier_pinn_loss(model, batch, domain, weights)
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
                "barrier": float(losses.barrier.detach().cpu()),
                "upper_boundary": float(losses.upper_boundary.detach().cpu()),
                "supervised": float(losses.supervised.detach().cpu()),
            }
            history.append(row)
            print(
                f"epoch={epoch:05d} total={row['total']:.6e} pde={row['pde']:.6e} terminal={row['terminal']:.6e} "
                f"barrier={row['barrier']:.6e} upper={row['upper_boundary']:.6e} supervised={row['supervised']:.6e}"
            )

    if args.lbfgs_steps > 0:
        batch = sample_barrier_batch(
            n_interior=args.n_interior,
            n_terminal=args.n_terminal,
            n_boundary=args.n_boundary,
            n_supervised=args.n_supervised,
            device=device,
            domain=domain,
            improved_sampling=not args.uniform_sampling,
        )
        lbfgs = torch.optim.LBFGS(model.parameters(), lr=args.lbfgs_lr, max_iter=args.lbfgs_steps, max_eval=args.lbfgs_steps * 2, history_size=50, line_search_fn="strong_wolfe")
        calls = 0

        def closure() -> torch.Tensor:
            nonlocal calls
            lbfgs.zero_grad()
            losses = barrier_pinn_loss(model, batch, domain, weights)
            losses.total.backward()
            calls += 1
            if calls == 1 or calls % 25 == 0:
                print(f"lbfgs_call={calls:05d} total={float(losses.total.detach().cpu()):.6e}")
            return losses.total

        lbfgs.step(closure)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "domain": domain.__dict__,
        "hidden_dim": args.hidden_dim,
        "hidden_layers": args.hidden_layers,
        "output_transform": args.output_transform,
        "option_type": "down_and_out_call",
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
    torch.save(checkpoint, out_dir / "down_out_barrier_pinn.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved model to {out_dir / 'down_out_barrier_pinn.pt'}")


if __name__ == "__main__":
    main()
