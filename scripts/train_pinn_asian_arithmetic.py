from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from neuroprice.pinn.asian import AsianArithmeticPINN, AsianOptionDomain, AsianPINNLossWeights, asian_pinn_loss, sample_asian_batch
from neuroprice.validation.asian_ref import asian_arithmetic_call_mc_np


def supervised_monte_carlo_loss(
    model: AsianArithmeticPINN,
    domain: AsianOptionDomain,
    n_samples: int,
    n_paths: int,
    n_steps: int,
    chunk_size: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    S = rng.uniform(1e-6, domain.S_max, size=(n_samples, 1))
    A = rng.uniform(1e-6, domain.A_max, size=(n_samples, 1))
    tau = rng.uniform(0.0, domain.T, size=(n_samples, 1))
    target = asian_arithmetic_call_mc_np(S, A, tau, domain.K, domain.r, domain.sigma, n_paths, n_steps, seed, chunk_size) / domain.S_max
    S_norm = torch.tensor(S / domain.S_max, dtype=torch.float32, device=device)
    A_norm = torch.tensor(A / domain.A_max, dtype=torch.float32, device=device)
    tau_norm = torch.tensor(tau / domain.T, dtype=torch.float32, device=device)
    target_tensor = torch.tensor(target, dtype=torch.float32, device=device)
    pred = model(S_norm, A_norm, tau_norm)
    return torch.mean((pred - target_tensor) ** 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pretrain-epochs", type=int, default=0)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--pretrain-samples", type=int, default=512)
    parser.add_argument("--pretrain-mc-paths", type=int, default=4096)
    parser.add_argument("--pretrain-mc-steps", type=int, default=32)
    parser.add_argument("--pretrain-mc-chunk-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=5)
    parser.add_argument("--n-interior", type=int, default=8192)
    parser.add_argument("--n-terminal", type=int, default=4096)
    parser.add_argument("--n-boundary", type=int, default=4096)
    parser.add_argument("--n-supervised", type=int, default=0)
    parser.add_argument("--lbfgs-steps", type=int, default=0)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--uniform-sampling", action="store_true")
    parser.add_argument("--K", type=float, default=100.0)
    parser.add_argument("--r", type=float, default=0.05)
    parser.add_argument("--sigma", type=float, default=0.20)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--S-max", type=float, default=300.0)
    parser.add_argument("--A-max", type=float, default=300.0)
    parser.add_argument("--pde-weight", type=float, default=1.0)
    parser.add_argument("--terminal-weight", type=float, default=10.0)
    parser.add_argument("--lower-boundary-weight", type=float, default=1.0)
    parser.add_argument("--upper-boundary-weight", type=float, default=1.0)
    parser.add_argument("--supervised-weight", type=float, default=0.0)
    parser.add_argument("--out-dir", type=str, default="artifacts/phase3_asian_arithmetic")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    domain = AsianOptionDomain(K=args.K, r=args.r, sigma=args.sigma, T=args.T, S_max=args.S_max, A_max=args.A_max)
    weights = AsianPINNLossWeights(
        pde=args.pde_weight,
        terminal=args.terminal_weight,
        lower_boundary=args.lower_boundary_weight,
        upper_boundary=args.upper_boundary_weight,
        supervised=args.supervised_weight,
    )
    model = AsianArithmeticPINN(hidden_dim=args.hidden_dim, hidden_layers=args.hidden_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history: list[dict[str, float | str]] = []

    if args.pretrain_epochs > 0:
        pretrain_optimizer = torch.optim.Adam(model.parameters(), lr=args.pretrain_lr)
        for epoch in range(1, args.pretrain_epochs + 1):
            loss = supervised_monte_carlo_loss(
                model,
                domain,
                args.pretrain_samples,
                args.pretrain_mc_paths,
                args.pretrain_mc_steps,
                args.pretrain_mc_chunk_size,
                args.seed + epoch,
                device,
            )
            pretrain_optimizer.zero_grad()
            loss.backward()
            pretrain_optimizer.step()
            if epoch == 1 or epoch % 25 == 0 or epoch == args.pretrain_epochs:
                row = {"phase": "pretrain_mc", "epoch": float(epoch), "supervised": float(loss.detach().cpu())}
                history.append(row)
                print(f"pretrain_epoch={epoch:05d} supervised_mc={row['supervised']:.6e}")

    for epoch in range(1, args.epochs + 1):
        batch = sample_asian_batch(
            n_interior=args.n_interior,
            n_terminal=args.n_terminal,
            n_boundary=args.n_boundary,
            n_supervised=args.n_supervised,
            device=device,
            domain=domain,
            improved_sampling=not args.uniform_sampling,
        )
        losses = asian_pinn_loss(model, batch, domain, weights)
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
                f"epoch={epoch:05d} total={row['total']:.6e} pde={row['pde']:.6e} terminal={row['terminal']:.6e} "
                f"lower={row['lower_boundary']:.6e} upper={row['upper_boundary']:.6e} supervised={row['supervised']:.6e}"
            )

    if args.lbfgs_steps > 0:
        batch = sample_asian_batch(
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
            losses = asian_pinn_loss(model, batch, domain, weights)
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
        "option_type": "asian_arithmetic_call",
        "output_scale": "normalized_by_S_max",
        "training": {
            "pretrain_epochs": args.pretrain_epochs,
            "pretrain_lr": args.pretrain_lr,
            "pretrain_samples": args.pretrain_samples,
            "pretrain_mc_paths": args.pretrain_mc_paths,
            "pretrain_mc_steps": args.pretrain_mc_steps,
            "adam_epochs": args.epochs,
            "adam_lr": args.lr,
            "lbfgs_steps": args.lbfgs_steps,
            "lbfgs_lr": args.lbfgs_lr,
            "improved_sampling": not args.uniform_sampling,
            "n_supervised": args.n_supervised,
            "loss_weights": weights.__dict__,
        },
    }
    torch.save(checkpoint, out_dir / "asian_arithmetic_pinn.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved model to {out_dir / 'asian_arithmetic_pinn.pt'}")


if __name__ == "__main__":
    main()
