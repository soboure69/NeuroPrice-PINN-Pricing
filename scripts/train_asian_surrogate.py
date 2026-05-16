from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from neuroprice.pinn.asian_surrogate import AsianArithmeticSurrogate, AsianSurrogateDomain
from neuroprice.validation.asian_ref import asian_arithmetic_call_mc_np


def supervised_loss(
    model: AsianArithmeticSurrogate,
    domain: AsianSurrogateDomain,
    n_samples: int,
    n_paths: int,
    n_steps: int,
    chunk_size: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    S = rng.uniform(1e-6, domain.S_max, size=(n_samples, 1))
    tau = rng.uniform(0.0, domain.T, size=(n_samples, 1))
    target = asian_arithmetic_call_mc_np(S, S, tau, domain.K, domain.r, domain.sigma, n_paths, n_steps, seed, chunk_size) / domain.S_max
    S_norm = torch.tensor(S / domain.S_max, dtype=torch.float32, device=device)
    tau_norm = torch.tensor(tau / domain.T, dtype=torch.float32, device=device)
    target_tensor = torch.tensor(target, dtype=torch.float32, device=device)
    pred = model(S_norm, tau_norm)
    abs_loss = torch.mean((pred - target_tensor) ** 2)
    rel_loss = torch.mean(((pred - target_tensor) / torch.clamp(target_tensor.abs(), min=1.0 / domain.S_max)) ** 2)
    return abs_loss + 0.1 * rel_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--mc-paths", type=int, default=4096)
    parser.add_argument("--mc-steps", type=int, default=64)
    parser.add_argument("--mc-chunk-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=5)
    parser.add_argument("--K", type=float, default=100.0)
    parser.add_argument("--r", type=float, default=0.05)
    parser.add_argument("--sigma", type=float, default=0.20)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--S-max", type=float, default=300.0)
    parser.add_argument("--out-dir", type=str, default="artifacts/phase3_asian_surrogate")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    domain = AsianSurrogateDomain(K=args.K, r=args.r, sigma=args.sigma, T=args.T, S_max=args.S_max)
    model = AsianArithmeticSurrogate(hidden_dim=args.hidden_dim, hidden_layers=args.hidden_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        loss = supervised_loss(model, domain, args.samples, args.mc_paths, args.mc_steps, args.mc_chunk_size, args.seed + epoch, device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            row = {"epoch": float(epoch), "loss": float(loss.detach().cpu())}
            history.append(row)
            print(f"epoch={epoch:05d} supervised_mc={row['loss']:.6e}")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "domain": domain.__dict__,
        "hidden_dim": args.hidden_dim,
        "hidden_layers": args.hidden_layers,
        "option_type": "asian_arithmetic_call_surrogate",
        "output_scale": "normalized_by_S_max",
        "training": {
            "epochs": args.epochs,
            "lr": args.lr,
            "samples": args.samples,
            "mc_paths": args.mc_paths,
            "mc_steps": args.mc_steps,
        },
    }
    torch.save(checkpoint, out_dir / "asian_surrogate.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved model to {out_dir / 'asian_surrogate.pt'}")


if __name__ == "__main__":
    main()
