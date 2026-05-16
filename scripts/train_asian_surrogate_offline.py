from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from neuroprice.pinn.asian_surrogate import AsianArithmeticSurrogate, AsianSurrogateDomain
from neuroprice.validation.asian_ref import asian_arithmetic_call_mc_np


def build_dataset(
    domain: AsianSurrogateDomain,
    n_samples: int,
    n_paths: int,
    n_steps: int,
    chunk_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    S = rng.uniform(1e-6, domain.S_max, size=(n_samples, 1))
    tau = rng.uniform(1e-6, domain.T, size=(n_samples, 1))
    target = asian_arithmetic_call_mc_np(S, S, tau, domain.K, domain.r, domain.sigma, n_paths, n_steps, seed, chunk_size)
    x = np.concatenate([S / domain.S_max, tau / domain.T], axis=1).astype(np.float32)
    y = (target / domain.S_max).astype(np.float32)
    return x, y, target.astype(np.float32)


def weighted_loss(pred: torch.Tensor, target: torch.Tensor, relative_floor: float) -> torch.Tensor:
    abs_loss = torch.mean((pred - target) ** 2)
    rel_denom = torch.clamp(target.abs(), min=relative_floor)
    rel_loss = torch.mean(((pred - target) / rel_denom) ** 2)
    itm_weight = 1.0 + 4.0 * (target > relative_floor).float()
    itm_loss = torch.mean(itm_weight * (pred - target) ** 2)
    return abs_loss + 0.05 * rel_loss + itm_loss


def evaluate(model: AsianArithmeticSurrogate, loader: DataLoader, device: torch.device, relative_floor: float) -> dict[str, float]:
    model.eval()
    preds: list[np.ndarray] = []
    refs: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb[:, :1], xb[:, 1:2]).cpu().numpy()
            preds.append(pred)
            refs.append(yb.numpy())
    pred_np = np.concatenate(preds, axis=0)
    ref_np = np.concatenate(refs, axis=0)
    abs_err = np.abs(pred_np - ref_np)
    rel_err = abs_err / np.maximum(np.abs(ref_np), relative_floor)
    return {
        "mae_norm": float(np.mean(abs_err)),
        "rmse_norm": float(np.sqrt(np.mean(abs_err**2))),
        "median_relative_error": float(np.median(rel_err)),
        "p95_relative_error": float(np.quantile(rel_err, 0.95)),
        "pct_under_10pct": float(np.mean(rel_err < 0.10) * 100.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--dataset-samples", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--mc-paths", type=int, default=20000)
    parser.add_argument("--mc-steps", type=int, default=64)
    parser.add_argument("--mc-chunk-size", type=int, default=2000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=5)
    parser.add_argument("--validation-split", type=float, default=0.20)
    parser.add_argument("--K", type=float, default=100.0)
    parser.add_argument("--r", type=float, default=0.05)
    parser.add_argument("--sigma", type=float, default=0.20)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--S-max", type=float, default=300.0)
    parser.add_argument("--out-dir", type=str, default="artifacts/phase3_asian_surrogate_offline")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    domain = AsianSurrogateDomain(K=args.K, r=args.r, sigma=args.sigma, T=args.T, S_max=args.S_max)

    print("Generating fixed Monte Carlo dataset...")
    x, y, target_prices = build_dataset(domain, args.dataset_samples, args.mc_paths, args.mc_steps, args.mc_chunk_size, args.seed)
    np.savez_compressed(out_dir / "dataset.npz", x=x, y=y, target_prices=target_prices)

    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(args.dataset_samples)
    n_val = int(args.dataset_samples * args.validation_split)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    train_ds = TensorDataset(torch.tensor(x[train_idx]), torch.tensor(y[train_idx]))
    val_ds = TensorDataset(torch.tensor(x[val_idx]), torch.tensor(y[val_idx]))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = AsianArithmeticSurrogate(hidden_dim=args.hidden_dim, hidden_layers=args.hidden_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    relative_floor_norm = 1.0 / domain.S_max
    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb[:, :1], xb[:, 1:2])
            loss = weighted_loss(pred, yb, relative_floor_norm)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        scheduler.step()

        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            val_metrics = evaluate(model, val_loader, device, relative_floor_norm)
            row = {"epoch": float(epoch), "train_loss": float(np.mean(train_losses)), **val_metrics}
            history.append(row)
            print(
                f"epoch={epoch:05d} train={row['train_loss']:.6e} "
                f"val_rmse={row['rmse_norm']:.6e} val_p95_rel={row['p95_relative_error']:.6f} "
                f"val_under_10pct={row['pct_under_10pct']:.2f}%"
            )
            if row["rmse_norm"] < best_val:
                best_val = row["rmse_norm"]
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "domain": domain.__dict__,
        "hidden_dim": args.hidden_dim,
        "hidden_layers": args.hidden_layers,
        "option_type": "asian_arithmetic_call_surrogate",
        "output_scale": "normalized_by_S_max",
        "training": {
            "mode": "offline_monte_carlo_dataset",
            "epochs": args.epochs,
            "lr": args.lr,
            "dataset_samples": args.dataset_samples,
            "batch_size": args.batch_size,
            "mc_paths": args.mc_paths,
            "mc_steps": args.mc_steps,
            "validation_split": args.validation_split,
        },
    }
    torch.save(checkpoint, out_dir / "asian_surrogate.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved model to {out_dir / 'asian_surrogate.pt'}")


if __name__ == "__main__":
    main()
