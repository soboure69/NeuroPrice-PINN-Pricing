from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neuroprice.pinn.heston_surrogate import HestonCallSurrogate, HestonSurrogateDomain


def load_dataset(dataset_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    data = np.load(dataset_path)
    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.float32)
    arrays = {key: data[key] for key in data.files}
    return x, y, arrays


def weighted_loss(pred: torch.Tensor, target: torch.Tensor, relative_floor: float) -> torch.Tensor:
    abs_loss = torch.mean((pred - target) ** 2)
    rel_denom = torch.clamp(target.abs(), min=relative_floor)
    rel_loss = torch.mean(((pred - target) / rel_denom) ** 2)
    itm_weight = 1.0 + 4.0 * (target > relative_floor).float()
    itm_loss = torch.mean(itm_weight * (pred - target) ** 2)
    return abs_loss + 0.05 * rel_loss + itm_loss


def evaluate(model: HestonCallSurrogate, loader: DataLoader, device: torch.device, relative_floor: float) -> dict[str, float]:
    model.eval()
    preds: list[np.ndarray] = []
    refs: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb.to(device)).cpu().numpy()
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
        "pct_under_5pct": float(np.mean(rel_err < 0.05) * 100.0),
        "pct_under_10pct": float(np.mean(rel_err < 0.10) * 100.0),
    }


def train_surrogate(
    *,
    dataset_path: Path,
    metadata_path: Path | None,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_dim: int,
    hidden_layers: int,
    validation_split: float,
    seed: int,
) -> list[dict[str, float]]:
    torch.manual_seed(seed)
    domain = HestonSurrogateDomain()
    x, y, _ = load_dataset(dataset_path)
    if x.shape[1] != domain.input_dim:
        raise ValueError(f"Dataset feature dimension {x.shape[1]} does not match expected {domain.input_dim}")

    source_metadata = {}
    if metadata_path is not None and metadata_path.exists():
        source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if source_metadata.get("dataset_version") != "heston_mc_dataset_v1":
            raise ValueError("Heston surrogate training requires dataset_version=heston_mc_dataset_v1")

    n_samples = x.shape[0]
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    n_val = max(1, int(n_samples * validation_split))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    train_ds = TensorDataset(torch.tensor(x[train_idx]), torch.tensor(y[train_idx]))
    val_ds = TensorDataset(torch.tensor(x[val_idx]), torch.tensor(y[val_idx]))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HestonCallSurrogate(input_dim=domain.input_dim, hidden_dim=hidden_dim, hidden_layers=hidden_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    relative_floor = 1.0 / domain.spot_max
    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = weighted_loss(pred, yb, relative_floor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            val_metrics = evaluate(model, val_loader, device, relative_floor)
            row = {"epoch": float(epoch), "train_loss": float(np.mean(losses)), **val_metrics}
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

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "domain": domain.__dict__,
        "hidden_dim": hidden_dim,
        "hidden_layers": hidden_layers,
        "input_dim": domain.input_dim,
        "option_type": "heston_call_surrogate",
        "output_scale": "normalized_by_spot_max",
        "dataset_metadata": source_metadata,
        "training": {
            "mode": "offline_monte_carlo_dataset",
            "dataset_path": str(dataset_path),
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "validation_split": validation_split,
            "seed": seed,
        },
    }
    torch.save(checkpoint, out_dir / "heston_surrogate.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved model to {out_dir / 'heston_surrogate.pt'}")
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="artifacts/phase6_heston_surrogate_dataset/dataset.npz")
    parser.add_argument("--metadata", type=str, default="artifacts/phase6_heston_surrogate_dataset/metadata.json")
    parser.add_argument("--out-dir", type=str, default="artifacts/phase6_heston_surrogate")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--hidden-layers", type=int, default=5)
    parser.add_argument("--validation-split", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    train_surrogate(
        dataset_path=Path(args.dataset),
        metadata_path=Path(args.metadata),
        out_dir=Path(args.out_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        validation_split=args.validation_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
