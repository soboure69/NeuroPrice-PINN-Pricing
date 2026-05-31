from __future__ import annotations

import json

import numpy as np
import torch

from scripts.train_basket_surrogate import load_dataset, train_surrogate


def test_train_basket_surrogate_smoke(tmp_path) -> None:
    n_samples = 12
    n_assets = 2
    input_dim = 3 * n_assets + 8
    rng = np.random.default_rng(123)
    x = rng.uniform(0.0, 1.0, size=(n_samples, input_dim)).astype(np.float32)
    y = rng.uniform(0.0, 0.5, size=(n_samples, 1)).astype(np.float32)
    dataset_path = tmp_path / "dataset.npz"
    metadata_path = tmp_path / "metadata.json"
    out_dir = tmp_path / "model"
    np.savez_compressed(
        dataset_path,
        x=x,
        y=y,
        spots=rng.uniform(50.0, 150.0, size=(n_samples, n_assets)).astype(np.float32),
        sigmas=rng.uniform(0.1, 0.6, size=(n_samples, n_assets)).astype(np.float32),
        weights=np.full((n_samples, n_assets), 1.0 / n_assets, dtype=np.float32),
        target_prices=(150.0 * y).astype(np.float32),
    )
    metadata_path.write_text(json.dumps({"dataset_version": "basket_mc_dataset_v2"}), encoding="utf-8")

    loaded_x, loaded_y, _ = load_dataset(dataset_path)
    assert loaded_x.shape == x.shape
    assert loaded_y.shape == y.shape

    history = train_surrogate(
        dataset_path=dataset_path,
        metadata_path=metadata_path,
        out_dir=out_dir,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        hidden_dim=8,
        hidden_layers=2,
        validation_split=0.25,
        seed=123,
    )
    assert history
    assert (out_dir / "basket_surrogate.pt").exists()
    assert (out_dir / "history.json").exists()
    checkpoint = torch.load(out_dir / "basket_surrogate.pt", map_location="cpu")
    assert checkpoint["option_type"] == "basket_call_surrogate"
    assert checkpoint["input_dim"] == input_dim
