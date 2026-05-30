from __future__ import annotations

import json

import numpy as np

from scripts.generate_basket_dataset import generate_dataset, save_dataset


def test_generate_basket_dataset_shapes_and_metadata() -> None:
    x, y, params, metadata = generate_dataset(
        n_samples=4,
        n_assets=3,
        n_paths=1000,
        chunk_size=500,
        seed=123,
    )
    assert x.shape == (4, 13)
    assert y.shape == (4, 1)
    assert params["spots"].shape == (4, 3)
    assert params["sigmas"].shape == (4, 3)
    assert params["weights"].shape == (4, 3)
    assert params["target_prices"].shape == (4, 1)
    assert np.allclose(params["weights"].sum(axis=1), 1.0)
    assert np.all(x >= 0.0)
    assert np.all(x <= 1.0)
    assert np.all(y >= 0.0)
    assert metadata["instrument"] == "basket_call"
    assert metadata["n_samples"] == 4
    assert metadata["n_assets"] == 3


def test_save_basket_dataset_files(tmp_path) -> None:
    x, y, params, metadata = generate_dataset(
        n_samples=3,
        n_assets=2,
        n_paths=1000,
        chunk_size=500,
        seed=321,
    )
    save_dataset(tmp_path, x, y, params, metadata)
    dataset_path = tmp_path / "dataset.npz"
    metadata_path = tmp_path / "metadata.json"
    assert dataset_path.exists()
    assert metadata_path.exists()
    loaded = np.load(dataset_path)
    loaded_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert loaded["x"].shape == x.shape
    assert loaded["y"].shape == y.shape
    assert loaded_metadata["dataset_version"] == "basket_mc_dataset_v1"
