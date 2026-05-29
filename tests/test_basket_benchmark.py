from __future__ import annotations

from scripts.benchmark_basket_dimension import build_basket_inputs, run_benchmark


def test_build_basket_inputs_shapes_and_weights() -> None:
    spots, sigmas, weights = build_basket_inputs(5)
    assert spots.shape == (5,)
    assert sigmas.shape == (5,)
    assert weights.shape == (5,)
    assert weights.sum() == 1.0


def test_run_basket_dimension_benchmark_smoke() -> None:
    result = run_benchmark(
        [2, 3],
        n_paths=1000,
        repeats=1,
        strike=100.0,
        rate=0.05,
        maturity=1.0,
        correlation=0.25,
        seed=123,
        chunk_size=500,
    )
    rows = result["results"]
    assert result["instrument"] == "basket_call"
    assert len(rows) == 2
    assert rows[0]["n_assets"] == 2
    assert rows[1]["n_assets"] == 3
    assert all(row["price_mean"] > 0.0 for row in rows)
    assert all(row["seconds_mean"] >= 0.0 for row in rows)
    assert all(row["paths_per_second"] > 0.0 for row in rows)
