# Phase 6 — Extension scientifique

## Objectif

Étendre NeuroPrice vers des instruments et modèles plus avancés afin de démontrer la montée en dimension et la capacité des méthodes PINN/surrogates à compléter les méthodes numériques classiques.

## Axe 1 — Options sur panier multi-actifs

### Instrument initial

`basket_call`

Payoff à maturité :

```text
max(sum_i w_i S_i(T) - K, 0)
```

### Référence Monte Carlo

La première brique Phase 6 ajoute une référence Monte Carlo vectorisée NumPy pour un panier Black-Scholes multi-actifs avec corrélation constante.

Paramètres API :

```text
spots: niveaux spot des actifs
sigmas: volatilités individuelles
weights: poids du panier
correlation: corrélation constante hors diagonale
n_paths: nombre de trajectoires Monte Carlo
seed: graine déterministe
```

Contraintes actuelles :

```text
2 <= nombre d'actifs <= 10
-0.99 <= correlation <= 0.99
1000 <= n_paths <= 200000
```

### Endpoint

L'instrument est exposé via l'endpoint existant :

```text
POST /api/v1/price
```

Exemple de payload :

```json
{
  "instrument": "basket_call",
  "S0": 100.0,
  "K": 100.0,
  "sigma": 0.2,
  "r": 0.05,
  "T": 1.0,
  "spots": [100.0, 105.0, 95.0],
  "sigmas": [0.2, 0.25, 0.18],
  "weights": [0.4, 0.3, 0.3],
  "correlation": 0.25,
  "n_paths": 20000,
  "seed": 42
}
```

### Benchmark dimensionnel

Le benchmark Phase 6 mesure le coût Monte Carlo lorsque le nombre d'actifs du panier augmente.

Commande recommandée :

```bash
python scripts/benchmark_basket_dimension.py --dimensions 2,3,5,10 --n-paths 50000 --repeats 3
```

Sortie par défaut :

```text
artifacts/phase6_basket_dimension/benchmark.json
```

Métriques enregistrées par dimension :

```text
price_mean
price_std
seconds_mean
seconds_std
paths_per_second
time_ratio_vs_first_dimension
throughput_ratio_vs_first_dimension
```

### Dataset Monte Carlo pour surrogate/PINN

Le dataset offline sert de base pour entraîner un surrogate panier multi-actifs.

Commande recommandée :

```bash
python scripts/generate_basket_dataset.py --n-samples 5000 --n-assets 5 --n-paths 20000
```

Sorties par défaut :

```text
artifacts/phase6_basket_surrogate_dataset/dataset.npz
artifacts/phase6_basket_surrogate_dataset/metadata.json
```

Le fichier `dataset.npz` contient :

```text
x: features normalisées pour le surrogate
y: prix normalisés par spot_max
spots
sigmas
weights
strikes
rates
maturities
correlations
target_prices
```

Ordre des features dans `x` :

```text
spots_norm, sigmas_norm, weights, strike_norm, rate_norm, maturity_norm, correlation_norm,
basket_spot_norm, moneyness_norm, effective_sigma_norm, intrinsic_norm
```

La version `basket_mc_dataset_v2` ajoute quatre features synthétiques pour améliorer la précision du surrogate :

```text
basket_spot_norm: panier spot initial normalisé
moneyness_norm: ratio panier spot / strike normalisé
effective_sigma_norm: volatilité panier approximative normalisée
intrinsic_norm: payoff intrinsèque initial normalisé
```

### Prototype surrogate/PINN panier

Le prototype haute dimension utilise un MLP PyTorch sur les features normalisées du dataset Monte Carlo.

Commande smoke test :

```bash
python scripts/train_basket_surrogate.py --dataset artifacts/phase6_basket_surrogate_dataset_smoke/dataset.npz --metadata artifacts/phase6_basket_surrogate_dataset_smoke/metadata.json --out-dir artifacts/phase6_basket_surrogate_smoke --epochs 5 --hidden-dim 32 --hidden-layers 2 --batch-size 32
```

Commande recommandée :

```bash
python scripts/train_basket_surrogate.py --dataset artifacts/phase6_basket_surrogate_dataset/dataset.npz --metadata artifacts/phase6_basket_surrogate_dataset/metadata.json --out-dir artifacts/phase6_basket_surrogate --epochs 800 --hidden-dim 256 --hidden-layers 5 --batch-size 512
```

Sorties :

```text
artifacts/phase6_basket_surrogate/basket_surrogate.pt
artifacts/phase6_basket_surrogate/history.json
```

### Benchmark surrogate vs Monte Carlo

Le benchmark compare le surrogate entraîné à la référence Monte Carlo sur un sous-échantillon du dataset.

Commande recommandée :

```bash
python scripts/benchmark_basket_surrogate.py --checkpoint artifacts/phase6_basket_surrogate/basket_surrogate.pt --dataset artifacts/phase6_basket_surrogate_dataset/dataset.npz --n-points 500 --mc-paths 20000
```

Sortie par défaut :

```text
artifacts/phase6_basket_surrogate/benchmark.json
```

Métriques principales :

```text
mae
rmse
median_relative_error
p95_relative_error
pct_under_5pct
pct_under_10pct
surrogate_seconds
monte_carlo_seconds
speedup_vs_monte_carlo
```

## Prochaines étapes

- Comparer les prix et temps de calcul pour `N = 2, 3, 5, 10` actifs.
- Ajouter un benchmark montrant la difficulté des schémas FDM au-delà de faibles dimensions.
- Ajouter une visualisation frontend pour les paramètres multi-actifs.
- Démarrer ensuite le modèle de Heston.
