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
spots_norm, sigmas_norm, weights, strike_norm, rate_norm, maturity_norm, correlation_norm
```

## Prochaines étapes

- Comparer les prix et temps de calcul pour `N = 2, 3, 5, 10` actifs.
- Ajouter un benchmark montrant la difficulté des schémas FDM au-delà de faibles dimensions.
- Ajouter une visualisation frontend pour les paramètres multi-actifs.
- Démarrer ensuite le modèle de Heston.
