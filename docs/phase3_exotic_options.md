# Phase 3 — Options exotiques

## Objectif

Étendre NeuroPrice aux options sans solution analytique simple en commençant par une option barrière Down-and-Out, puis une option asiatique à moyenne arithmétique.

## Étape 1 — Call barrière Down-and-Out

L'option Down-and-Out call suit la PDE Black-Scholes standard sur le domaine `S > B`, avec une condition absorbante à la barrière :

```text
V(B, tau) = 0
```

La condition terminale est :

```text
V(S, 0) = max(S - K, 0) pour S > B
V(S, 0) = 0 pour S <= B
```

Le premier cas implémenté utilise :

```text
K = 100
B = 70
r = 0.05
sigma = 0.20
T = 1.0
S_max = 300
```

## Référence de validation

La validation utilise une formule semi-analytique par méthode des images, valable pour le cas `B < K`.

Fichier :

```text
neuroprice/validation/barrier_ref.py
```

Fonction :

```text
down_and_out_call_price_np
```

## Implémentation PINN

Fichier :

```text
neuroprice/pinn/barrier.py
```

Composants principaux :

```text
BarrierOptionDomain
DownAndOutBarrierPINN
sample_barrier_batch
barrier_pinn_loss
```

Le modèle supporte deux transformations de sortie :

```text
direct
barrier
```

Le mode `barrier` impose directement l'annulation à la barrière via :

```text
V_norm(S, tau) = max((S - B) / (S_max - B), 0) * raw_network(S_norm, tau_norm)
```

## Entraînement initial

Commande recommandée :

```bash
python -m scripts.train_pinn_barrier_down_out --pretrain-epochs 3000 --epochs 8000 --hidden-dim 128 --hidden-layers 5 --output-transform barrier --n-interior 8192 --n-terminal 4096 --n-boundary 4096 --n-supervised 8192 --supervised-weight 0.1 --lbfgs-steps 500 --out-dir artifacts/phase3_down_out_barrier
```

## Entraînement stabilisé près de la barrière

La zone `S ≈ B` est difficile car le prix tend vers zéro et les erreurs relatives deviennent très instables. La version stabilisée renforce l'échantillonnage près de la barrière et pondère la loss supervisée dans cette zone.

Commande recommandée :

```bash
python -m scripts.train_pinn_barrier_down_out --pretrain-epochs 3000 --epochs 8000 --hidden-dim 128 --hidden-layers 5 --output-transform barrier --n-interior 8192 --n-terminal 4096 --n-boundary 4096 --n-supervised 8192 --supervised-weight 0.1 --near-barrier-supervised-weight 2.0 --lbfgs-steps 500 --out-dir artifacts/phase3_down_out_barrier_stabilized
```

## Validation

```bash
python -m scripts.validate_pinn_barrier_down_out --checkpoint artifacts/phase3_down_out_barrier/down_out_barrier_pinn.pt --out artifacts/phase3_down_out_barrier/benchmark.json --n-points 10000 --relative-floor 1.0 --min-reference-price 1.0
```

## Métriques suivies

```text
relative_l2
tradable_p95_point_relative_error
tradable_pct_points_under_1pct_error
tradable_pct_points_under_5pct_error
near_barrier_mean_relative_error
near_barrier_mean_absolute_error
near_barrier_median_absolute_error
near_barrier_p95_absolute_error
near_barrier_tradable_p95_relative_error
near_strike_mean_relative_error
pinn_points_per_second
analytic_points_per_second
```

## Critères de succès initiaux

Comme la Phase 3 porte sur des produits plus difficiles que les vanilles européennes, le premier jalon vise :

```text
relative_l2 < 2%
tradable_p95_point_relative_error < 10%
tradable_pct_points_under_5pct_error > 90%
V(B, tau) ≈ 0 par construction ou par perte barrière
```

Ces seuils pourront être resserrés après stabilisation de l'entraînement.

## Décision barrière Down-and-Out

Le modèle baseline est retenu comme artefact validé :

```text
artifacts/phase3_down_out_barrier/down_out_barrier_pinn.pt
artifacts/phase3_down_out_barrier/benchmark.json
```

Résultats baseline :

```text
relative_l2 = 0.2957%
tradable_p95_point_relative_error = 9.8904%
tradable_pct_points_under_5pct_error = 93.43%
pinn_points_per_second = 474,467
```

La tentative `phase3_down_out_barrier_stabilized` est documentée comme ablation négative. Elle dégrade les métriques globales et tradables malgré le renforcement near-barrier :

```text
relative_l2 = 0.5480%
tradable_p95_point_relative_error = 14.0588%
tradable_pct_points_under_5pct_error = 91.89%
near_barrier_mean_relative_error = 65.03%
```

Conclusion : conserver le baseline pour la brique barrière et passer à l'option asiatique.

## Étape 2 — Call asiatique à moyenne arithmétique

L'option asiatique introduit une dimension d'état supplémentaire `A`, la moyenne courante. Le PINN utilise les variables normalisées :

```text
S_norm = S / S_max
A_norm = A / A_max
tau_norm = tau / T
```

La PDE utilisée en temps restant `tau` est :

```text
∂V/∂tau - 0.5 σ² S² ∂²V/∂S² - rS ∂V/∂S - (S-A)/tau ∂V/∂A + rV = 0
```

La condition terminale est :

```text
V(S, A, 0) = max(A - K, 0)
```

Fichiers ajoutés :

```text
neuroprice/pinn/asian.py
neuroprice/validation/asian_ref.py
scripts/train_pinn_asian_arithmetic.py
scripts/validate_pinn_asian_arithmetic.py
```

### Référence Monte Carlo

La validation utilise un Monte Carlo vectorisé sur la dynamique de Black-Scholes, avec moyenne arithmétique discrète.

Fonction :

```text
asian_arithmetic_call_mc_np
```

### Entraînement initial asiatique

```bash
python -m scripts.train_pinn_asian_arithmetic --epochs 8000 --hidden-dim 128 --hidden-layers 5 --n-interior 8192 --n-terminal 4096 --n-boundary 4096 --lbfgs-steps 500 --out-dir artifacts/phase3_asian_arithmetic
```

### Validation initiale asiatique

```bash
python -m scripts.validate_pinn_asian_arithmetic --checkpoint artifacts/phase3_asian_arithmetic/asian_arithmetic_pinn.pt --out artifacts/phase3_asian_arithmetic/benchmark.json --n-points 1000 --mc-paths 20000 --mc-steps 64 --mc-chunk-size 2000 --relative-floor 1.0 --min-reference-price 1.0
```

### Constat premier benchmark asiatique

Le premier entraînement PDE-only est très rapide face au Monte Carlo, mais insuffisant en précision :

```text
relative_l2 = 64.74%
tradable_p95_point_relative_error = 883.36%
tradable_pct_points_under_10pct_error = 8.06%
pinn_vs_monte_carlo_speedup = 810.77x
```

Conclusion : la vitesse est validée, mais la précision nécessite un pré-entraînement supervisé Monte Carlo.

### Entraînement avec prétraining Monte Carlo

Commande recommandée pour le second essai :

```bash
python -m scripts.train_pinn_asian_arithmetic --pretrain-epochs 500 --pretrain-samples 512 --pretrain-mc-paths 4096 --pretrain-mc-steps 32 --pretrain-mc-chunk-size 1024 --epochs 8000 --hidden-dim 128 --hidden-layers 5 --n-interior 8192 --n-terminal 4096 --n-boundary 4096 --lbfgs-steps 500 --out-dir artifacts/phase3_asian_arithmetic_mc_pretrained
```

Validation correspondante :

```bash
python -m scripts.validate_pinn_asian_arithmetic --checkpoint artifacts/phase3_asian_arithmetic_mc_pretrained/asian_arithmetic_pinn.pt --out artifacts/phase3_asian_arithmetic_mc_pretrained/benchmark.json --n-points 1000 --mc-paths 20000 --mc-steps 64 --mc-chunk-size 2000 --relative-floor 1.0 --min-reference-price 1.0
```

Le prétraining Monte Carlo naïf sur le PINN 3D n'a pas amélioré le benchmark :

```text
relative_l2 = 65.14%
tradable_p95_point_relative_error = 878.54%
tradable_pct_points_under_10pct_error = 8.36%
pinn_vs_monte_carlo_speedup = 1537.11x
```

Conclusion : le PINN asiatique 3D reste expérimental. Le blocage vient probablement de la formulation d'état/PDE et de la cohérence entre `A`, `tau` et la moyenne discrète Monte Carlo.

### Surrogate asiatique supervisé Monte Carlo

Pour obtenir un jalon asiatique exploitable sans bloquer la Phase 3, un surrogate supervisé Monte Carlo est ajouté sur le cas standard `A0 = S0`, avec variables :

```text
S0_norm = S0 / S_max
tau_norm = tau / T
```

Fichiers :

```text
neuroprice/pinn/asian_surrogate.py
scripts/train_asian_surrogate.py
scripts/validate_asian_surrogate.py
```

Entraînement :

```bash
python -m scripts.train_asian_surrogate --epochs 2000 --samples 1024 --mc-paths 4096 --mc-steps 64 --mc-chunk-size 1024 --hidden-dim 128 --hidden-layers 5 --out-dir artifacts/phase3_asian_surrogate
```

Validation :

```bash
python -m scripts.validate_asian_surrogate --checkpoint artifacts/phase3_asian_surrogate/asian_surrogate.pt --out artifacts/phase3_asian_surrogate/benchmark.json --n-points 1000 --mc-paths 20000 --mc-steps 64 --mc-chunk-size 2000 --relative-floor 1.0 --min-reference-price 1.0
```

Premier résultat surrogate online :

```text
relative_l2 = 55.41%
tradable_p95_point_relative_error = 66.27%
tradable_pct_points_under_10pct_error = 1.34%
pinn_vs_monte_carlo_speedup = 1556.77x
```

Le surrogate online réduit les erreurs extrêmes du PINN 3D, mais reste trop imprécis. La stratégie est donc remplacée par un entraînement sur dataset Monte Carlo offline fixe, avec split train/validation et mini-batches réutilisables.

Entraînement offline :

```bash
python -m scripts.train_asian_surrogate_offline --epochs 3000 --dataset-samples 20000 --batch-size 512 --mc-paths 20000 --mc-steps 64 --mc-chunk-size 2000 --hidden-dim 128 --hidden-layers 5 --out-dir artifacts/phase3_asian_surrogate_offline
```

Validation offline :

```bash
python -m scripts.validate_asian_surrogate --checkpoint artifacts/phase3_asian_surrogate_offline/asian_surrogate.pt --out artifacts/phase3_asian_surrogate_offline/benchmark.json --n-points 1000 --mc-paths 20000 --mc-steps 64 --mc-chunk-size 2000 --relative-floor 1.0 --min-reference-price 1.0
```

Résultat offline validé :

```text
relative_l2 = 0.16%
tradable_p95_point_relative_error = 1.39%
tradable_pct_points_under_5pct_error = 99.11%
tradable_pct_points_under_10pct_error = 99.85%
max_absolute_error = 1.26
pinn_vs_monte_carlo_speedup = 4680.98x
```

Conclusion : le surrogate asiatique offline supervisé Monte Carlo valide le jalon asiatique en précision et en vitesse. Le PINN 3D reste documenté comme piste expérimentale à reprendre avec une formulation d'état plus rigoureuse.

## Option Lookback Floating-Strike Call

La troisième exotique retenue pour compléter le livrable Phase 3 est une option lookback floating-strike call :

```text
payoff = max(S_T - min(S_t), 0)
```

Elle dépend du chemin via le minimum réalisé du sous-jacent. Pour valider rapidement le livrable, l'approche retenue est un surrogate supervisé Monte Carlo offline sur le cas initial standard `min_0 = S0`, avec variables :

```text
S0_norm = S0 / S_max
tau_norm = tau / T
```

Fichiers :

```text
neuroprice/validation/lookback_ref.py
neuroprice/pinn/lookback_surrogate.py
scripts/train_lookback_surrogate_offline.py
scripts/validate_lookback_surrogate.py
```

Entraînement :

```bash
python -m scripts.train_lookback_surrogate_offline --epochs 3000 --dataset-samples 20000 --batch-size 512 --mc-paths 20000 --mc-steps 64 --mc-chunk-size 2000 --hidden-dim 128 --hidden-layers 5 --out-dir artifacts/phase3_lookback_surrogate_offline
```

Validation :

```bash
python -m scripts.validate_lookback_surrogate --checkpoint artifacts/phase3_lookback_surrogate_offline/lookback_surrogate.pt --out artifacts/phase3_lookback_surrogate_offline/benchmark.json --n-points 1000 --mc-paths 20000 --mc-steps 64 --mc-chunk-size 2000 --relative-floor 1.0 --min-reference-price 1.0
```

Résultat validé :

```text
relative_l2 = 0.66%
tradable_p95_point_relative_error = 1.48%
tradable_pct_points_under_5pct_error = 99.48%
tradable_pct_points_under_10pct_error = 99.69%
max_absolute_error = 0.63
pinn_vs_monte_carlo_speedup = 5475.93x
```

Conclusion : le surrogate lookback floating-strike call valide la troisième exotique Phase 3 en précision et en vitesse.

## Clôture du livrable Phase 3

Les trois types d'options exotiques demandés sont maintenant pricés et validés :

```text
1. Down-and-Out barrier call : validée contre référence semi-analytique
2. Asian arithmetic call : validée contre Monte Carlo haute précision
3. Lookback floating-strike call : validée contre Monte Carlo haute précision
```

### Critères de succès initiaux asiatiques

Pour ce premier cas multi-dimensionnel, les critères initiaux sont volontairement plus larges :

```text
relative_l2 < 10%
tradable_p95_point_relative_error < 25%
tradable_pct_points_under_10pct_error > 70%
PINN au moins 50x plus rapide que Monte Carlo 20k paths
```

Ces critères servent à valider l'architecture 3D avant optimisation de précision.

## Tests

Les tests unitaires couvrent :

```text
référence Down-and-Out vectorisée
forme de sortie du PINN barrière
finitude de la loss barrière
référence Monte Carlo asiatique vectorisée
forme de sortie du PINN asiatique 3D
finitude de la loss asiatique
```

Commande :

```bash
python -m pytest tests/test_pinn_bs.py -q
```
