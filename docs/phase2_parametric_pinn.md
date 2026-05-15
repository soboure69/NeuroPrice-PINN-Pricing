# Phase 2 — PINN Black-Scholes paramétrique

## Objectif

La Phase 2 vise à entraîner un seul réseau capable de pricer une option européenne call Black-Scholes sur un espace de paramètres variable.

Entrées du modèle :

```text
x = log(S / K)
tau / T
sigma
r
T
K
```

Sortie du modèle :

```text
u = V / K
```

Le prix physique est reconstruit par :

```text
V = K * u
```

## Domaine paramétrique

Le domaine initial suit le roadmap Phase 2 :

```text
S     ∈ [20, 200]
K     ∈ [20, 200]
sigma ∈ [0.05, 0.80]
r     ∈ [0.00, 0.15]
T     ∈ [0.1, 5.0]
```

La variable de réseau est `x = log(S / K)`, bornée par défaut sur :

```text
x ∈ [-2.5, 2.5]
```

## PDE utilisée

En formulation log-moneyness, avec `u = V / K`, la PDE devient :

```text
u_tau - 0.5 sigma^2 u_xx - (r - 0.5 sigma^2) u_x + r u = 0
```

Cette formulation est reprise de la Phase 1, où elle a permis d'atteindre :

```text
Relative L2 error = 0.000877 ≈ 0.0877%
```

## Fichiers Phase 2

- `neuroprice/pinn/parametric_bs.py`
  - modèle `ParametricBlackScholesPINN`
  - domaine `ParametricBlackScholesDomain`
  - loss PDE paramétrique
  - sampling multi-paramètres

- `scripts/train_pinn_bs_parametric.py`
  - entraînement Adam
  - L-BFGS optionnel
  - sauvegarde checkpoint

- `scripts/validate_pinn_bs_parametric.py`
  - validation sur points aléatoires
  - erreur relative L2
  - distribution des erreurs relatives point par point
  - benchmark vitesse PINN vs formule analytique

## Commande de test

```bash
pytest -q
```

## Première expérience rapide

Cette commande vérifie que le pipeline apprend et produit un checkpoint, sans viser encore le score final Phase 2 :

```bash
python -m scripts.train_pinn_bs_parametric --epochs 2000 --hidden-dim 128 --hidden-layers 4 --n-interior 4096 --n-terminal 2048 --n-boundary 2048 --n-supervised 4096 --pde-weight 1 --terminal-weight 10 --lower-boundary-weight 1 --upper-boundary-weight 1 --supervised-weight 0.1 --out-dir artifacts/phase2_bs_pinn_parametric_smoke
```

Validation rapide :

```bash
python -m scripts.validate_pinn_bs_parametric --checkpoint artifacts/phase2_bs_pinn_parametric_smoke/bs_pinn_parametric.pt --out artifacts/phase2_bs_pinn_parametric_smoke/benchmark.json --n-points 10000
```

## Entraînement Phase 2 recommandé

```bash
python -m scripts.train_pinn_bs_parametric --epochs 10000 --hidden-dim 128 --hidden-layers 4 --n-interior 8192 --n-terminal 4096 --n-boundary 4096 --n-supervised 8192 --lbfgs-steps 500 --pde-weight 1 --terminal-weight 10 --lower-boundary-weight 1 --upper-boundary-weight 1 --supervised-weight 0.1 --out-dir artifacts/phase2_bs_pinn_parametric
```

Validation benchmark :

```bash
python -m scripts.validate_pinn_bs_parametric --checkpoint artifacts/phase2_bs_pinn_parametric/bs_pinn_parametric.pt --out artifacts/phase2_bs_pinn_parametric/benchmark.json --n-points 10000
```

## Critère Phase 2

Le critère du roadmap est :

```text
Erreur < 0.5% pour 95% des points
```

Le script de validation reporte :

```text
pct_points_under_0_5pct_error
p95_point_relative_error
relative_l2
pinn_points_per_second
analytic_points_per_second
```

Les erreurs relatives point par point utilisent un plancher de prix afin d'éviter les explosions numériques lorsque le prix analytique est quasi nul :

```text
point_relative_error = abs(pred - ref) / max(abs(ref), relative_floor)
```

Par défaut :

```text
relative_floor = 1.0
min_reference_price = 1.0
```

Le script reporte aussi des métriques `tradable_*` calculées uniquement sur les points dont le prix analytique est supérieur à `min_reference_price`.

## Prochaines améliorations si nécessaire

Si la première version ne satisfait pas le critère `95% < 0.5%`, les pistes prioritaires sont :

- augmenter `hidden_dim` à `192` ou `256` ;
- augmenter `n_supervised` et `supervised_weight` en pré-entraînement implicite ;
- utiliser le pré-entraînement supervisé multi-paramètres via `--pretrain-epochs` ;
- réduire temporairement le domaine paramétrique puis l'élargir progressivement ;
- utiliser Fourier features sur `x`, `tau` et `sigma`.

## Commande améliorée avec pré-entraînement supervisé

Le premier benchmark Phase 2 sans pré-entraînement a montré que le PINN paramétrique ne convergeait pas encore suffisamment sur le domaine large. La stratégie recommandée est de commencer par approximer la formule analytique sur l'espace paramétrique, puis de fine-tuner avec la PDE et les conditions terminales/limites.

En coordonnées `x = log(S/K)` avec une sortie normalisée `V/K`, la solution Black-Scholes paramétrique dépend de `x`, `tau`, `sigma`, `r` et `T`, mais pas directement de `K`. Les nouveaux entraînements ignorent donc `K_norm` par défaut afin de retirer une dimension parasite. L'option `--use-strike-input` reste disponible uniquement pour compatibilité ou ablation.

Après benchmark segmenté, les erreurs les plus fortes apparaissent surtout pour les strikes élevés et autour/au-dessous du strike. Cela indique que l'ancien réseau 6D apprenait une dépendance artificielle à `K`, alors que cette dépendance est déjà absorbée par `x` et la normalisation par `K`.

Les benchmarks suivants ont montré que l'erreur reste particulièrement élevée pour les maturités très courtes, les faibles volatilités et la zone autour du strike. La transformation de sortie `--output-transform terminal` impose donc exactement la condition terminale :

```text
u(x, tau) = max(exp(x) - 1, 0) + tau_norm * raw_network(x, tau_norm, sigma, r, T)
```

Cette contrainte réduit la difficulté près de `tau = 0`, où le payoff est non lisse autour du strike.

Le benchmark de cette transformation a toutefois dégradé la précision globale. La contrainte terminale seule rend le résidu appris plus difficile à optimiser sur le domaine complet. La prochaine expérience recommandée revient donc à `--output-transform direct` et ajoute des Fourier features pour mieux représenter les régimes abrupts autour du strike et des maturités courtes.

Le meilleur résultat PINN observé avec Fourier features améliore nettement la distribution d'erreur, mais reste loin du critère strict `95% < 0.5%`. Pour isoler la limite d'architecture de la contrainte PDE, le script supporte maintenant `--skip-pinn-finetune`, qui conserve uniquement l'entraînement supervisé analytique. Ce mode correspond à un surrogate paramétrique Black-Scholes plutôt qu'à un PINN complet, mais il permet de tester si l'architecture peut atteindre la précision roadmap avant de réintroduire progressivement la PDE.

Expérience intermédiaire :

```bash
python -m scripts.train_pinn_bs_parametric --pretrain-epochs 5000 --pretrain-lr 0.001 --pretrain-samples 16384 --epochs 8000 --hidden-dim 192 --hidden-layers 5 --n-interior 8192 --n-terminal 4096 --n-boundary 4096 --n-supervised 8192 --lbfgs-steps 500 --pde-weight 1 --terminal-weight 10 --lower-boundary-weight 1 --upper-boundary-weight 1 --supervised-weight 0.05 --out-dir artifacts/phase2_bs_pinn_parametric_pretrained
```

Validation :

```bash
python -m scripts.validate_pinn_bs_parametric --checkpoint artifacts/phase2_bs_pinn_parametric_pretrained/bs_pinn_parametric.pt --out artifacts/phase2_bs_pinn_parametric_pretrained/benchmark.json --n-points 10000 --relative-floor 1.0 --min-reference-price 1.0
```

Si le critère reste trop loin, lancer une version plus large :

```bash
python -m scripts.train_pinn_bs_parametric --pretrain-epochs 10000 --pretrain-lr 0.001 --pretrain-samples 32768 --epochs 12000 --hidden-dim 256 --hidden-layers 5 --n-interior 12000 --n-terminal 6000 --n-boundary 6000 --n-supervised 12000 --lbfgs-steps 800 --pde-weight 1 --terminal-weight 10 --lower-boundary-weight 1 --upper-boundary-weight 1 --supervised-weight 0.05 --out-dir artifacts/phase2_bs_pinn_parametric_large
```

Version recommandée après diagnostic segmenté :

```bash
python -m scripts.train_pinn_bs_parametric --output-transform direct --fourier-features 4 --pretrain-epochs 10000 --pretrain-lr 0.001 --pretrain-samples 32768 --pretrain-relative-weight 1.0 --relative-floor 0.01 --epochs 12000 --hidden-dim 256 --hidden-layers 5 --n-interior 12000 --n-terminal 6000 --n-boundary 6000 --n-supervised 12000 --supervised-weight 0.05 --supervised-relative-weight 1.0 --lbfgs-steps 500 --pde-weight 1 --terminal-weight 10 --lower-boundary-weight 1 --upper-boundary-weight 1 --out-dir artifacts/phase2_bs_pinn_parametric_fourier
```

Validation :

```bash
python -m scripts.validate_pinn_bs_parametric --checkpoint artifacts/phase2_bs_pinn_parametric_fourier/bs_pinn_parametric.pt --out artifacts/phase2_bs_pinn_parametric_fourier/benchmark.json --n-points 10000 --relative-floor 1.0 --min-reference-price 1.0
```

Test surrogate supervisé ciblé précision :

```bash
python -m scripts.train_pinn_bs_parametric --output-transform direct --fourier-features 4 --skip-pinn-finetune --pretrain-epochs 15000 --pretrain-lr 0.001 --pretrain-samples 65536 --pretrain-relative-weight 2.0 --relative-floor 0.01 --hidden-dim 256 --hidden-layers 6 --out-dir artifacts/phase2_bs_surrogate_fourier
```

Validation :

```bash
python -m scripts.validate_pinn_bs_parametric --checkpoint artifacts/phase2_bs_surrogate_fourier/bs_pinn_parametric.pt --out artifacts/phase2_bs_surrogate_fourier/benchmark.json --n-points 10000 --relative-floor 1.0 --min-reference-price 1.0
```

## Benchmark vitesse complet avec Monte Carlo

Le script de validation compare maintenant trois méthodes :

- PINN paramétrique ;
- Black-Scholes analytique ;
- Monte Carlo européen vectorisé.

Commande recommandée sur le meilleur artefact Phase 2 actuel :

```bash
python -m scripts.validate_pinn_bs_parametric --checkpoint artifacts/phase2_bs_pinn_parametric_fourier/bs_pinn_parametric.pt --out artifacts/phase2_bs_pinn_parametric_fourier/benchmark_with_mc.json --n-points 10000 --relative-floor 1.0 --min-reference-price 1.0 --mc-paths 10000 --mc-chunk-size 1000
```

Métriques ajoutées au JSON :

```text
monte_carlo_seconds
monte_carlo_points_per_second
pinn_vs_monte_carlo_speedup
monte_carlo_relative_l2
monte_carlo_mean_point_relative_error
monte_carlo_p95_point_relative_error
```

Critère de satisfaction vitesse Phase 2 :

```text
PINN >= 50k prix/seconde
PINN >= 100x plus rapide que Monte Carlo 10k paths
PINN pas plus de 20x plus lent que Black-Scholes analytique
```
