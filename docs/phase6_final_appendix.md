# Appendice final Phase 6

## Intégration API des surrogates Phase 6

Les instruments `basket_call` et `heston_call` supportent maintenant `method=auto`, `method=model` et `method=reference`.

Comportement :

```text
method=auto: utilise le surrogate si le checkpoint est disponible, sinon fallback Monte Carlo
method=model: force le surrogate et renvoie une erreur si le checkpoint est absent ou incompatible
method=reference: force la référence Monte Carlo
```

Checkpoints attendus :

```text
artifacts/phase6_basket_surrogate/basket_surrogate.pt
artifacts/phase6_heston_surrogate/heston_surrogate.pt
```

Versions API :

```text
basket_surrogate_v2
heston_surrogate_v1
basket_monte_carlo_v1
heston_monte_carlo_v1
```

## Résultats scientifiques Phase 6

Basket surrogate v2 vs Monte Carlo :

```text
speedup_vs_monte_carlo: 12.61
mae: 0.827446
rmse: 1.152939
median_relative_error: 0.045533
p95_relative_error: 0.300232
pct_under_10pct: 72.80%
```

Heston surrogate vs Monte Carlo :

```text
speedup_vs_monte_carlo: 14257.40
mae: 0.967111
rmse: 1.411719
median_relative_error: 0.043530
p95_relative_error: 0.269083
pct_under_10pct: 70.00%
```
