# Benchmark Phase 1 — PINN Black-Scholes 1D

## Objectif

La Phase 1 vise à valider un PINN 1D pour le pricing d'une option européenne call sous le modèle de Black-Scholes.

L'objectif scientifique du projet pour OS1 est :

```text
Erreur relative L2 < 0.5%
```

Une cible expérimentale plus stricte a aussi été suivie pendant les essais :

```text
Erreur relative L2 < 0.1%
```

## Configuration de validation

Le script de validation compare le PINN entraîné à la formule analytique Black-Scholes sur une grille régulière en `(S, tau)`.

Commande type :

```bash
python -m scripts.validate_pinn_bs --checkpoint artifacts/<run>/bs_pinn.pt --out artifacts/<run>/validation_error.png
```

Métriques reportées :

- `Relative L2 error` : erreur globale prix.
- `Relative L2 error (tau > 0.02)` : erreur hors maturité immédiate.
- `Relative L2 error (outside strike band 0.05)` : erreur hors zone proche du strike.
- `Relative L2 error (regular zone)` : erreur hors maturité immédiate et hors zone proche du strike.
- `Delta relative L2 error` : erreur du Delta calculé par autograd.
- `Gamma relative L2 error` : erreur du Gamma calculé par autograd.

## Résultats observés

| Modèle | Prix global | Zone régulière | Delta | Gamma | Statut |
| --- | ---: | ---: | ---: | ---: | --- |
| Log-moneyness `x = log(S / K)` | 0.0877% | 0.0837% | 4.25% | 33.36% | Meilleur score observé, cible stricte validée |
| Direct — meilleur historique | 0.202% | 0.191% | 1.55% | 18.31% | Ancien meilleur score |
| Direct relancé | 0.353% | 0.351% | 4.69% | 36.11% | Moins bon |
| Supervised weight `1.0` | 0.380% | 0.378% | 4.42% | 34.35% | Rejeté |
| Supervised weight `0.1` | 0.561% | 0.561% | 6.14% | 38.98% | Rejeté |
| Ansatz contrainte | 1.085% | 0.680% | 20.23% | 100.29% | Rejeté |
| Pré-entraînement + fine-tuning PINN | 0.258% | 0.255% | 4.05% | 34.27% | Validé OS1 mais pas meilleur |

## Meilleur modèle récent reproductible

Le meilleur modèle validé utilise la transformation de variable `x = log(S / K)` et donne :

```text
Relative L2 error: 0.000877
Relative L2 error (tau > 0.02): 0.000864
Relative L2 error (outside strike band 0.05): 0.000843
Relative L2 error (regular zone): 0.000837
Delta relative L2 error: 0.042456
Gamma relative L2 error: 0.333569
```

Soit :

```text
Prix global ≈ 0.0877%
```

Ce résultat satisfait l'objectif OS1 du projet et la cible expérimentale stricte :

```text
0.0877% < 0.5%
0.0877% < 0.1%
```

## Commande de reproduction du modèle log-space

```bash
python -m scripts.train_pinn_bs_log --pretrain-epochs 3000 --pretrain-lr 0.001 --pretrain-samples 8192 --epochs 10000 --lbfgs-steps 700 --hidden-dim 96 --hidden-layers 5 --n-interior 4096 --n-terminal 2048 --n-boundary 2048 --n-supervised 4096 --terminal-weight 10 --lower-boundary-weight 1 --upper-boundary-weight 1 --pde-weight 1 --supervised-weight 0.01 --out-dir artifacts/phase1_bs_pinn_log
```

Validation :

```bash
python -m scripts.validate_pinn_bs_log --checkpoint artifacts/phase1_bs_pinn_log/bs_pinn_log.pt --out artifacts/phase1_bs_pinn_log/validation_error.png
```

## Interprétation

La Phase 1 est validée au niveau du cahier des charges scientifique initial : le PINN Black-Scholes 1D atteint une erreur relative inférieure à `0.5%` contre la solution analytique.

La cible expérimentale stricte `<0.1%` est maintenant atteinte grâce à la transformation de variable `x = log(S / K)`. Les essais suivants n'avaient pas amélioré le meilleur modèle direct historique :

- augmentation simple de la taille du réseau ;
- ansatz de sortie contrainte ;
- loss supervisée analytique mélangée dès le début ;
- pré-entraînement supervisé puis fine-tuning PINN.

Les erreurs sur les Greeks confirment que `Gamma` reste la métrique la plus difficile, car elle dépend d'une dérivée seconde du réseau. Le résultat log-space améliore fortement le prix, mais ne résout pas encore complètement la précision des Greeks.

## Conclusion Phase 1

Statut :

```text
Phase 1 validée pour OS1 et pour la cible stricte <0.1%
```

Critères validés :

```text
Erreur relative L2 < 0.5%
Erreur relative L2 < 0.1%
```

Résultat retenu :

```text
Erreur relative L2 ≈ 0.0877%
```

Le pipeline complet est opérationnel :

- PINN Black-Scholes 1D ;
- collocation PDE ;
- conditions terminales et limites ;
- Adam + L-BFGS ;
- validation analytique ;
- métriques détaillées ;
- validation Delta/Gamma par autograd ;
- pré-entraînement supervisé optionnel ;
- formulation log-moneyness `x = log(S / K)`.

## Recommandation

Pour respecter le roadmap, il est recommandé de passer à la Phase 2 :

- options exotiques simples ;
- validation Monte Carlo ;
- premiers cas sans formule analytique fermée.

Si l'amélioration des Greeks devient prioritaire, les prochaines pistes techniques sont :

- Fourier features ou SIREN ;
- sampling adaptatif basé sur le résidu PDE ;
- early stopping sur une grille de validation analytique ;
- optimisation dédiée des Greeks, surtout Gamma.
