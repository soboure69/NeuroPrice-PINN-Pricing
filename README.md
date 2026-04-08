# NeuroPrice-PINN-Pricing
## Contexte et genèse du projet:
La valorisation d'instruments financiers dérivés (options, produits structurés, contrats futures) est au cœur de la finance de marché mondiale. Chaque jour, des milliards d'euros de contrats sont pricés, échangés, couverts sur les marchés mondiaux.

Pricer un instrument financier, c'est répondre à une question fondamentale :

> *"Quelle est la juste valeur aujourd'hui d'un contrat dont le payoff dépend d'un événement futur incertain ?"*

Pour répondre à cette question, les quants (mathématiciens de la finance) ont développé depuis les années 1970 des modèles mathématiques sophistiqués, dont le plus célèbre est le modèle de **Black-Scholes-Merton** (Prix Nobel d'Économie 1997).
Ces modèles de pricing se formulent naturellement comme des **équations aux dérivées partielles (EDPs)**. Le modèle de Black-Scholes donne naissance à l'EDP :

$$
\frac{\partial V}{\partial t}
+\frac{1}{2}\sigma^{2} S^{2}\frac{\partial^{2} V}{\partial S^{2}}
+ r S \frac{\partial V}{\partial S}
- rV = 0.
$$

Pour les options européennes simples, cette EDP possède une solution analytique fermée (la formule de Black-Scholes). Mais dans la réalité des marchés, les praticiens ont besoin de pricer des instruments bien plus complexes :

- **Options exotiques** : barrières, lookback, asiatiques, digitales...
- **Modèles de volatilité avancés** : volatilité locale (Dupire), volatilité stochastique (Heston, SABR)
- **Instruments multi-sous-jacents** : options sur panier, produits de corrélation
- **Dérivés de taux d'intérêt** : swaps, caps, floors, swaptions

Pour ces instruments, **il n'existe pas de formule analytique**. Les banques et hedge funds utilisent alors deux approches classiques :

**Méthode 1 — Monte Carlo** : simuler des milliers/millions de trajectoires stochastiques et moyenner les payoffs. Précis mais extrêmement lent (minutes à heures pour les produits complexes).

**Méthode 2 — Différences Finies (FDM)** : discrétiser l'EDP sur une grille et résoudre numériquement. Plus rapide mais limité à des dimensions basses (malédiction de la dimensionnalité au-delà de 3-4 sous-jacents).

En 2019, une publication scientifique majeure de **Raissi, Perdikaris & Karniadakis** (MIT/Brown University) publiée dans le *Journal of Computational Physics* révolutionne l'approche : les **Physics-Informed Neural Networks (PINNs)**.

L'idée fondamentale est élégante :

> Au lieu de discrétiser une EDP sur une grille, on entraîne un réseau de neurones à satisfaire simultanément l'EDP et les conditions aux limites, en intégrant les lois physiques directement dans la fonction de perte.

Cette approche a d'abord été développée pour la mécanique des fluides, la physique des matériaux, et l'ingénierie. Son application à la **finance quantitative** est un domaine de recherche émergent, avec des premiers résultats publiés depuis 2020-2022.

Aujourd'hui :

- Les **banques** et **hedge funds** top-tier (Goldman Sachs, Citadel, Two Sigma) explorent les PINNs en interne, en R&D confidentielle
- Il n'existe **aucune plateforme commerciale** accessible offrant du pricing financier par PINN
- Les **PME financières, FinTechs, family offices, trésoreries d'entreprises** n'ont pas accès à ces technologies de pointe
- Les **outils disponibles** (QuantLib, Bloomberg) utilisent encore des méthodes numériques classiques des années 1990-2000

**C'est précisément ce gap que ce projet adresse.**

## Problématique 

**Comment résoudre efficacement des EDPs financières complexes, notamment dans des espaces de haute dimension, en exploitant la puissance des réseaux de neurones informés par la physique, et rendre cette technologie accessible via une interface web démocratisant le pricing quantitatif avancé ?**

## Les défis autour du projet

### Défi 1 — Formuler l'EDP comme problème d'optimisation

**Description :** Un PINN résout une EDP en minimisant une fonction de perte composite :

$$L_{\text{total}}
= L_{\text{EDP}}
+ \lambda_{1}\, L_{\text{CI}}
+ \lambda_{2}\, L_{\text{CL}}.$$

Où :
- $L_{\text{EDP}}$ : résidu de l'équation différentielle sur des points de collocation intérieurs
- $L_{\text{CI}}$ : erreur sur les conditions initiales (payoff à maturité)
- $L_{\text{CL}}$ : erreur sur les conditions aux limites (comportement aux frontières du domaine)

**Défi :** Choisir les bons points de collocation, équilibrer les termes de la loss, et s'assurer de la convergence vers la bonne solution.

**Lien avec ma formation MAM :** C'est exactement la formulation variationnelle / méthode de Galerkin étudier en éléments finis — mais ici, les fonctions de base sont les couches du réseau de neurones.

### Défi 2 — Maîtriser la malédiction de la dimensionnalité

**Description :** Pour une option sur un panier de 10 actifs, l'EDP vit dans un espace à 10 dimensions. Une grille de différences finies avec 100 points par dimension nécessiterait 100¹⁰ = 10²⁰ points, computationnellement impossible.

**Défi :** Un PINN échantillonne des points de collocation aléatoirement dans l'espace de haute dimension (méthode de Monte Carlo en quelque sorte), permettant théoriquement d'attaquer des dimensions élevées.

**Verrou actuel :** L'entraînement devient difficile en très haute dimension. Des architectures spécialisées (attention mechanisms, factorized networks) sont nécessaires.

### Défi 3 — Auto-différentiation pour les Greeks

**Description :** Les Greeks (Delta, Gamma, Vega...) sont les dérivées partielles du prix par rapport aux paramètres. Dans un PINN, ces dérivées sont calculées **exactement** par rétropropagation (automatic differentiation) — pas par différences finies approchées.

**Avantage :** Les Greeks obtenus sont plus précis et cohérents que par différences finies. C'est un avantage majeur du PINN sur les méthodes classiques.

**Défi :** Assurer la stabilité numérique des dérivées d'ordre 2 (Gamma) et plus.

### Défi 4 — Calibration et généralisation

**Description :** Un PINN classique est entraîné pour des paramètres fixes (σ, r, T donnés). En pratique, un trader veut pricer pour *n'importe quelle* combinaison de paramètres en temps réel.

**Innovation clé du projet :** Entraîner un **PINN paramétrique** qui prend en entrée non seulement $(S, t)$ mais aussi $(σ, r, T, K)$, et donne directement le prix. Une fois entraîné, ce réseau est un *universal pricer* pour la classe d'instruments considérée.

### Défi 5 — Validation et confiance du modèle

**Description :** En finance, un modèle doit être validé rigoureusement avant tout usage opérationnel. Il faut :
- Comparer aux solutions analytiques connues (Black-Scholes pour les cas simples)
- Comparer aux méthodes numériques de référence (Monte Carlo précis)
- Mesurer l'incertitude des prédictions (Bayesian PINNs)
- Tester la robustesse aux paramètres extrêmes

---

*Document descriptif · NeuroPrice Project · BELLO Soboure, Polytech Lyon MAM*  
