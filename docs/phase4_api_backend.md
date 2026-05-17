# Phase 4 — API & Backend

## Objectif

Transformer les modèles et références NeuroPrice en service web consommable via FastAPI.

## Fonctionnalités implémentées

```text
GET  /health
POST /api/v1/price
POST /api/v1/price/batch
```

## Instruments supportés

```text
european_call
down_out_barrier_call
asian_arithmetic_call
lookback_floating_call
```

## Architecture

```text
api/main.py
api/schemas.py
api/pricing_service.py
api/cache.py
```

### `api/schemas.py`

Définit les schémas Pydantic :

```text
PricingRequest
PricingResponse
BatchPricingRequest
BatchPricingResponse
```

Validation incluse :

```text
S0 > 0
K > 0 si requis
sigma entre 0 et 2
T > 0
barrier obligatoire pour down_out_barrier_call
batch entre 1 et 512 requêtes
```

### `api/pricing_service.py`

Service de pricing avec chargement des surrogates en mémoire via `lru_cache`.

Méthodes de pricing :

```text
european_call              -> formule analytique Black-Scholes
down_out_barrier_call      -> référence semi-analytique
asian_arithmetic_call      -> surrogate offline si checkpoint disponible, sinon Monte Carlo
lookback_floating_call     -> surrogate offline si checkpoint disponible, sinon Monte Carlo
```

### `api/cache.py`

Cache de résultats de pricing pour requêtes fréquentes.

```text
Redis si REDIS_URL est défini et accessible
fallback mémoire TTL sinon
TTL par défaut : 300 secondes
clé stable SHA-256 du payload JSON trié
```

### `api/main.py`

Application FastAPI avec :

```text
préchargement des modèles au démarrage FastAPI
cache par requête sur POST /api/v1/price
logging structuré simple
mapping PricingError -> HTTP 422
mapping erreur interne -> HTTP 500
Swagger automatique via /docs
OpenAPI via /openapi.json
```

## Lancement local

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Healthcheck :

```bash
curl http://127.0.0.1:8000/health
```

Documentation Swagger :

```text
http://127.0.0.1:8000/docs
```

OpenAPI JSON :

```text
http://127.0.0.1:8000/openapi.json
```

## Cache Redis optionnel

Sans configuration, l'API utilise un cache mémoire local.

Pour utiliser Redis :

```bash
set REDIS_URL=redis://localhost:6379/0
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Le healthcheck indique le backend actif :

```bash
curl http://127.0.0.1:8000/health
```

Réponse attendue sans Redis :

```json
{
  "status": "ok",
  "service": "neuroprice-api",
  "cache_backend": "memory"
}
```

## Exemple single pricing

```bash
curl -X POST http://127.0.0.1:8000/api/v1/price \
  -H "Content-Type: application/json" \
  -d '{
    "instrument": "european_call",
    "S0": 100,
    "K": 100,
    "sigma": 0.2,
    "r": 0.05,
    "T": 1.0,
    "greeks": true
  }'
```

## Exemple batch pricing

```bash
curl -X POST http://127.0.0.1:8000/api/v1/price/batch \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {
        "instrument": "european_call",
        "S0": 100,
        "K": 100,
        "sigma": 0.2,
        "r": 0.05,
        "T": 1.0
      },
      {
        "instrument": "down_out_barrier_call",
        "S0": 120,
        "K": 100,
        "barrier": 70,
        "sigma": 0.2,
        "r": 0.05,
        "T": 1.0
      }
    ]
  }'
```

## Tests

```bash
python -m pytest tests/test_api.py -q
```

## Test de charge Locust

Terminal 1 — lancer l'API :

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Vérifier le healthcheck avant le test :

```bash
curl http://127.0.0.1:8000/health
```

Terminal 2 — lancer Locust en mode headless avec cible 100 req/s approximée :

```bash
locust -f tests/locustfile.py --host http://127.0.0.1:8000 --headless -u 100 -r 20 -t 60s
```

Si Locust retourne `ConnectionRefusedError`, l'API n'est pas lancée ou n'écoute pas sur `127.0.0.1:8000`.

Critère Phase 4 :

```text
0 échec HTTP
latence p95 stable
débit proche de 100 req/s sur endpoints cachables
```

Résultat validé :

```text
Durée : 60s
Utilisateurs : 100
Ramp-up : 20 utilisateurs/s
Requêtes totales : 12 745
Échecs : 0 (0.00%)
Débit agrégé : 214.43 req/s
POST /api/v1/price : 169.01 req/s
POST /api/v1/price/batch : 43.74 req/s
p95 agrégé : 760 ms
Exit code : 0
```

## Checklist Phase 4 — Semaine 23-25

```text
[x] Endpoints de pricing POST /api/v1/price
[x] Batch pricing POST /api/v1/price/batch
[x] Validation des entrées Pydantic
[x] Gestion des erreurs et exceptions
[x] Logging structuré simple
[x] Documentation Swagger auto-générée
```

## Checklist Phase 4 — Semaine 26-27

```text
[x] Cache Redis pour résultats fréquents
[x] Fallback cache mémoire si Redis indisponible
[x] Chargement des modèles en mémoire au démarrage
[x] Tests de charge Locust : 100 req/s cible
[x] Documentation Swagger auto-générée
```

## Dockerisation

Fichiers ajoutés :

```text
Dockerfile
docker-compose.yml
.dockerignore
.github/workflows/ci.yml
```

### Build image API

```bash
docker build -t neuroprice-api:local .
```

### Lancement API seule

```bash
docker run --rm -p 8000:8000 neuroprice-api:local
```

### Lancement API + Redis

```bash
docker compose up --build
```

### Test end-to-end local

Healthcheck :

```bash
curl http://127.0.0.1:8000/health
```

Réponse attendue avec `docker compose` :

```json
{
  "status": "ok",
  "service": "neuroprice-api",
  "cache_backend": "redis"
}
```

Pricing API :

```bash
curl -X POST http://127.0.0.1:8000/api/v1/price \
  -H "Content-Type: application/json" \
  -d '{
    "instrument": "european_call",
    "S0": 100,
    "K": 100,
    "sigma": 0.2,
    "r": 0.05,
    "T": 1.0,
    "greeks": true
  }'
```

Arrêt :

```bash
docker compose down
```

### CI/CD GitHub Actions

Le workflow `.github/workflows/ci.yml` exécute :

```text
installation des dépendances Python
pytest tests -q
docker build -t neuroprice-api:ci .
```

Déclencheurs :

```text
push sur main
pull request vers main
workflow_dispatch manuel
```

## Checklist Phase 4 — Semaine 28

```text
[x] Dockerfile API
[x] docker-compose complet API + Redis
[x] Test en local end-to-end documenté
[x] CI/CD GitHub Actions : tests automatiques
```

## Prochaines étapes

```text
1. Exécuter le test end-to-end Docker local.
2. Publier éventuellement l'image API sur un registre.
3. Préparer la Phase 5 Frontend & Déploiement.
```
