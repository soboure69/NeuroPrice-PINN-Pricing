# Phase 5 — Frontend & Déploiement

## Objectif

Créer une interface web utilisable par des non-programmeurs pour consommer l'API NeuroPrice.

## Frontend Next.js

Application créée dans :

```text
frontend/
```

Fichiers principaux :

```text
frontend/package.json
frontend/app/layout.tsx
frontend/app/page.tsx
frontend/app/globals.css
frontend/components/PricingDashboard.tsx
frontend/.env.example
frontend/next.config.mjs
frontend/tailwind.config.ts
frontend/tsconfig.json
frontend/postcss.config.mjs
```

## Fonctionnalités implémentées

```text
Landing page produit
Présentation des plans SaaS MVP
Dashboard de pricing
Formulaire instrument + paramètres
Connexion à POST /api/v1/price
Affichage prix, modèle, temps d'inférence et Greeks
Graphique Plotly surface de prix preview
Configuration API via NEXT_PUBLIC_API_URL
Connexion beta locale via localStorage
Quota freemium mensuel côté navigateur
```

## Authentification et quotas MVP

La première version Phase 5 inclut un MVP local pour valider l'expérience SaaS sans backend utilisateur.

Implémentation :

```text
frontend/components/AuthProvider.tsx
frontend/components/AuthQuotaPanel.tsx
frontend/components/PricingDashboard.tsx
```

Fonctionnement :

```text
Session stockée dans localStorage
Plans : free, quant, enterprise
Quota free : 50 pricings / mois
Quota quant : 10 000 pricings / mois
Quota enterprise : quota élevé pour tests MVP
Blocage du bouton pricing sans session utilisateur
Décrément du quota uniquement après succès API
Réinitialisation automatique par mois YYYY-MM
```

Limites de cette version :

```text
Pas d'authentification serveur réelle
Quota modifiable côté navigateur
Pas de base de données utilisateur
Pas de billing Stripe
```

Migration recommandée pour production :

```text
NextAuth.js pour OAuth/email login
PostgreSQL Neon pour users, plans, usage_events
Middleware API pour enforce quota côté serveur
Stripe pour subscription plans
Sentry pour erreurs frontend/backend
PostHog ou Plausible pour analytics produit
```

## Lancement local

Terminal 1 — lancer l'API :

```bash
docker compose up --build
```

Vérifier :

```bash
curl http://127.0.0.1:8000/health
```

Terminal 2 — installer et lancer le frontend :

```bash
cd frontend
npm install
copy .env.example .env.local
npm run dev
```

URL frontend :

```text
http://127.0.0.1:3000
```

## Configuration

Variable d'environnement locale :

```text
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

Sur Vercel, définir :

```text
NEXT_PUBLIC_API_URL=https://<api-cloud-url>
```

## Build de validation

```bash
cd frontend
npm run build
```

Commandes Windows depuis la racine du projet :

```bat
cd C:\Users\bello\Documents\NeuroPrice-PINN-Pricing\frontend
npm run build
```

Checklist de validation MVP :

```text
[ ] npm run build passe sans erreur
[ ] docker compose up --build démarre Redis + API
[ ] GET http://127.0.0.1:8000/health retourne status=ok
[ ] npm run dev démarre le frontend sur http://127.0.0.1:3000
[ ] Pricing bloqué sans connexion locale
[ ] Connexion locale beta fonctionne
[ ] Pricing API fonctionne après connexion
[ ] Quota décrémenté après succès API
[ ] Déconnexion sans erreur runtime
```

## Déploiement Vercel

Pré-requis :

```text
Root directory : frontend
Framework preset : Next.js
Build command : npm run build
Output : .next
```

Le fichier suivant explicite la configuration Vercel :

```text
frontend/vercel.json
```

Variables d'environnement Vercel :

```text
NEXT_PUBLIC_API_URL=https://<api-cloud-url>
```

## Déploiement API

Cible MVP recommandée :

```text
Render ou Railway pour l'API FastAPI containerisée
Upstash Redis pour le cache managé
```

Variables API cloud :

```text
REDIS_URL=<upstash-redis-url>
```

Le fichier suivant fournit une configuration Render Docker minimale :

```text
render.yaml
```

Paramètres Render recommandés :

```text
Service type : Web Service
Environment : Docker
Dockerfile path : ./Dockerfile
Health check path : /health
Environment variable : REDIS_URL=<upstash-redis-url>
```

Image Docker MVP :

```text
Le Dockerfile utilise requirements-api.txt pour éviter d'installer Jupyter, Locust, Torch CUDA et les artefacts lourds sur Render Free.
Les options exotiques utilisent les méthodes de référence si les checkpoints Torch ne sont pas disponibles.
```

## Authentification et quotas

La première itération pose l'interface publique, le dashboard et un MVP auth/quota local.

À intégrer ensuite pour production :

```text
NextAuth.js pour sessions utilisateur
PostgreSQL Neon pour users, plans et quotas
Middleware quota par tier freemium
Analytics PostHog ou Plausible
Sentry pour monitoring erreurs
```

## Checklist Phase 5 — Semaine 29-32

```text
[x] Landing page présentation produit et plans
[x] Dashboard de pricing formulaire + résultats
[x] Graphiques Plotly : surface de prix preview
[x] MVP authentification locale beta
[x] MVP quotas par tier freemium côté navigateur
[ ] Système d'authentification NextAuth.js production
[ ] Gestion des quotas par tier freemium côté serveur
```

## Checklist Phase 5 — Semaine 33-34

```text
[ ] Déploiement API sur Render / Railway
[ ] Déploiement Frontend sur Vercel
[ ] Base de données PostgreSQL Neon
[ ] Redis Upstash
[ ] Monitoring Sentry
[ ] Analytics Plausible ou PostHog
```
