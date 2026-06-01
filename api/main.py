from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

import sentry_sdk
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from posthog import Posthog
from sentry_sdk.integrations.fastapi import FastApiIntegration

from api.cache import get_cache
from api.pricing_service import PricingError, preload_models, price
from api.quota import get_plan_catalog, get_quota_store
from api.schemas import BatchPricingRequest, BatchPricingResponse, PricingRequest, PricingResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("neuroprice.api")

DEFAULT_CORS_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"
cors_origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", DEFAULT_CORS_ORIGINS).split(",") if origin.strip()]
sentry_dsn = os.getenv("SENTRY_DSN")

_posthog_key = os.getenv("POSTHOG_PROJECT_API_KEY", "")
posthog = Posthog(
    project_api_key=_posthog_key,
    host=os.getenv("POSTHOG_HOST", "https://eu.i.posthog.com"),
) if _posthog_key else None

if sentry_dsn:
    sentry_sdk.init(
        dsn=sentry_dsn,
        environment=os.getenv("SENTRY_ENVIRONMENT", os.getenv("RENDER_SERVICE_NAME", "production")),
        release=os.getenv("SENTRY_RELEASE"),
        traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
        integrations=[FastApiIntegration()],
    )


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    loaded_models = preload_models()
    cache = get_cache()
    quota_store = get_quota_store()
    quota_store.init_schema()
    logger.info("startup loaded_models=%s cache_backend=%s quota_backend=%s", loaded_models, cache.backend, quota_store.backend)
    yield
    if posthog:
        await asyncio.to_thread(posthog.flush)


app = FastAPI(
    title="NeuroPrice API",
    description="PINN and Monte Carlo pricing service for vanilla and exotic options.",
    version="0.4.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    logger.info("method=%s path=%s status=%s elapsed_ms=%.3f", request.method, request.url.path, response.status_code, elapsed_ms)
    response.headers["X-NeuroPrice-CORS-Origins"] = ",".join(cors_origins)
    if posthog:
        distinct_id = request.headers.get("X-NeuroPrice-User-Email") or "anonymous"
        await asyncio.to_thread(
            posthog.capture,
            distinct_id=distinct_id,
            event="api_request",
            properties={
                "method": request.method,
                "path": str(request.url.path),
                "status_code": response.status_code,
                "elapsed_ms": round(elapsed_ms, 3),
            },
        )
    return response


@app.exception_handler(PricingError)
async def pricing_error_handler(_: Request, exc: PricingError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "neuroprice-api", "cache_backend": get_cache().backend, "quota_backend": get_quota_store().backend}


@app.get("/debug/cors")
def debug_cors(request: Request) -> dict[str, object]:
    return {"origin": request.headers.get("origin"), "allowed_origins": cors_origins}


@app.get("/api/v1/plans")
def list_plans() -> dict[str, object]:
    return {"plans": get_plan_catalog()}


@app.get("/api/v1/admin/summary")
def admin_summary(http_request: Request) -> dict[str, object]:
    expected_secret = os.getenv("ADMIN_API_SECRET")
    provided_secret = http_request.headers.get("X-NeuroPrice-Admin-Secret")
    if not expected_secret or provided_secret != expected_secret:
        raise HTTPException(status_code=401, detail="Invalid admin API secret.")
    return get_quota_store().admin_summary()


@app.post("/api/v1/internal/users/plan")
def update_user_plan(payload: dict[str, str], http_request: Request) -> dict[str, str]:
    expected_secret = os.getenv("INTERNAL_API_SECRET")
    provided_secret = http_request.headers.get("X-NeuroPrice-Internal-Secret")
    if not expected_secret or provided_secret != expected_secret:
        raise HTTPException(status_code=401, detail="Invalid internal API secret.")
    email = payload.get("email")
    plan = payload.get("plan")
    if not email or not plan:
        raise HTTPException(status_code=422, detail="Missing email or plan.")
    try:
        return get_quota_store().set_plan(email, plan)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/v1/price", response_model=PricingResponse)
def price_instrument(request: PricingRequest, http_request: Request) -> PricingResponse:
    try:
        user_email = http_request.headers.get("X-NeuroPrice-User-Email")
        user_plan = http_request.headers.get("X-NeuroPrice-User-Plan")
        quota_store = get_quota_store()
        quota_status = quota_store.check(user_email, user_plan)
        if not quota_status.allowed:
            if posthog and user_email:
                posthog.capture(distinct_id=user_email, event="quota_exceeded", properties={"plan": quota_status.plan, "quota": quota_status.quota, "used": quota_status.used})
            raise HTTPException(status_code=429, detail=f"Quota mensuel épuisé pour le plan {quota_status.plan} ({quota_status.used}/{quota_status.quota}).")
        cache = get_cache()
        key = cache.make_key(request.model_dump())
        cached = cache.get(key)
        if cached is not None:
            quota_status = quota_store.consume(user_email, user_plan)
            cached["warnings"] = [*cached.get("warnings", []), f"cache hit: {cache.backend}"]
            cached["warnings"] = [*cached["warnings"], f"server quota: {quota_status.remaining}/{quota_status.quota} remaining ({quota_status.backend})"]
            return PricingResponse(**cached)
        response = price(request)
        quota_status = quota_store.consume(user_email, user_plan)
        response.warnings = [*response.warnings, f"server quota: {quota_status.remaining}/{quota_status.quota} remaining ({quota_status.backend})"]
        cache.set(key, response.model_dump())
        return response
    except PricingError:
        raise
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unhandled pricing error")
        raise HTTPException(status_code=500, detail="Internal pricing error") from exc


@app.post("/api/v1/price/batch", response_model=BatchPricingResponse)
def price_batch(request: BatchPricingRequest) -> BatchPricingResponse:
    start = time.perf_counter()
    results = [price(item) for item in request.requests]
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return BatchPricingResponse(results=results, count=len(results), total_inference_time_ms=round(elapsed_ms, 4))
