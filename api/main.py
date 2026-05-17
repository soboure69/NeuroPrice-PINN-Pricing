from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from api.cache import get_cache
from api.pricing_service import PricingError, preload_models, price
from api.schemas import BatchPricingRequest, BatchPricingResponse, PricingRequest, PricingResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("neuroprice.api")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    loaded_models = preload_models()
    cache = get_cache()
    logger.info("startup loaded_models=%s cache_backend=%s", loaded_models, cache.backend)
    yield


app = FastAPI(
    title="NeuroPrice API",
    description="PINN and Monte Carlo pricing service for vanilla and exotic options.",
    version="0.4.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    logger.info("method=%s path=%s status=%s elapsed_ms=%.3f", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


@app.exception_handler(PricingError)
async def pricing_error_handler(_: Request, exc: PricingError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "neuroprice-api", "cache_backend": get_cache().backend}


@app.post("/api/v1/price", response_model=PricingResponse)
def price_instrument(request: PricingRequest) -> PricingResponse:
    try:
        cache = get_cache()
        key = cache.make_key(request.model_dump())
        cached = cache.get(key)
        if cached is not None:
            cached["warnings"] = [*cached.get("warnings", []), f"cache hit: {cache.backend}"]
            return PricingResponse(**cached)
        response = price(request)
        cache.set(key, response.model_dump())
        return response
    except PricingError:
        raise
    except Exception as exc:
        logger.exception("Unhandled pricing error")
        raise HTTPException(status_code=500, detail="Internal pricing error") from exc


@app.post("/api/v1/price/batch", response_model=BatchPricingResponse)
def price_batch(request: BatchPricingRequest) -> BatchPricingResponse:
    start = time.perf_counter()
    results = [price_instrument(item) for item in request.requests]
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return BatchPricingResponse(results=results, count=len(results), total_inference_time_ms=round(elapsed_ms, 4))
