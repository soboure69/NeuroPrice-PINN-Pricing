from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

InstrumentType = Literal[
    "european_call",
    "down_out_barrier_call",
    "asian_arithmetic_call",
    "lookback_floating_call",
    "basket_call",
    "heston_call",
]

PricingMethod = Literal["auto", "model", "reference"]


class PricingRequest(BaseModel):
    instrument: InstrumentType
    S0: float = Field(gt=0.0)
    K: float | None = Field(default=None, gt=0.0)
    sigma: float = Field(gt=0.0, lt=2.0)
    r: float = Field(ge=-0.10, le=1.0)
    T: float = Field(gt=0.0, le=30.0)
    barrier: float | None = Field(default=None, gt=0.0)
    spots: list[float] | None = None
    sigmas: list[float] | None = None
    weights: list[float] | None = None
    correlation: float | None = Field(default=0.0, ge=-0.99, le=0.99)
    n_paths: int | None = Field(default=20000, ge=1000, le=200000)
    n_steps: int | None = Field(default=128, ge=8, le=1024)
    seed: int | None = Field(default=123, ge=0)
    v0: float | None = Field(default=None, gt=0.0, lt=4.0)
    kappa: float | None = Field(default=None, gt=0.0, le=20.0)
    theta: float | None = Field(default=None, gt=0.0, lt=4.0)
    xi: float | None = Field(default=None, gt=0.0, le=5.0)
    rho: float | None = Field(default=None, ge=-0.99, le=0.99)
    method: PricingMethod = "auto"
    greeks: bool = False

    @model_validator(mode="after")
    def validate_instrument_fields(self) -> "PricingRequest":
        if self.instrument in {"european_call", "down_out_barrier_call", "asian_arithmetic_call", "basket_call", "heston_call"} and self.K is None:
            raise ValueError("K is required for this instrument")
        if self.instrument == "down_out_barrier_call" and self.barrier is None:
            raise ValueError("barrier is required for down_out_barrier_call")
        if self.instrument == "basket_call":
            if self.spots is None or self.sigmas is None or self.weights is None:
                raise ValueError("spots, sigmas, and weights are required for basket_call")
            if not (2 <= len(self.spots) <= 10):
                raise ValueError("basket_call requires between 2 and 10 assets")
            if len(self.spots) != len(self.sigmas) or len(self.spots) != len(self.weights):
                raise ValueError("spots, sigmas, and weights must have the same length")
            if any(value <= 0.0 for value in self.spots):
                raise ValueError("all basket spots must be positive")
            if any(value <= 0.0 or value >= 2.0 for value in self.sigmas):
                raise ValueError("all basket sigmas must be in (0, 2)")
            if any(value < 0.0 for value in self.weights) or sum(self.weights) <= 0.0:
                raise ValueError("basket weights must be non-negative and sum to a positive value")
        if self.instrument == "heston_call":
            if self.v0 is None or self.kappa is None or self.theta is None or self.xi is None or self.rho is None:
                raise ValueError("v0, kappa, theta, xi, and rho are required for heston_call")
        return self


class PricingResponse(BaseModel):
    instrument: InstrumentType
    price: float
    method: str
    model_version: str
    inference_time_ms: float
    greeks: dict[str, float] | None = None
    warnings: list[str] = Field(default_factory=list)


class BatchPricingRequest(BaseModel):
    requests: list[PricingRequest] = Field(min_length=1, max_length=512)


class BatchPricingResponse(BaseModel):
    results: list[PricingResponse]
    count: int
    total_inference_time_ms: float
