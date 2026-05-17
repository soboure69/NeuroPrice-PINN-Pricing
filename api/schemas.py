from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

InstrumentType = Literal[
    "european_call",
    "down_out_barrier_call",
    "asian_arithmetic_call",
    "lookback_floating_call",
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
    method: PricingMethod = "auto"
    greeks: bool = False

    @model_validator(mode="after")
    def validate_instrument_fields(self) -> "PricingRequest":
        if self.instrument in {"european_call", "down_out_barrier_call", "asian_arithmetic_call"} and self.K is None:
            raise ValueError("K is required for this instrument")
        if self.instrument == "down_out_barrier_call" and self.barrier is None:
            raise ValueError("barrier is required for down_out_barrier_call")
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
