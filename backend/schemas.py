"""Pydantic request models for the ZeroLeakX API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class KeyWindow(BaseModel):
    """One window of per-keystroke timing (ms). No key content is sent."""
    dwell: list[float] = Field(default_factory=list)    # hold time per key
    flight: list[float] = Field(default_factory=list)   # gap between keys
    backspace_rate: float = 0.0
    key_count: int = 0


class LoginReq(BaseModel):
    username: str = "guest"


class EnrollReq(BaseModel):
    windows: list[KeyWindow] = Field(default_factory=list)
    pointer: list[list[float]] | None = None
    device_fp: str | None = None


class ScoreReq(BaseModel):
    window: KeyWindow | None = None
    pointer: list[float] | None = None
    device_fp: str | None = None
    nav: bool = False
    txn: bool = False
    paste: str | None = None          # field name a value was pasted into (real signal)
    # Scenario overrides (demo simulator — scored for real):
    device_ok: bool | None = None
    geo_km: float | None = None
    swipe: float | None = None
    nav_score: float | None = None
    channel: str = "web"              # "web" | "mobile"
    label: str = "live"


class StepupVerifyReq(BaseModel):
    code: str
