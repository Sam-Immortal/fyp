from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class NlpScoreRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000, description="SMS message content")
    sender: str | None = Field(default=None, max_length=255, description="Sender phone number or ID")
    timestamp: str | None = Field(default=None, description="ISO 8601 timestamp of receipt")

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "URGENT: Your bank account will be closed. Verify now at bit.ly/fake-bank",
                "sender": "+15551234567",
                "timestamp": "2025-01-15T10:30:00Z",
            }
        }
    }


class ScamPattern(BaseModel):
    urgency: bool = False
    authority_impersonation: bool = False
    credential_request: bool = False
    threat: bool = False
    reward: bool = False


class NlpScoreResponse(BaseModel):
    event_id: int
    status: Literal["pending", "completed", "failed"]
    scam_probability: float = Field(ge=0.0, le=1.0)
    risk_level: Literal["pending", "low", "medium", "high", "critical"]
    patterns: ScamPattern
    explanation: str
    model: str


class NlpScoreEventResponse(BaseModel):
    id: int
    status: Literal["pending", "completed", "failed"]
    sender: str
    message_preview: str
    score: float
    threshold: float
    is_high_risk: bool
    risk_level: Literal["pending", "low", "medium", "high", "critical"] | None = None
    patterns: ScamPattern | None = None
    explanation: str | None = None
    error: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}