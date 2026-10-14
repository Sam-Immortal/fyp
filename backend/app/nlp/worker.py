"""NLP worker for asynchronous scam classification via Hugging Face Inference API."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx
import jwt

from app.core.config import get_settings
from app.core.security import create_access_token
from app.db.models import NlpScoreEvent, User
from app.db.session import async_session_factory

logger = logging.getLogger(__name__)

_settings = get_settings()

HF_INFERENCE_URL = "https://api-inference.huggingface.co/models"


@dataclass
class NlpAnalysisResult:
    scam_probability: float
    risk_level: str
    patterns: dict
    explanation: str
    model: str


def build_scam_prompt(text: str, sender: str | None) -> str:
    """Build structured prompt for scam classification."""
    sender_info = f" from {sender}" if sender else ""
    return f"""You are a cybersecurity expert analyzing SMS messages for scam/fraud indicators.
Analyze the following SMS message{sender_info} and return a JSON object with exactly these fields:

{{
  "scam_probability": <float 0.0-1.0>,
  "risk_level": "<low|medium|high|critical>",
  "patterns": {{
    "urgency": <bool>,
    "authority_impersonation": <bool>,
    "credential_request": <bool>,
    "threat": <bool>,
    "reward": <bool>
  }},
  "explanation": "<concise explanation of detected patterns>"
}}

Message: "{text}"

Scam indicators to detect:
- urgency: artificial time pressure ("act now", "urgent", "immediate", "expires today")
- authority_impersonation: pretending to be bank, government, tech support, delivery service
- credential_request: asking for passwords, OTPs, PINs, SSN, credit card details
- threat: negative consequences if not complied ("account closed", "legal action", "arrest")
- reward: unsolicited prizes, lottery, refund, gift card offers

Return ONLY the JSON object, no extra text."""


async def call_hf_inference(prompt: str) -> str:
    """Call Hugging Face Inference API for Mistral-7B."""
    token = _settings.hf_token or os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not configured")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 256,
            "temperature": 0.1,
            "top_p": 0.9,
            "return_full_text": False,
        },
    }

    url = f"{HF_INFERENCE_URL}/{_settings.hf_llm_model}"
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    # HF Inference API returns list of dicts with "generated_text"
    if isinstance(data, list) and data:
        return data[0].get("generated_text", "")
    return ""


def parse_llm_response(raw: str) -> NlpAnalysisResult:
    """Parse and validate LLM JSON response."""
    # Extract JSON from response (may have extra text)
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in response")

    json_str = raw[start:end + 1]
    data = json.loads(json_str)

    # Validate and normalize
    prob = float(data.get("scam_probability", 0.0))
    prob = max(0.0, min(1.0, prob))

    risk = str(data.get("risk_level", "low")).lower()
    if risk not in ("low", "medium", "high", "critical"):
        risk = "low"

    patterns = data.get("patterns", {})
    patterns_normalized = {
        "urgency": bool(patterns.get("urgency", False)),
        "authority_impersonation": bool(patterns.get("authority_impersonation", False)),
        "credential_request": bool(patterns.get("credential_request", False)),
        "threat": bool(patterns.get("threat", False)),
        "reward": bool(patterns.get("reward", False)),
    }

    explanation = str(data.get("explanation", "")).strip()

    return NlpAnalysisResult(
        scam_probability=prob,
        risk_level=risk,
        patterns=patterns_normalized,
        explanation=explanation,
        model=_settings.hf_llm_model,
    )


async def run_nlp_analysis(event_id: int, text: str, sender: str | None) -> None:
    """Run NLP analysis and update database event."""
    logger.info("Starting NLP analysis for event %d", event_id)

    prompt = build_scam_prompt(text, sender)
    raw_response = ""
    try:
        raw_response = await call_hf_inference(prompt)
        result = parse_llm_response(raw_response)
    except Exception as e:
        logger.exception("NLP analysis failed for event %d: %s", event_id, e)
        # Mark as failed with minimal info
        result = NlpAnalysisResult(
            scam_probability=0.0,
            risk_level="low",
            patterns={k: False for k in ["urgency", "authority_impersonation", "credential_request", "threat", "reward"]},
            explanation=f"Analysis failed: {e}",
            model=_settings.hf_llm_model,
        )

    # Update event in database
    async with async_session_factory() as session:
        event = await session.get(NlpScoreEvent, event_id)
        if not event:
            logger.error("NlpScoreEvent %d not found", event_id)
            return

        event.score = result.scam_probability
        event.threshold = _settings.safety_threshold
        event.is_high_risk = result.scam_probability >= _settings.safety_threshold
        event.raw_response = raw_response[:5000] if raw_response else None
        await session.commit()

    logger.info(
        "NLP analysis complete for event %d: score=%.3f risk=%s patterns=%s",
        event_id,
        result.scam_probability,
        result.risk_level,
        result.patterns,
    )


async def queue_nlp_analysis(event_id: int, text: str, sender: str | None) -> None:
    """Queue NLP analysis via Taskiq (fire-and-forget)."""
    # Import here to avoid circular dependency
    from app.nlp.tasks import nlp_analysis_task

    await nlp_analysis_task.kiq(event_id, text, sender)