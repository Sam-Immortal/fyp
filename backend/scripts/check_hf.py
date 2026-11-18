#!/usr/bin/env python3
"""Check Hugging Face Inference Providers API connectivity and model availability.

Usage:
    HF_TOKEN=your_token python -m backend.scripts.check_hf

Exits with:
    0 - Success (model available, benign and scam samples parsed correctly)
    1 - Failure (auth error, model not found, parsing error, etc.)
"""

from __future__ import annotations

import asyncio
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.nlp.worker import (
    HF_INFERENCE_BASE_URL,
    HF_LLM_MODEL,
    HF_TOKEN,
    build_scam_prompt,
    call_hf_inference,
    parse_llm_response,
)


async def main() -> int:
    # Check HF_TOKEN
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not configured (set HF_TOKEN environment variable)")
        return 1

    print(f"Using model: {HF_LLM_MODEL}")
    print(f"Using endpoint: {HF_INFERENCE_BASE_URL}")
    print()

    # Test 1: Benign message
    print("Test 1: Benign message")
    benign_text = "Hi, are we still meeting at 5pm for coffee?"
    benign_sender = "+15551234567"
    benign_messages = build_scam_prompt(benign_text, benign_sender)

    try:
        raw = await call_hf_inference(benign_messages)
        result = parse_llm_response(raw)
        print(f"  HTTP Status: 200 (success)")
        print(f"  Model: {result.model}")
        print(f"  Scam Probability: {result.scam_probability:.3f}")
        print(f"  Risk Level: {result.risk_level}")
        print(f"  Patterns: {result.patterns}")
        print(f"  Explanation: {result.explanation[:100]}...")
        
        # Validate benign expectations
        if result.risk_level not in ("low", "medium"):
            print(f"  WARNING: Expected low/medium risk for benign message, got {result.risk_level}")
    except Exception as e:
        print(f"  ERROR: {e}")
        return 1

    print()

    # Test 2: Scam message
    print("Test 2: Scam message")
    scam_text = "URGENT: Your bank account will be closed in 24 hours. Verify immediately at http://fake-bank.com/verify or face legal action."
    scam_sender = "+15559876543"
    scam_messages = build_scam_prompt(scam_text, scam_sender)

    try:
        raw = await call_hf_inference(scam_messages)
        result = parse_llm_response(raw)
        print(f"  HTTP Status: 200 (success)")
        print(f"  Model: {result.model}")
        print(f"  Scam Probability: {result.scam_probability:.3f}")
        print(f"  Risk Level: {result.risk_level}")
        print(f"  Patterns: {result.patterns}")
        print(f"  Explanation: {result.explanation[:100]}...")
        
        # Validate scam expectations
        if result.risk_level not in ("high", "critical"):
            print(f"  WARNING: Expected high/critical risk for scam message, got {result.risk_level}")
    except Exception as e:
        print(f"  ERROR: {e}")
        return 1

    print()
    print("All tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))