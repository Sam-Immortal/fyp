"""Tests for strict NLP response parsing."""

from __future__ import annotations

import re

import pytest
import httpx

from app.nlp.worker import parse_llm_response, NlpParseError, call_hf_inference


def test_parse_valid_response():
    """Valid response with all required fields parses correctly."""
    raw = '''
    {
      "scam_probability": 0.85,
      "risk_level": "high",
      "patterns": {
        "urgency": true,
        "authority_impersonation": false,
        "credential_request": true,
        "threat": false,
        "reward": false
      },
      "explanation": "Message shows urgency and credential request"
    }
    '''
    result = parse_llm_response(raw)
    assert result.scam_probability == 0.85
    assert result.risk_level == "high"
    assert result.patterns == {
        "urgency": True,
        "authority_impersonation": False,
        "credential_request": True,
        "threat": False,
        "reward": False,
    }
    assert result.explanation == "Message shows urgency and credential request"


def test_parse_valid_response_with_code_fences():
    """Valid response wrapped in code fences parses correctly."""
    raw = '''
    ```json
    {
      "scam_probability": 0.3,
      "risk_level": "low",
      "patterns": {
        "urgency": false,
        "authority_impersonation": false,
        "credential_request": false,
        "threat": false,
        "reward": false
      },
      "explanation": "No scam indicators detected"
    }
    ```
    '''
    result = parse_llm_response(raw)
    assert result.scam_probability == 0.3
    assert result.risk_level == "low"
    assert all(v is False for v in result.patterns.values())


def test_parse_missing_scam_probability():
    """Missing scam_probability raises NlpParseError."""
    raw = '''
    {
      "risk_level": "low",
      "patterns": {"urgency": false, "authority_impersonation": false, "credential_request": false, "threat": false, "reward": false},
      "explanation": "test"
    }
    '''
    with pytest.raises(NlpParseError, match="Missing required field: scam_probability"):
        parse_llm_response(raw)


def test_parse_missing_risk_level():
    """Missing risk_level raises NlpParseError."""
    raw = '''
    {
      "scam_probability": 0.5,
      "patterns": {"urgency": false, "authority_impersonation": false, "credential_request": false, "threat": false, "reward": false},
      "explanation": "test"
    }
    '''
    with pytest.raises(NlpParseError, match="Missing required field: risk_level"):
        parse_llm_response(raw)


def test_parse_missing_patterns():
    """Missing patterns raises NlpParseError."""
    raw = '''
    {
      "scam_probability": 0.5,
      "risk_level": "low",
      "explanation": "test"
    }
    '''
    with pytest.raises(NlpParseError, match="Missing required field: patterns"):
        parse_llm_response(raw)


def test_parse_missing_explanation():
    """Missing explanation raises NlpParseError."""
    raw = '''
    {
      "scam_probability": 0.5,
      "risk_level": "low",
      "patterns": {"urgency": false, "authority_impersonation": false, "credential_request": false, "threat": false, "reward": false}
    }
    '''
    with pytest.raises(NlpParseError, match="Missing required field: explanation"):
        parse_llm_response(raw)


def test_parse_invalid_scam_probability_type():
    """Non-numeric scam_probability raises NlpParseError."""
    raw = '''
    {
      "scam_probability": "high",
      "risk_level": "low",
      "patterns": {"urgency": false, "authority_impersonation": false, "credential_request": false, "threat": false, "reward": false},
      "explanation": "test"
    }
    '''
    with pytest.raises(NlpParseError, match="scam_probability must be a number"):
        parse_llm_response(raw)


def test_parse_scam_probability_out_of_range():
    """scam_probability outside 0-1 raises NlpParseError."""
    for val in [-0.1, 1.1, 2.0]:
        raw = f'''
        {{
          "scam_probability": {val},
          "risk_level": "low",
          "patterns": {{"urgency": false, "authority_impersonation": false, "credential_request": false, "threat": false, "reward": false}},
          "explanation": "test"
        }}
        '''
        with pytest.raises(NlpParseError, match="scam_probability must be between 0.0 and 1.0"):
            parse_llm_response(raw)


def test_parse_invalid_risk_level():
    """Invalid risk_level raises NlpParseError."""
    raw = '''
    {
      "scam_probability": 0.5,
      "risk_level": "extreme",
      "patterns": {"urgency": false, "authority_impersonation": false, "credential_request": false, "threat": false, "reward": false},
      "explanation": "test"
    }
    '''
    with pytest.raises(NlpParseError, match=re.escape("risk_level must be one of low|medium|high|critical")):
        parse_llm_response(raw)


def test_parse_risk_level_not_string():
    """Non-string risk_level raises NlpParseError."""
    raw = '''
    {
      "scam_probability": 0.5,
      "risk_level": 123,
      "patterns": {"urgency": false, "authority_impersonation": false, "credential_request": false, "threat": false, "reward": false},
      "explanation": "test"
    }
    '''
    with pytest.raises(NlpParseError, match="risk_level must be a string"):
        parse_llm_response(raw)


def test_parse_patterns_not_object():
    """Non-object patterns raises NlpParseError."""
    raw = '''
    {
      "scam_probability": 0.5,
      "risk_level": "low",
      "patterns": "not an object",
      "explanation": "test"
    }
    '''
    with pytest.raises(NlpParseError, match="patterns must be an object"):
        parse_llm_response(raw)


def test_parse_missing_pattern_field():
    """Missing pattern field raises NlpParseError."""
    raw = '''
    {
      "scam_probability": 0.5,
      "risk_level": "low",
      "patterns": {"urgency": false, "authority_impersonation": false, "credential_request": false, "threat": false},
      "explanation": "test"
    }
    '''
    with pytest.raises(NlpParseError, match="patterns missing required field: reward"):
        parse_llm_response(raw)


def test_parse_pattern_field_not_boolean():
    """Non-boolean pattern field raises NlpParseError."""
    raw = '''
    {
      "scam_probability": 0.5,
      "risk_level": "low",
      "patterns": {"urgency": "yes", "authority_impersonation": false, "credential_request": false, "threat": false, "reward": false},
      "explanation": "test"
    }
    '''
    with pytest.raises(NlpParseError, match="patterns.urgency must be boolean"):
        parse_llm_response(raw)


def test_parse_pattern_field_null():
    """Null pattern field raises NlpParseError."""
    raw = '''
    {
      "scam_probability": 0.5,
      "risk_level": "low",
      "patterns": {"urgency": null, "authority_impersonation": false, "credential_request": false, "threat": false, "reward": false},
      "explanation": "test"
    }
    '''
    with pytest.raises(NlpParseError, match="patterns.urgency must be boolean"):
        parse_llm_response(raw)


def test_parse_empty_explanation():
    """Empty explanation raises NlpParseError."""
    raw = '''
    {
      "scam_probability": 0.5,
      "risk_level": "low",
      "patterns": {"urgency": false, "authority_impersonation": false, "credential_request": false, "threat": false, "reward": false},
      "explanation": ""
    }
    '''
    with pytest.raises(NlpParseError, match="explanation must be non-empty"):
        parse_llm_response(raw)


def test_parse_explanation_not_string():
    """Non-string explanation raises NlpParseError."""
    raw = '''
    {
      "scam_probability": 0.5,
      "risk_level": "low",
      "patterns": {"urgency": false, "authority_impersonation": false, "credential_request": false, "threat": false, "reward": false},
      "explanation": 123
    }
    '''
    with pytest.raises(NlpParseError, match="explanation must be a string"):
        parse_llm_response(raw)


def test_parse_invalid_json():
    """Invalid JSON raises NlpParseError."""
    raw = "{ not valid json }"
    with pytest.raises(NlpParseError, match="Invalid JSON"):
        parse_llm_response(raw)


def test_parse_no_json_object():
    """Response with no JSON object raises NlpParseError."""
    raw = "This is just text with no JSON"
    with pytest.raises(NlpParseError, match="No JSON object found in response"):
        parse_llm_response(raw)


def test_parse_case_insensitive_risk_level():
    """Risk level is case-insensitive."""
    raw = '''
    {
      "scam_probability": 0.5,
      "risk_level": "HIGH",
      "patterns": {"urgency": false, "authority_impersonation": false, "credential_request": false, "threat": false, "reward": false},
      "explanation": "test"
    }
    '''
    result = parse_llm_response(raw)
    assert result.risk_level == "high"


def test_parse_boolean_true_false():
    """Boolean values true/false are accepted."""
    raw = '''
    {
      "scam_probability": 0.5,
      "risk_level": "low",
      "patterns": {"urgency": true, "authority_impersonation": true, "credential_request": true, "threat": true, "reward": true},
      "explanation": "test"
    }
    '''
    result = parse_llm_response(raw)
    assert all(v is True for v in result.patterns.values())


def test_parse_integer_scam_probability():
    """Integer scam_probability is accepted and converted to float."""
    raw = '''
    {
      "scam_probability": 1,
      "risk_level": "critical",
      "patterns": {"urgency": true, "authority_impersonation": true, "credential_request": true, "threat": true, "reward": true},
      "explanation": "test"
    }
    '''
    result = parse_llm_response(raw)
    assert result.scam_probability == 1.0


def test_call_hf_inference_400_includes_body_no_token():
    """400 response includes truncated body in error, but not the Authorization token."""
    from unittest.mock import AsyncMock, patch, MagicMock

    # Mock the settings to have a token
    with patch("app.nlp.worker.HF_TOKEN", "test-secret-token-12345"), \
         patch("app.nlp.worker.HF_INFERENCE_BASE_URL", "https://test.api"), \
         patch("app.nlp.worker.HF_LLM_MODEL", "test-model"):

        # Mock httpx.AsyncClient.post to return a 400 response
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = '{"error": "invalid request", "details": "missing required field"}'
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        # post is an async method that returns the mock_response
        mock_client.post = AsyncMock(return_value=mock_response)

        # The __aenter__ should return the mock_client itself
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(RuntimeError) as exc:
                import asyncio
                asyncio.run(call_hf_inference([{"role": "user", "content": "test"}]))

            error_msg = str(exc.value)
            # Check that error includes the response body with new format "HTTP 400: ..."
            assert "HTTP 400:" in error_msg
            assert "invalid request" in error_msg
            assert "missing required field" in error_msg
            # Check that token is NOT in the error message
            assert "test-secret-token-12345" not in error_msg
            assert "Bearer" not in error_msg
            assert "Authorization" not in error_msg


async def test_run_nlp_analysis_stores_error_with_body_no_token():
    """run_nlp_analysis stores error with response body but no token when HF returns 400."""
    from unittest.mock import AsyncMock, patch, MagicMock

    # Mock the settings to have a token
    with patch("app.nlp.worker.HF_TOKEN", "test-secret-token-12345"), \
         patch("app.nlp.worker.HF_INFERENCE_BASE_URL", "https://test.api"), \
         patch("app.nlp.worker.HF_LLM_MODEL", "test-model"):

        # Mock httpx.AsyncClient.post to return a 400 response
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = '{"error": "invalid request", "details": "missing required field"}'
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            # Import here to avoid issues with patched settings
            from app.nlp.worker import run_nlp_analysis

            # Create a mock session factory that doesn't actually hit DB
            mock_session = AsyncMock()
            
            # Use a simple class instead of MagicMock to properly track attribute changes
            class MockEvent:
                def __init__(self):
                    self.id = 1
                    self.score = 0.0
                    self.status = "pending"
                    self.risk_level = None
                    self.patterns = None
                    self.error = None
                    self.raw_response = None
                    self.is_high_risk = False
                    self.threshold = 0.75
                    self.explanation = None

            mock_event = MockEvent()
            
            # Properly mock the async context manager
            mock_session.get = AsyncMock(return_value=mock_event)
            mock_session.commit = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_session_factory = lambda: mock_session

            await run_nlp_analysis(1, "test message", "+15551234567", session_factory=mock_session_factory)

            # Verify the error was stored with body but no token
            assert mock_event.status == "failed"
            assert mock_event.risk_level is None
            assert mock_event.patterns is None
            assert mock_event.score == 0.0
            error_text = mock_event.error
            assert "HTTP 400:" in error_text
            assert "invalid request" in error_text
            assert "missing required field" in error_text
            # Token must not appear in stored error
            assert "test-secret-token-12345" not in error_text
            assert "Bearer" not in error_text
            assert "Authorization" not in error_text


async def test_call_hf_inference_500_succeeds_on_retry():
    """500 response retries with backoff and succeeds on second attempt."""
    from unittest.mock import AsyncMock, patch, MagicMock

    with patch("app.nlp.worker.HF_TOKEN", "test-token"), \
         patch("app.nlp.worker.HF_INFERENCE_BASE_URL", "https://test.api"), \
         patch("app.nlp.worker.HF_LLM_MODEL", "test-model"), \
         patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:

        # First call returns 500, second succeeds
        mock_response_500 = MagicMock()
        mock_response_500.status_code = 500
        mock_response_500.text = "Internal Server Error"
        mock_response_500.raise_for_status = MagicMock()

        mock_response_ok = MagicMock()
        mock_response_ok.status_code = 200
        mock_response_ok.text = '{"choices": [{"message": {"content": "success"}}]}'
        mock_response_ok.raise_for_status = MagicMock()
        mock_response_ok.json = MagicMock(return_value={"choices": [{"message": {"content": "success"}}]})

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[mock_response_500, mock_response_ok])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await call_hf_inference([{"role": "user", "content": "test"}])

        assert result == "success"
        assert mock_client.post.call_count == 2
        mock_sleep.assert_called_once()


async def test_call_hf_inference_400_not_retried():
    """400 response fails immediately without retry."""
    from unittest.mock import AsyncMock, patch, MagicMock

    with patch("app.nlp.worker.HF_TOKEN", "test-token"), \
         patch("app.nlp.worker.HF_INFERENCE_BASE_URL", "https://test.api"), \
         patch("app.nlp.worker.HF_LLM_MODEL", "test-model"):

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = '{"error": "bad request"}'
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(RuntimeError) as exc:
                await call_hf_inference([{"role": "user", "content": "test"}])

        assert "HTTP 400:" in str(exc.value)
        assert mock_client.post.call_count == 1  # No retries


async def test_call_hf_inference_429_retry_after():
    """429 response uses Retry-After header (capped at 60s) for backoff."""
    from unittest.mock import AsyncMock, patch, MagicMock

    with patch("app.nlp.worker.HF_TOKEN", "test-token"), \
         patch("app.nlp.worker.HF_INFERENCE_BASE_URL", "https://test.api"), \
         patch("app.nlp.worker.HF_LLM_MODEL", "test-model"), \
         patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:

        # First call returns 429 with Retry-After: 5, second succeeds
        mock_response_429 = MagicMock()
        mock_response_429.status_code = 429
        mock_response_429.headers = {"Retry-After": "5"}
        mock_response_429.text = "Rate limited"
        mock_response_429.raise_for_status = MagicMock()

        mock_response_ok = MagicMock()
        mock_response_ok.status_code = 200
        mock_response_ok.text = '{"choices": [{"message": {"content": "success"}}]}'
        mock_response_ok.raise_for_status = MagicMock()
        mock_response_ok.json = MagicMock(return_value={"choices": [{"message": {"content": "success"}}]})

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[mock_response_429, mock_response_ok])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await call_hf_inference([{"role": "user", "content": "test"}])

        assert result == "success"
        mock_sleep.assert_called_once_with(5)  # Retry-After: 5 seconds


async def test_call_hf_inference_429_retry_after_capped():
    """429 response with Retry-After > 60 uses cap of 60s."""
    from unittest.mock import AsyncMock, patch, MagicMock

    with patch("app.nlp.worker.HF_TOKEN", "test-token"), \
         patch("app.nlp.worker.HF_INFERENCE_BASE_URL", "https://test.api"), \
         patch("app.nlp.worker.HF_LLM_MODEL", "test-model"), \
         patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:

        mock_response_429 = MagicMock()
        mock_response_429.status_code = 429
        mock_response_429.headers = {"Retry-After": "120"}  # > 60
        mock_response_429.text = "Rate limited"
        mock_response_429.raise_for_status = MagicMock()

        mock_response_ok = MagicMock()
        mock_response_ok.status_code = 200
        mock_response_ok.text = '{"choices": [{"message": {"content": "success"}}]}'
        mock_response_ok.raise_for_status = MagicMock()
        mock_response_ok.json = MagicMock(return_value={"choices": [{"message": {"content": "success"}}]})

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[mock_response_429, mock_response_ok])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            await call_hf_inference([{"role": "user", "content": "test"}])

        mock_sleep.assert_called_once_with(60)  # Capped at 60


async def test_call_hf_inference_body_truncated_to_500():
    """Error body longer than 500 chars is truncated in stored error."""
    from unittest.mock import AsyncMock, patch, MagicMock

    long_body = "x" * 600  # 600 chars

    with patch("app.nlp.worker.HF_TOKEN", "test-token"), \
         patch("app.nlp.worker.HF_INFERENCE_BASE_URL", "https://test.api"), \
         patch("app.nlp.worker.HF_LLM_MODEL", "test-model"):

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = long_body
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(RuntimeError) as exc:
                await call_hf_inference([{"role": "user", "content": "test"}])

        error_msg = str(exc.value)
        # Body should be truncated to 500 chars in the error message
        assert len(error_msg) <= 550  # "HTTP 400: " + 500 chars
        assert error_msg.endswith("x" * 500) or "x" * 500 in error_msg