"""Tests for eval_sms.py - resume behaviour, failed-row handling, metric calculation."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add scripts to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from eval_sms import (
    EvalRow,
    load_csv,
    load_existing_results,
    save_row,
    compute_metrics,
    process_row,
)


# Sample CSV data for testing
SAMPLE_CSV = """id,text,sender,label,category,split,source,notes
1,"Your OTP is 123456","VM-SBI",legit,otp-bank-legit,dev,synthetic,
2,"Urgent: Update KYC at bit.ly/xyz","AD-FAKE",scam,otp-bank-scam,dev,synthetic,
3,"Order shipped track at amazon.in","JD-AMZN",legit,service-legit,test,synthetic,
4,"You won 10000! Claim at bit.ly/abc","+919876543210",scam,lottery-prize-scam,test,synthetic,
5,"Your balance is 5000","VM-HDFC",legit,otp-bank-legit,test,synthetic,
"""


def test_load_csv_filtering():
    """Test CSV loading with split filtering."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(SAMPLE_CSV)
        f.flush()
        path = Path(f.name)

    try:
        # Test dev split
        dev_rows = load_csv(path, "dev", None)
        assert len(dev_rows) == 2
        assert all(r[0].split == "dev" for r in dev_rows)

        # Test test split
        test_rows = load_csv(path, "test", None)
        assert len(test_rows) == 3
        assert all(r[0].split == "test" for r in test_rows)

        # Test all split
        all_rows = load_csv(path, "all", None)
        assert len(all_rows) == 5

        # Test limit
        limited = load_csv(path, "all", 2)
        assert len(limited) == 2

        # Check tuple structure (EvalRow, text)
        for row, text in all_rows:
            assert isinstance(row, EvalRow)
            assert isinstance(text, str)
    finally:
        path.unlink()


def test_load_existing_results():
    """Test loading completed rows from results CSV."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "label", "category", "split", "sender",
            "score", "risk_level", "patterns", "explanation",
            "status", "error", "total_tokens", "seconds"
        ])
        writer.writeheader()
        writer.writerow({
            "id": "1",
            "label": "legit",
            "category": "otp-bank-legit",
            "split": "dev",
            "sender": "VM-SBI",
            "score": "0.1",
            "risk_level": "low",
            "patterns": json.dumps({"urgency": False}),
            "explanation": "No scam indicators",
            "status": "completed",
            "error": "",
            "total_tokens": "100",
            "seconds": "1.5",
        })
        writer.writerow({
            "id": "2",
            "label": "scam",
            "category": "otp-bank-scam",
            "split": "dev",
            "sender": "AD-FAKE",
            "score": "0.9",
            "risk_level": "high",
            "patterns": json.dumps({"urgency": True}),
            "explanation": "Urgent KYC request",
            "status": "completed",
            "error": "",
            "total_tokens": "120",
            "seconds": "2.0",
        })
        # Failed row should NOT be loaded
        writer.writerow({
            "id": "3",
            "label": "legit",
            "category": "service-legit",
            "split": "test",
            "sender": "JD-AMZN",
            "score": "0.0",
            "risk_level": "",
            "patterns": "",
            "explanation": "Analysis failed: RuntimeError",
            "status": "failed",
            "error": "API error: Rate limited",
            "total_tokens": "0",
            "seconds": "30.0",
        })
        f.flush()
        path = Path(f.name)

    try:
        completed = load_existing_results(path)
        assert len(completed) == 2
        assert 1 in completed
        assert 2 in completed
        assert 3 not in completed  # Failed rows skipped

        assert completed[1].score == 0.1
        assert completed[1].risk_level == "low"
        assert completed[1].patterns == {"urgency": False}
        assert completed[1].total_tokens == 100
    finally:
        path.unlink()


def test_load_existing_results_keeps_last():
    """Test that when reloading results, the LAST row per id is kept (failed->completed transition)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "label", "category", "split", "sender",
            "score", "risk_level", "patterns", "explanation",
            "status", "error", "total_tokens", "seconds"
        ])
        writer.writeheader()
        # First row: failed
        writer.writerow({
            "id": "1",
            "label": "scam",
            "category": "otp-bank-scam",
            "split": "test",
            "sender": "AD-FAKE",
            "score": "0.0",
            "risk_level": "",
            "patterns": "",
            "explanation": "Failed",
            "status": "failed",
            "error": "API error",
            "total_tokens": "0",
            "seconds": "30.0",
        })
        # Second row: completed (should override)
        writer.writerow({
            "id": "1",
            "label": "scam",
            "category": "otp-bank-scam",
            "split": "test",
            "sender": "AD-FAKE",
            "score": "0.85",
            "risk_level": "high",
            "patterns": json.dumps({"urgency": True}),
            "explanation": "Urgent KYC",
            "status": "completed",
            "error": "",
            "total_tokens": "150",
            "seconds": "2.0",
        })
        f.flush()
        path = Path(f.name)

    try:
        completed = load_existing_results(path)
        assert len(completed) == 1
        assert 1 in completed
        assert completed[1].status == "completed"
        assert completed[1].score == 0.85
        assert completed[1].risk_level == "high"
        assert completed[1].total_tokens == 150
    finally:
        path.unlink()


def test_save_row():
    """Test saving a single row to CSV."""
    tmpdir = tempfile.mkdtemp()
    path = Path(tmpdir) / "test_results.csv"

    try:
        row = EvalRow(
            id=1,
            label="scam",
            category="otp-bank-scam",
            split="test",
            sender="AD-FAKE",
            score=0.85,
            risk_level="high",
            patterns={"urgency": True, "credential_request": True},
            explanation="Urgent KYC with credential request",
            status="completed",
            error=None,
            total_tokens=150,
            seconds=2.5,
        )

        fieldnames = [
            "id", "label", "category", "split", "sender",
            "score", "risk_level", "patterns", "explanation",
            "status", "error", "total_tokens", "seconds"
        ]

        save_row(path, row, fieldnames)

        # Read back
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            saved = list(reader)

        assert len(saved) == 1
        assert saved[0]["id"] == "1"
        assert saved[0]["score"] == "0.85"
        assert saved[0]["risk_level"] == "high"
        patterns = json.loads(saved[0]["patterns"])
        assert patterns["urgency"] is True
        assert saved[0]["total_tokens"] == "150"
    finally:
        path.unlink(missing_ok=True)
        Path(tmpdir).rmdir()


def test_compute_metrics_basic():
    """Test metric calculation on hand-checked example."""
    # 4 rows: 2 scam, 2 legit
    # Threshold 0.75
    # Row 1: legit, score 0.1 -> pred legit (TN)
    # Row 2: scam, score 0.9 -> pred scam (TP)
    # Row 3: legit, score 0.2 -> pred legit (TN)
    # Row 4: scam, score 0.8 -> pred scam (TP)
    rows = [
        EvalRow(id=1, label="legit", category="c1", split="test", sender="A", score=0.1, status="completed"),
        EvalRow(id=2, label="scam", category="c1", split="test", sender="B", score=0.9, status="completed"),
        EvalRow(id=3, label="legit", category="c2", split="test", sender="C", score=0.2, status="completed"),
        EvalRow(id=4, label="scam", category="c2", split="test", sender="D", score=0.8, status="completed"),
    ]

    metrics = compute_metrics(rows, 0.75)

    assert metrics["total_rows"] == 4
    assert metrics["completed"] == 4
    assert metrics["failed"] == 0
    assert metrics["confusion_matrix"]["tp"] == 2
    assert metrics["confusion_matrix"]["tn"] == 2
    assert metrics["confusion_matrix"]["fp"] == 0
    assert metrics["confusion_matrix"]["fn"] == 0
    assert metrics["recall"] == 1.0
    assert metrics["fpr"] == 0.0
    assert metrics["precision"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["mean_scam_score"] == 0.85
    assert metrics["mean_legit_score"] == 0.15


def test_compute_metrics_with_errors():
    """Test metrics with some failed rows and wrong predictions."""
    # 6 rows: 3 scam, 3 legit
    # Threshold 0.75
    rows = [
        # Correct predictions
        EvalRow(id=1, label="legit", category="c1", split="test", sender="A", score=0.1, status="completed"),  # TN
        EvalRow(id=2, label="scam", category="c1", split="test", sender="B", score=0.9, status="completed"),  # TP
        EvalRow(id=3, label="legit", category="c2", split="test", sender="C", score=0.2, status="completed"),  # TN
        # Wrong predictions
        EvalRow(id=4, label="scam", category="c2", split="test", sender="D", score=0.5, status="completed"),  # FN (score < 0.75)
        EvalRow(id=5, label="legit", category="c3", split="test", sender="E", score=0.9, status="completed"),  # FP (score >= 0.75)
        # Failed row
        EvalRow(id=6, label="scam", category="c3", split="test", sender="F", score=0.0, status="failed"),     # Excluded by default
    ]

    metrics = compute_metrics(rows, 0.75)

    # All rows: 6 total, 5 completed, 1 failed
    assert metrics["total_rows"] == 6
    assert metrics["completed"] == 5
    assert metrics["failed"] == 1

    # CM: TP=1, TN=2, FP=1, FN=1 (only completed rows)
    assert metrics["confusion_matrix"]["tp"] == 1
    assert metrics["confusion_matrix"]["tn"] == 2
    assert metrics["confusion_matrix"]["fp"] == 1
    assert metrics["confusion_matrix"]["fn"] == 1

    # Recall = 1/(1+1) = 0.5
    assert metrics["recall"] == 0.5
    # FPR = 1/(1+2) = 0.333...
    assert abs(metrics["fpr"] - 0.3333) < 0.001
    # Precision = 1/(1+1) = 0.5
    assert metrics["precision"] == 0.5
    # Accuracy = (1+2)/5 = 0.6
    assert metrics["accuracy"] == 0.6

    # Wrong predictions: ID 4 (FN), ID 5 (FP)
    assert len(metrics["wrong_predictions"]) == 2
    wrong_ids = {w["id"] for w in metrics["wrong_predictions"]}
    assert wrong_ids == {4, 5}


def test_compute_metrics_failed_as_miss():
    """Test failed_as_miss=True counts failed rows as misses."""
    rows = [
        EvalRow(id=1, label="legit", category="c1", split="test", sender="A", score=0.1, status="completed"),
        EvalRow(id=2, label="scam", category="c1", split="test", sender="B", score=0.9, status="completed"),
        EvalRow(id=3, label="scam", category="c2", split="test", sender="C", score=0.0, status="failed"),  # Counts as FN
    ]

    metrics_default = compute_metrics(rows, 0.75, failed_as_miss=False)
    metrics_miss = compute_metrics(rows, 0.75, failed_as_miss=True)

    # Default: total_rows = 3, completed = 2, failed = 1
    assert metrics_default["total_rows"] == 3
    assert metrics_default["completed"] == 2
    assert metrics_default["failed"] == 1
    # Only 2 completed rows used for CM: TP=1, TN=1
    assert metrics_default["confusion_matrix"]["tp"] == 1
    assert metrics_default["confusion_matrix"]["tn"] == 1
    assert metrics_default["recall"] == 1.0

    # Failed-as-miss: 3 rows, failed scam counts as FN
    assert metrics_miss["total_rows"] == 3
    assert metrics_miss["completed"] == 2
    assert metrics_miss["failed"] == 1
    assert metrics_miss["confusion_matrix"]["fn"] == 1  # The failed scam
    assert metrics_miss["recall"] == 0.5  # 1/(1+1)


def test_compute_metrics_failed_as_miss_failed_legit_is_fp():
    """Test failed_as_miss: failed legit row counts as FP."""
    rows = [
        EvalRow(id=1, label="legit", category="c1", split="test", sender="A", score=0.1, status="completed"),
        EvalRow(id=2, label="scam", category="c1", split="test", sender="B", score=0.9, status="completed"),
        EvalRow(id=3, label="legit", category="c2", split="test", sender="C", score=0.0, status="failed"),  # Counts as FP
    ]

    metrics_miss = compute_metrics(rows, 0.75, failed_as_miss=True)

    # 3 rows, failed legit counts as FP
    assert metrics_miss["confusion_matrix"]["fp"] == 1
    assert metrics_miss["confusion_matrix"]["fn"] == 0
    assert metrics_miss["recall"] == 1.0  # 1/(1+0)
    # Precision = 1/(1+1) = 0.5
    assert metrics_miss["precision"] == 0.5


def test_compute_metrics_failed_as_miss_app_rule():
    """Test failed_as_miss applies to app rule: failed scam=FN, failed legit=FP."""
    # Threshold 0.75
    rows = [
        EvalRow(id=1, label="legit", category="c1", split="test", sender="A", score=0.1, risk_level="low", status="completed"),      # TN (both rules)
        EvalRow(id=2, label="scam", category="c1", split="test", sender="B", score=0.9, risk_level="high", status="completed"),    # TP (both rules)
        EvalRow(id=3, label="scam", category="c2", split="test", sender="C", score=0.0, risk_level="low", status="failed"),       # failed scam -> FN
        EvalRow(id=4, label="legit", category="c2", split="test", sender="D", score=0.0, risk_level="low", status="failed"),      # failed legit -> FP
    ]

    metrics = compute_metrics(rows, 0.75, failed_as_miss=True)

    # Score-only rule:
    # Row 1: legit, score 0.1 < 0.75 -> pred legit (TN)
    # Row 2: scam, score 0.9 >= 0.75 -> pred scam (TP)
    # Row 3: failed scam -> pred legit (FN)
    # Row 4: failed legit -> pred scam (FP)
    assert metrics["confusion_matrix"]["tp"] == 1
    assert metrics["confusion_matrix"]["tn"] == 1
    assert metrics["confusion_matrix"]["fp"] == 1
    assert metrics["confusion_matrix"]["fn"] == 1

    # App rule:
    # Row 1: legit, risk_level="low", score 0.1 < 0.75 -> pred legit (TN)
    # Row 2: scam, risk_level="high" -> pred scam (TP)
    # Row 3: failed scam -> pred legit (FN)
    # Row 4: failed legit -> pred scam (FP)
    assert metrics["confusion_matrix_app"]["tp"] == 1
    assert metrics["confusion_matrix_app"]["tn"] == 1
    assert metrics["confusion_matrix_app"]["fp"] == 1
    assert metrics["confusion_matrix_app"]["fn"] == 1


def test_compute_metrics_error_rate_by_category():
    """Test error rate calculation per category."""
    rows = [
        EvalRow(id=1, label="legit", category="catA", split="test", sender="A", score=0.1, status="completed"),
        EvalRow(id=2, label="scam", category="catA", split="test", sender="B", score=0.0, status="failed"),
        EvalRow(id=3, label="legit", category="catB", split="test", sender="C", score=0.2, status="completed"),
        EvalRow(id=4, label="scam", category="catB", split="test", sender="D", score=0.9, status="completed"),
        EvalRow(id=5, label="legit", category="catB", split="test", sender="E", score=0.0, status="failed"),
    ]

    metrics = compute_metrics(rows, 0.75)

    # catA: 1 completed, 1 failed -> error rate 50%
    # catB: 2 completed, 1 failed -> error rate 33%
    assert metrics["error_rate_by_category"]["catA"] == 0.5
    assert abs(metrics["error_rate_by_category"]["catB"] - 0.3333) < 0.001


def test_compute_metrics_app_rule_different():
    """Test that app rule confusion matrix differs from score-only rule."""
    # Threshold 0.75
    # Row 1: legit, score 0.8, risk_level="high" -> score rule: FP, app rule: FP (high risk)
    # Row 2: scam, score 0.6, risk_level="high" -> score rule: FN, app rule: TP (high risk)
    # Row 3: legit, score 0.1, risk_level="low" -> score rule: TN, app rule: TN
    # Row 4: scam, score 0.9, risk_level="critical" -> score rule: TP, app rule: TP
    rows = [
        EvalRow(id=1, label="legit", category="c1", split="test", sender="A", score=0.8, risk_level="high", status="completed"),
        EvalRow(id=2, label="scam", category="c1", split="test", sender="B", score=0.6, risk_level="high", status="completed"),
        EvalRow(id=3, label="legit", category="c2", split="test", sender="C", score=0.1, risk_level="low", status="completed"),
        EvalRow(id=4, label="scam", category="c2", split="test", sender="D", score=0.9, risk_level="critical", status="completed"),
    ]

    metrics = compute_metrics(rows, 0.75)

    # Score-only: threshold=0.75
    # Row 1: legit, score 0.8 >= 0.75 -> pred scam (FP)
    # Row 2: scam, score 0.6 < 0.75 -> pred legit (FN)
    # Row 3: legit, score 0.1 < 0.75 -> pred legit (TN)
    # Row 4: scam, score 0.9 >= 0.75 -> pred scam (TP)
    assert metrics["confusion_matrix"]["tp"] == 1
    assert metrics["confusion_matrix"]["tn"] == 1
    assert metrics["confusion_matrix"]["fp"] == 1
    assert metrics["confusion_matrix"]["fn"] == 1

    # App rule: risk_level in (high, critical) OR score >= 0.75
    # Row 1: legit, risk_level="high" -> pred scam (FP)
    # Row 2: scam, risk_level="high" -> pred scam (TP)
    # Row 3: legit, risk_level="low", score 0.1 < 0.75 -> pred legit (TN)
    # Row 4: scam, risk_level="critical" -> pred scam (TP)
    assert metrics["confusion_matrix_app"]["tp"] == 2
    assert metrics["confusion_matrix_app"]["tn"] == 1
    assert metrics["confusion_matrix_app"]["fp"] == 1
    assert metrics["confusion_matrix_app"]["fn"] == 0

    # App rule has better recall (2/2 = 1.0 vs 1/2 = 0.5)
    assert metrics["recall_app"] == 1.0
    assert metrics["recall"] == 0.5


@pytest.mark.asyncio
async def test_process_row_success():
    """Test process_row with successful LLM response."""
    row = EvalRow(
        id=1,
        label="scam",
        category="otp-bank-scam",
        split="test",
        sender="AD-FAKE",
    )

    mock_content = json.dumps({
        "scam_probability": 0.85,
        "risk_level": "high",
        "patterns": {
            "urgency": True,
            "authority_impersonation": True,
            "credential_request": True,
            "threat": False,
            "reward": False,
        },
        "explanation": "Urgent KYC request with credential phishing",
    })
    mock_usage = {"total_tokens": 200, "prompt_tokens": 150, "completion_tokens": 50}

    with patch("eval_sms.call_llm_raw", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = (mock_content, mock_usage)

        processed = await process_row(row, "Urgent: Update KYC at bit.ly/xyz", "test-model", 0.75)

    assert processed.status == "completed"
    assert processed.score == 0.85
    assert processed.risk_level == "high"
    assert processed.patterns["urgency"] is True
    assert processed.total_tokens == 200
    assert processed.seconds >= 0


@pytest.mark.asyncio
async def test_process_row_parse_error():
    """Test process_row with parse error."""
    row = EvalRow(
        id=1,
        label="scam",
        category="otp-bank-scam",
        split="test",
        sender="AD-FAKE",
    )

    with patch("eval_sms.call_llm_raw", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = ("not valid json", {"total_tokens": 100})

        processed = await process_row(row, "Urgent: Update KYC", "test-model", 0.75)

    assert processed.status == "failed"
    assert "Parse error" in processed.error
    assert processed.score == 0.0
    assert processed.risk_level is None
    assert processed.patterns is None
    assert processed.total_tokens == 0


@pytest.mark.asyncio
async def test_process_row_api_error():
    """Test process_row with API error (RuntimeError)."""
    row = EvalRow(
        id=1,
        label="scam",
        category="otp-bank-scam",
        split="test",
        sender="AD-FAKE",
    )

    with patch("eval_sms.call_llm_raw", new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = RuntimeError("Rate limited after 2 retries")

        processed = await process_row(row, "Urgent: Update KYC", "test-model", 0.75)

    assert processed.status == "failed"
    assert "API error" in processed.error
    assert processed.score == 0.0
    assert processed.total_tokens == 0


@pytest.mark.asyncio
async def test_process_row_unexpected_error():
    """Test process_row with unexpected exception."""
    row = EvalRow(
        id=1,
        label="scam",
        category="otp-bank-scam",
        split="test",
        sender="AD-FAKE",
    )

    with patch("eval_sms.call_llm_raw", new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = ValueError("Unexpected")

        processed = await process_row(row, "Urgent: Update KYC", "test-model", 0.75)

    assert processed.status == "failed"
    assert "Unexpected error" in processed.error
    assert "ValueError" in processed.error


@pytest.mark.asyncio
async def test_call_llm_raw_model_param():
    """Test that call_llm_raw uses the model parameter in the payload."""
    from app.nlp.worker import call_llm_raw

    mock_content = json.dumps({
        "scam_probability": 0.5,
        "risk_level": "low",
        "patterns": {"urgency": False, "authority_impersonation": False, "credential_request": False, "threat": False, "reward": False},
        "explanation": "Test",
    })
    mock_usage = {"total_tokens": 100}

    # Capture the payload sent to the API
    captured_payload = {}

    async def mock_post(*args, **kwargs):
        captured_payload["json"] = kwargs.get("json", {})
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = json.dumps({"choices": [{"message": {"content": mock_content}}], "usage": mock_usage})
        mock_response.json = MagicMock(return_value={"choices": [{"message": {"content": mock_content}}], "usage": mock_usage})
        mock_response.raise_for_status = MagicMock()
        return mock_response

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=mock_post)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_client), \
         patch("app.nlp.worker.HF_TOKEN", "test-token"), \
         patch("app.nlp.worker.HF_INFERENCE_BASE_URL", "https://test.api"), \
         patch("app.nlp.worker.HF_LLM_MODEL", "default-model"):

        await call_llm_raw([{"role": "user", "content": "test"}], model="custom-model")

    assert captured_payload["json"]["model"] == "custom-model"

    # Test default model
    captured_payload.clear()
    with patch("httpx.AsyncClient", return_value=mock_client), \
         patch("app.nlp.worker.HF_TOKEN", "test-token"), \
         patch("app.nlp.worker.HF_INFERENCE_BASE_URL", "https://test.api"), \
         patch("app.nlp.worker.HF_LLM_MODEL", "default-model"):

        await call_llm_raw([{"role": "user", "content": "test"}])

    assert captured_payload["json"]["model"] == "default-model"


@pytest.mark.asyncio
async def test_call_llm_raw_null_content():
    """Test that null content in response gives parse error, not TypeError."""
    from app.nlp.worker import call_llm_raw

    # Response with null content
    mock_usage = {"total_tokens": 100, "prompt_tokens": 0, "completion_tokens": 0}
    mock_response_data = {
        "choices": [{"message": {"content": None}}],
        "usage": mock_usage
    }

    async def mock_post(*args, **kwargs):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(mock_response_data)
        mock_response.json = MagicMock(return_value=mock_response_data)
        mock_response.raise_for_status = MagicMock()
        return mock_response

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=mock_post)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_client), \
         patch("app.nlp.worker.HF_TOKEN", "test-token"), \
         patch("app.nlp.worker.HF_INFERENCE_BASE_URL", "https://test.api"), \
         patch("app.nlp.worker.HF_LLM_MODEL", "default-model"):

        content, usage = await call_llm_raw([{"role": "user", "content": "test"}])

    # content should be empty string, not None
    assert content == ""
    assert usage == mock_usage


@pytest.mark.asyncio
async def test_call_llm_raw_429_retry_passes_model():
    """Test that 429 retry passes model parameter."""
    from app.nlp.worker import call_llm_raw

    mock_content = json.dumps({
        "scam_probability": 0.5,
        "risk_level": "low",
        "patterns": {"urgency": False, "authority_impersonation": False, "credential_request": False, "threat": False, "reward": False},
        "explanation": "Test",
    })
    mock_usage = {"total_tokens": 100, "prompt_tokens": 50, "completion_tokens": 50}

    captured_payloads = []

    async def mock_post(*args, **kwargs):
        captured_payloads.append(kwargs.get("json", {}))
        if len(captured_payloads) == 1:
            # First call: 429
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_response.headers = {"Retry-After": "1"}
            mock_response.text = "Rate limited"
            mock_response.raise_for_status = MagicMock()
            return mock_response
        else:
            # Second call: 200
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = json.dumps({"choices": [{"message": {"content": mock_content}}], "usage": mock_usage})
            mock_response.json = MagicMock(return_value={"choices": [{"message": {"content": mock_content}}], "usage": mock_usage})
            mock_response.raise_for_status = MagicMock()
            return mock_response

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=mock_post)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_client), \
         patch("app.nlp.worker.HF_TOKEN", "test-token"), \
         patch("app.nlp.worker.HF_INFERENCE_BASE_URL", "https://test.api"), \
         patch("app.nlp.worker.HF_LLM_MODEL", "default-model"), \
         patch("asyncio.sleep", new_callable=AsyncMock):

        await call_llm_raw([{"role": "user", "content": "test"}], model="custom-model")

    assert len(captured_payloads) == 2
    assert captured_payloads[0]["model"] == "custom-model"
    assert captured_payloads[1]["model"] == "custom-model"


@pytest.mark.asyncio
async def test_call_llm_raw_5xx_retry_passes_model():
    """Test that 5xx retry passes model parameter."""
    from app.nlp.worker import call_llm_raw

    mock_content = json.dumps({
        "scam_probability": 0.5,
        "risk_level": "low",
        "patterns": {"urgency": False, "authority_impersonation": False, "credential_request": False, "threat": False, "reward": False},
        "explanation": "Test",
    })
    mock_usage = {"total_tokens": 100, "prompt_tokens": 50, "completion_tokens": 50}

    captured_payloads = []

    async def mock_post(*args, **kwargs):
        captured_payloads.append(kwargs.get("json", {}))
        if len(captured_payloads) == 1:
            # First call: 500
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_response.raise_for_status = MagicMock()
            return mock_response
        else:
            # Second call: 200
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = json.dumps({"choices": [{"message": {"content": mock_content}}], "usage": mock_usage})
            mock_response.json = MagicMock(return_value={"choices": [{"message": {"content": mock_content}}], "usage": mock_usage})
            mock_response.raise_for_status = MagicMock()
            return mock_response

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=mock_post)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_client), \
         patch("app.nlp.worker.HF_TOKEN", "test-token"), \
         patch("app.nlp.worker.HF_INFERENCE_BASE_URL", "https://test.api"), \
         patch("app.nlp.worker.HF_LLM_MODEL", "default-model"), \
         patch("asyncio.sleep", new_callable=AsyncMock):

        await call_llm_raw([{"role": "user", "content": "test"}], model="custom-model")

    assert len(captured_payloads) == 2
    assert captured_payloads[0]["model"] == "custom-model"
    assert captured_payloads[1]["model"] == "custom-model"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])