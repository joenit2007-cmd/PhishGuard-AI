"""
Unit tests for the Prediction Engine.
"""

import pytest
from src.prediction import PhishingPredictor


@pytest.fixture(scope="module")
def predictor():
    pred = PhishingPredictor()
    assert pred.is_ready(), "Predictor model must be ready and loaded for tests"
    return pred


class TestPredictionEngine:
    """Tests for model inference, confidence scoring, and edge cases."""

    def test_phishing_email_prediction(self, predictor):
        subject = "URGENT: Chase Online Banking Access Blocked"
        body = (
            "Dear Customer, your Chase debit account is suspended due to unauthorized transactions. "
            "Please verify your password and SSN immediately: "
            "http://192.168.1.100/chase-auth/login.php"
        )
        result = predictor.predict(subject, body)

        assert result["success"] is True
        assert result["verdict"] == "PHISHING"
        assert result["is_phishing"] is True
        assert 0.0 <= result["confidence_score"] <= 1.0
        assert result["risk_level"] in ["CRITICAL RISK", "HIGH RISK"]
        assert len(result["reasons"]) > 0
        assert "recommendation" in result
        assert "breakdown" in result

    def test_legitimate_email_prediction(self, predictor):
        subject = "Sprint 42 Retrospective and Planning Agenda"
        body = (
            "Hi team, let's meet on Thursday at 10 AM to discuss sprint retro items. "
            "Please update your Jira tickets before the meeting. "
            "Meeting link: https://meet.google.com/abc-defg-hij"
        )
        result = predictor.predict(subject, body)

        assert result["success"] is True
        assert result["verdict"] == "SAFE"
        assert result["is_phishing"] is False
        assert 0.0 <= result["confidence_score"] <= 1.0
        assert result["risk_level"] == "LOW RISK"
        assert len(result["reasons"]) > 0

    def test_empty_input_handling(self, predictor):
        result = predictor.predict("", "")
        assert result["success"] is False
        assert "empty" in result["error"].lower()
        assert result["verdict"] == "UNKNOWN"

    def test_whitespace_input_handling(self, predictor):
        result = predictor.predict("   \t", "\n\n   ")
        assert result["success"] is False
        assert "empty" in result["error"].lower()

    def test_too_brief_input_handling(self, predictor):
        result = predictor.predict("Hi", "a")
        assert result["success"] is False
        assert "too brief" in result["error"].lower()

    def test_none_input_handling(self, predictor):
        result = predictor.predict(None, None)
        assert result["success"] is False

    def test_disclaimer_presence(self, predictor):
        result = predictor.predict("Hello Team", "Please review project docs.")
        assert "disclaimer" in result
        assert len(result["disclaimer"]) > 10
