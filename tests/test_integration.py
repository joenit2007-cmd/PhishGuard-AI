"""
Integration tests for End-to-End Phishing Detection Pipeline.
"""

import pytest
from src.prediction import PhishingPredictor
from src.preprocessing import clean_html, combine_subject_and_body
from src.feature_extraction import extract_urls, analyze_single_url


@pytest.fixture(scope="module")
def predictor():
    return PhishingPredictor()


class TestEndToEndPipeline:
    """Tests the full integrated pipeline from raw email to final verdict."""

    def test_full_pipeline_html_phishing_email(self, predictor):
        raw_html_body = """
        <html>
            <body>
                <h2 style="color: red;">FINAL NOTICE: ACCOUNT SUSPENDED</h2>
                <p>Dear customer, your credit card transaction of <b>$899.00</b> failed.</p>
                <p>You must confirm your billing credentials within 24 hours.</p>
                <a href="http://appleid-security-auth.apple-portal.xyz/login.php">Click here to restore account</a>
            </body>
        </html>
        """
        raw_subject = "Action Required: Suspicious Account Login"

        result = predictor.predict(raw_subject, raw_html_body)

        assert result["success"] is True
        assert result["verdict"] == "PHISHING"
        assert result["phishing_probability"] > 0.70
        assert "within 24 hours" in result["breakdown"]["linguistic"]["urgency_found"]
        assert len(result["breakdown"]["urls"]) >= 1

    def test_full_pipeline_html_legitimate_email(self, predictor):
        raw_html_body = """
        <html>
            <body>
                <h2>Your Amazon Order Has Shipped</h2>
                <p>Hello Alex, your order #112-98214 has been dispatched via UPS.</p>
                <p>Track your delivery online at <a href="https://www.amazon.com/orders">Amazon Orders</a>.</p>
            </body>
        </html>
        """
        raw_subject = "Amazon.com: Your package is on the way"

        result = predictor.predict(raw_subject, raw_html_body)

        assert result["success"] is True
        assert result["verdict"] == "SAFE"
        assert result["is_phishing"] is False
        assert result["risk_level"] == "LOW RISK"

    def test_pipeline_static_url_security_flagging(self, predictor):
        # Email with an IP address URL and high urgency
        subject = "Urgent: Direct Deposit Verification Required"
        body = "Please update your bank routing number immediately at http://192.168.1.200/payroll/login."

        result = predictor.predict(subject, body)
        assert result["verdict"] == "PHISHING"
        url_findings = result["breakdown"]["urls"][0]
        assert url_findings["is_ip"] is True
        assert url_findings["risk_level"] == "CRITICAL RISK"
