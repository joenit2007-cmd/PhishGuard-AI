"""
Unit tests for Feature Extraction and Static URL Security Analysis.
"""

import numpy as np
import pytest
from src.feature_extraction import (
    FEATURE_NAMES,
    EmailStructuralFeatureExtractor,
    analyze_linguistic_features,
    analyze_single_url,
    extract_urls,
    get_email_feature_breakdown,
)


class TestFeatureExtraction:
    """Tests for URL heuristics, linguistic features, and custom transformer."""

    def test_extract_urls(self):
        text = "Check http://example.com/login and https://secure.portal.xyz/auth?user=test for details."
        urls = extract_urls(text)
        assert len(urls) == 2
        assert "http://example.com/login" in urls
        assert "https://secure.portal.xyz/auth?user=test" in urls

    def test_extract_urls_empty(self):
        assert extract_urls("No URLs here at all.") == []
        assert extract_urls("") == []
        assert extract_urls(None) == []

    def test_analyze_single_url_ip_host(self):
        analysis = analyze_single_url("http://192.168.1.100/login/bank.html")
        assert analysis["is_ip"] is True
        assert analysis["is_https"] is False
        assert "login" in analysis["suspicious_keywords_found"]
        assert analysis["risk_score"] >= 6
        assert analysis["risk_level"] == "CRITICAL RISK"

    def test_analyze_single_url_suspicious_tld(self):
        analysis = analyze_single_url("http://paypal-verification.security-portal.xyz/verify")
        assert analysis["has_suspicious_tld"] is True
        assert "verify" in analysis["suspicious_keywords_found"]
        assert analysis["risk_level"] in ["HIGH RISK", "CRITICAL RISK"]

    def test_analyze_single_url_at_symbol(self):
        analysis = analyze_single_url("http://google.com@phishingsite.ru/auth")
        assert analysis["has_at_symbol"] is True
        assert analysis["risk_score"] >= 4

    def test_analyze_single_url_clean(self):
        analysis = analyze_single_url("https://www.google.com/search")
        assert analysis["is_ip"] is False
        assert analysis["is_https"] is True
        assert analysis["has_suspicious_tld"] is False
        assert analysis["risk_level"] == "LOW RISK"

    def test_analyze_linguistic_features_urgency_and_credentials(self):
        text = "URGENT ACTION REQUIRED: Your password has expired! Verify your identity within 24 hours."
        ling = analyze_linguistic_features(text)
        assert ling["urgency_count"] >= 2
        assert "urgent" in ling["urgency_found"]
        assert "within 24 hours" in ling["urgency_found"]
        assert ling["credential_count"] >= 1
        assert "password" in ling["credential_found"]
        assert ling["uppercase_ratio"] > 0.10
        assert ling["exclamation_count"] >= 1

    def test_analyze_linguistic_features_clean_text(self):
        text = "Hi team, let's meet at 3pm for coffee and review the project documentation."
        ling = analyze_linguistic_features(text)
        assert ling["urgency_count"] == 0
        assert ling["credential_count"] == 0
        assert ling["financial_count"] == 0
        assert ling["cta_count"] == 0

    def test_structural_feature_extractor_transformer(self):
        extractor = EmailStructuralFeatureExtractor()
        samples = [
            "URGENT: Click http://192.168.1.5/bank to reset password immediately!",
            "Hi Sarah, can you review the attached pull request when you have time?",
            "",  # edge case: empty
        ]
        X_trans = extractor.fit_transform(samples)

        assert isinstance(X_trans, np.ndarray)
        assert X_trans.shape == (3, len(FEATURE_NAMES))
        # Ensure non-negativity for MultinomialNB compatibility
        assert np.all(X_trans >= 0.0)

        # First sample should have high urgency & IP flag
        assert X_trans[0, FEATURE_NAMES.index("url_has_ip")] == 1.0
        assert X_trans[0, FEATURE_NAMES.index("linguistic_urgency_count")] > 0
        assert X_trans[0, FEATURE_NAMES.index("linguistic_credential_count")] > 0

    def test_get_email_feature_breakdown(self):
        text = "Suspended account notice: Update credit card info at http://fraud.top/pay now."
        breakdown = get_email_feature_breakdown(text)
        assert "linguistic" in breakdown
        assert "urls" in breakdown
        assert breakdown["url_count"] == 1
        assert breakdown["total_red_flags"] > 0
