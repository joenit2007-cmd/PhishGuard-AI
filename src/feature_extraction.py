"""
Feature Extraction Module for Phishing Email Detection System.

Implements extraction of:
1. URL-based security features (IP host, suspicious TLDs, subdomains, @ presence, HTTPS indicator)
2. Linguistic features (urgency, credential, financial, CTA, uppercase ratio)
3. A Scikit-learn compatible Transformer for pipeline integration
4. Static heuristic URL risk analysis (strictly non-network, safe analysis)
"""

import math
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# -------------------------------------------------------------
# Heuristic Lists & Regular Expressions
# -------------------------------------------------------------

URL_REGEX = re.compile(
    r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
)

IPV4_REGEX = re.compile(r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b")

SUSPICIOUS_TLDS = {
    ".xyz", ".top", ".club", ".work", ".ru", ".tk", ".ml", ".ga",
    ".cf", ".gq", ".cn", ".cc", ".info", ".buzz", ".icu", ".rest",
    ".fit", ".link", ".live", ".shop", ".site", ".country", ".stream"
}

SUSPICIOUS_URL_KEYWORDS = [
    "login", "signin", "verify", "account", "banking", "secure",
    "update", "password", "auth", "confirm", "ebayisapi", "paypal",
    "wp-content", "admin", "recover", "restore", "wallet", "support"
]

URGENCY_KEYWORDS = [
    "urgent", "immediately", "immediate", "within 24 hours", "within 48 hours",
    "suspended", "suspension", "action required", "deadline", "compromised",
    "unauthorized", "act now", "critical", "expire", "expires", "expiring",
    "terminate", "terminated", "freeze", "frozen", "locked", "alert",
    "attention", "final notice", "restricted", "time-sensitive"
]

CREDENTIAL_KEYWORDS = [
    "password", "passcode", "credentials", "credential", "ssn",
    "social security", "pin", "username", "login", "log in", "sign in",
    "signin", "reset your password", "verify your identity", "secret phrase",
    "seed phrase", "authentication", "2fa", "two-factor", "security code"
]

FINANCIAL_KEYWORDS = [
    "bank", "banking", "invoice", "wire", "wire transfer", "payment",
    "credit card", "debit card", "bitcoin", "crypto", "btc", "eth",
    "refund", "billing", "statement", "overdue", "charge", "charged",
    "wallet", "payout", "direct deposit", "settlement", "tax refund"
]

CTA_PHRASES = [
    "click here", "click below", "follow this link", "follow the link",
    "update your account", "confirm your identity", "open the attached",
    "log in now", "verify now", "click the link", "sign in to", "claim your",
    "claim now", "dispute transaction", "unlock your account"
]


# -------------------------------------------------------------
# Static URL Extraction & Analysis
# -------------------------------------------------------------

def extract_urls(text: Optional[str]) -> List[str]:
    """
    Extracts all embedded URLs from text safely without accessing the network.

    Args:
        text: Email text string.

    Returns:
        List of extracted URL strings.
    """
    if not text or not isinstance(text, str):
        return []

    matches = URL_REGEX.findall(text)
    urls = []
    for match in matches:
        url = match[0] if isinstance(match, tuple) else match
        url = url.strip().rstrip(".,;:!?)'\"")
        if url and url not in urls:
            urls.append(url)

    # Also capture href URLs if text contains HTML anchor tags
    href_matches = re.findall(r'href=["\'](https?://[^"\']+)["\']', text, flags=re.IGNORECASE)
    for h_url in href_matches:
        h_url = h_url.strip().rstrip(".,;:!?)'\"")
        if h_url and h_url not in urls:
            urls.append(h_url)

    return urls


def analyze_single_url(url: str) -> Dict[str, Any]:
    """
    Performs static, non-network heuristic inspection on a single URL.
    Never initiates any HTTP request or DNS resolution.

    Args:
        url: Raw URL string.

    Returns:
        Dictionary of static security flags and risk assessment.
    """
    findings: List[str] = []
    risk_score = 0  # 0 to 10 scale

    # Normalize url format for parsing
    parse_target = url if (url.startswith("http://") or url.startswith("https://")) else "http://" + url

    try:
        parsed = urlparse(parse_target)
        hostname = (parsed.hostname or "").lower()
        path = (parsed.path or "").lower()
        query = (parsed.query or "").lower()
    except Exception:
        hostname = ""
        path = ""
        query = ""

    # Check for IP-based host
    is_ip = bool(IPV4_REGEX.search(hostname))
    if is_ip:
        risk_score += 4
        findings.append("Host is a raw numerical IP address (avoids domain reputation checks)")

    # Check for suspicious TLD
    has_suspicious_tld = False
    for tld in SUSPICIOUS_TLDS:
        if hostname.endswith(tld):
            has_suspicious_tld = True
            risk_score += 3
            findings.append(f"High-risk top-level domain detected ({tld})")
            break

    # Check excessive subdomains
    domain_parts = [p for p in hostname.split(".") if p]
    subdomain_count = max(0, len(domain_parts) - 2)
    if subdomain_count >= 2:
        risk_score += 2
        findings.append(f"Excessive subdomains count ({subdomain_count}) - potential domain spoofing")

    # Check for @ symbol in URL (credentials trick in URLs)
    has_at_symbol = "@" in url
    if has_at_symbol:
        risk_score += 4
        findings.append("Contains '@' symbol: URL masquerades destination before the '@'")

    # Check protocol
    is_https = url.lower().startswith("https://")
    if not is_https:
        risk_score += 1
        findings.append("Unencrypted connection (HTTP instead of HTTPS)")

    # Check suspicious keywords in path or hostname
    found_keywords = []
    combined_url_lower = url.lower()
    for kw in SUSPICIOUS_URL_KEYWORDS:
        if kw in combined_url_lower:
            found_keywords.append(kw)

    if found_keywords:
        risk_score += min(3, len(found_keywords))
        findings.append(f"Deceptive security/login keywords in URL: {', '.join(found_keywords[:3])}")

    # Check URL length
    url_length = len(url)
    if url_length > 80:
        risk_score += 1
        findings.append(f"Abnormally long URL ({url_length} characters)")

    # Determine risk level
    if risk_score >= 6:
        risk_level = "CRITICAL RISK"
    elif risk_score >= 4:
        risk_level = "HIGH RISK"
    elif risk_score >= 2:
        risk_level = "MEDIUM RISK"
    else:
        risk_level = "LOW RISK"

    return {
        "url": url,
        "hostname": hostname,
        "is_ip": is_ip,
        "has_suspicious_tld": has_suspicious_tld,
        "subdomain_count": subdomain_count,
        "has_at_symbol": has_at_symbol,
        "is_https": is_https,
        "url_length": url_length,
        "suspicious_keywords_found": found_keywords,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "findings": findings if findings else ["Standard structure; no anomalous flags observed."]
    }


# -------------------------------------------------------------
# Linguistic & Text Feature Analysis
# -------------------------------------------------------------

def analyze_linguistic_features(text: Optional[str]) -> Dict[str, Any]:
    """
    Computes linguistic security metrics for an email text.

    Args:
        text: Plain text of email.

    Returns:
        Dictionary of detected keywords, counts, and ratios.
    """
    if not text or not isinstance(text, str):
        return {
            "urgency_count": 0,
            "urgency_found": [],
            "credential_count": 0,
            "credential_found": [],
            "financial_count": 0,
            "financial_found": [],
            "cta_count": 0,
            "cta_found": [],
            "uppercase_ratio": 0.0,
            "exclamation_count": 0,
            "char_count": 0,
            "word_count": 0,
        }

    text_lower = text.lower()

    # Urgency keywords
    urgency_found = [kw for kw in URGENCY_KEYWORDS if kw in text_lower]
    urgency_count = sum(text_lower.count(kw) for kw in urgency_found)

    # Credential keywords
    credential_found = [kw for kw in CREDENTIAL_KEYWORDS if kw in text_lower]
    credential_count = sum(text_lower.count(kw) for kw in credential_found)

    # Financial keywords
    financial_found = [kw for kw in FINANCIAL_KEYWORDS if kw in text_lower]
    financial_count = sum(text_lower.count(kw) for kw in financial_found)

    # Call to action phrases
    cta_found = [phrase for phrase in CTA_PHRASES if phrase in text_lower]
    cta_count = sum(text_lower.count(phrase) for phrase in cta_found)

    # Uppercase ratio
    alpha_chars = [c for c in text if c.isalpha()]
    upper_chars = [c for c in alpha_chars if c.isupper()]
    uppercase_ratio = (len(upper_chars) / len(alpha_chars)) if alpha_chars else 0.0

    # Exclamation count
    exclamation_count = text.count("!")

    return {
        "urgency_count": urgency_count,
        "urgency_found": urgency_found,
        "credential_count": credential_count,
        "credential_found": credential_found,
        "financial_count": financial_count,
        "financial_found": financial_found,
        "cta_count": cta_count,
        "cta_found": cta_found,
        "uppercase_ratio": round(uppercase_ratio, 4),
        "exclamation_count": exclamation_count,
        "char_count": len(text),
        "word_count": len(text.split()),
    }


def get_email_feature_breakdown(text: Optional[str]) -> Dict[str, Any]:
    """
    Comprehensive analysis combining linguistic indicators and URL heuristics.
    Ideal for dashboard display and model explanation.

    Args:
        text: Email text.

    Returns:
        Structured dictionary of all security indicators.
    """
    ling = analyze_linguistic_features(text)
    urls = extract_urls(text)
    url_analyses = [analyze_single_url(u) for u in urls]

    high_risk_urls_count = sum(1 for a in url_analyses if a["risk_score"] >= 4)

    return {
        "linguistic": ling,
        "urls": url_analyses,
        "url_count": len(urls),
        "high_risk_urls_count": high_risk_urls_count,
        "total_red_flags": (
            ling["urgency_count"] +
            ling["credential_count"] +
            ling["financial_count"] +
            ling["cta_count"] +
            (2 if ling["uppercase_ratio"] > 0.15 else 0) +
            high_risk_urls_count * 2
        ),
    }


# -------------------------------------------------------------
# Scikit-learn Transformer for Model Pipeline Integration
# -------------------------------------------------------------

FEATURE_NAMES = [
    "url_count",
    "url_avg_length",
    "url_has_ip",
    "url_has_suspicious_tld",
    "url_excessive_subdomains",
    "url_has_at_symbol",
    "url_http_only_ratio",
    "url_suspicious_structure_count",
    "linguistic_urgency_count",
    "linguistic_credential_count",
    "linguistic_financial_count",
    "linguistic_cta_count",
    "linguistic_uppercase_ratio",
    "linguistic_exclamation_count",
    "text_log_char_length",
]


class EmailStructuralFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-learn Transformer that converts raw email strings into
    a dense 2D NumPy array of non-negative engineered structural features.
    Fully compatible with FeatureUnion and MultinomialNB.
    """

    def __init__(self):
        self.feature_names_ = FEATURE_NAMES

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transforms a collection of email text strings into a 2D float array.

        Args:
            X: Iterable of string email texts.

        Returns:
            np.ndarray of shape (n_samples, len(FEATURE_NAMES)).
        """
        features_list = []

        for item in X:
            text = str(item) if item is not None else ""
            ling = analyze_linguistic_features(text)
            urls = extract_urls(text)

            url_count = len(urls)
            if url_count > 0:
                url_analyses = [analyze_single_url(u) for u in urls]
                avg_len = sum(a["url_length"] for a in url_analyses) / url_count
                has_ip = 1.0 if any(a["is_ip"] for a in url_analyses) else 0.0
                has_tld = 1.0 if any(a["has_suspicious_tld"] for a in url_analyses) else 0.0
                excess_sub = 1.0 if any(a["subdomain_count"] >= 2 for a in url_analyses) else 0.0
                has_at = 1.0 if any(a["has_at_symbol"] for a in url_analyses) else 0.0
                http_only_ratio = sum(1.0 for a in url_analyses if not a["is_https"]) / url_count
                suspicious_struct = sum(len(a["suspicious_keywords_found"]) for a in url_analyses)
            else:
                avg_len = 0.0
                has_ip = 0.0
                has_tld = 0.0
                excess_sub = 0.0
                has_at = 0.0
                http_only_ratio = 0.0
                suspicious_struct = 0.0

            log_length = math.log1p(len(text))

            row = [
                float(url_count),
                float(avg_len),
                has_ip,
                has_tld,
                excess_sub,
                has_at,
                http_only_ratio,
                float(suspicious_struct),
                float(ling["urgency_count"]),
                float(ling["credential_count"]),
                float(ling["financial_count"]),
                float(ling["cta_count"]),
                float(ling["uppercase_ratio"]),
                float(ling["exclamation_count"]),
                float(log_length),
            ]
            features_list.append(row)

        return np.array(features_list, dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)
