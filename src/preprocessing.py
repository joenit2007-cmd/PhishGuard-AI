"""
Email Preprocessing Module for Phishing Email Detection System.

Provides modular text cleaning, HTML stripping, entity decoding,
normalization, and input validation functions.
"""

import html
import re
from typing import Optional, Tuple


def clean_html(text: Optional[str]) -> str:
    """
    Strips HTML tags, styles, and script blocks from the given email text.

    Args:
        text: Raw email string potentially containing HTML markup.

    Returns:
        Cleaned plain text string.
    """
    if text is None or not isinstance(text, str):
        return ""

    # Remove script and style blocks entirely
    cleaned = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text, flags=re.DOTALL | re.IGNORECASE)

    # Preserve href URLs in anchor tags so embedded hyperlinks are not lost during HTML stripping
    cleaned = re.sub(r'<a\s+[^>]*href=["\'](https?://[^"\']+)["\'][^>]*>', r" \1 ", cleaned, flags=re.IGNORECASE)

    # Replace HTML break and paragraph tags with newlines for readability
    cleaned = re.sub(r"<(br|p|div|tr)[\s/>]", "\n", cleaned, flags=re.IGNORECASE)

    # Remove all remaining HTML tags
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)

    # Decode HTML entities (e.g., &amp; -> &, &nbsp; -> space)
    cleaned = html.unescape(cleaned)

    return cleaned


def normalize_whitespace(text: Optional[str]) -> str:
    """
    Normalizes excessive whitespace, carriage returns, and tabs.

    Args:
        text: Input string.

    Returns:
        String with single spaces between words and standardized linebreaks.
    """
    if text is None or not isinstance(text, str):
        return ""

    # Replace multiple horizontal whitespaces with a single space
    cleaned = re.sub(r"[ \t]+", " ", text)
    # Collapse multiple consecutive newlines into at most two
    cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)
    return cleaned.strip()


def normalize_text(text: Optional[str], lowercase: bool = False) -> str:
    """
    Normalizes text by cleaning HTML, decoding entities, and collapsing whitespaces.

    Args:
        text: Input text.
        lowercase: Whether to convert text to lowercase.

    Returns:
        Normalized text.
    """
    if text is None or not isinstance(text, str):
        return ""

    cleaned = clean_html(text)
    cleaned = normalize_whitespace(cleaned)

    if lowercase:
        cleaned = cleaned.lower()

    return cleaned


def combine_subject_and_body(subject: Optional[str], body: Optional[str]) -> str:
    """
    Combines email subject and body into a structured, unified text representation.

    Args:
        subject: Email subject line.
        body: Email body content.

    Returns:
        Combined normalized email text string.
    """
    clean_subj = normalize_text(subject, lowercase=False)
    clean_body = normalize_text(body, lowercase=False)

    if clean_subj and clean_body:
        return f"Subject: {clean_subj}\n\n{clean_body}"
    elif clean_subj:
        return f"Subject: {clean_subj}"
    elif clean_body:
        return clean_body
    else:
        return ""


def validate_email_input(subject: Optional[str], body: Optional[str]) -> Tuple[bool, str]:
    """
    Validates user input to ensure at least some meaningful content is provided.

    Args:
        subject: Email subject text.
        body: Email body text.

    Returns:
        Tuple of (is_valid: bool, error_message: str).
    """
    clean_subj = clean_html(subject).strip() if subject else ""
    clean_body = clean_html(body).strip() if body else ""

    if not clean_subj and not clean_body:
        return False, "Both email subject and body are empty. Please enter email content to analyze."

    total_chars = len(clean_subj) + len(clean_body)
    if total_chars < 5:
        return False, "Input is too brief for reliable security analysis. Please provide a more complete email sample."

    return True, ""
