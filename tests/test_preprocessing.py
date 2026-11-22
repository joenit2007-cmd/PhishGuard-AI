"""
Unit tests for Email Preprocessing module.
"""

import pytest
from src.preprocessing import (
    clean_html,
    combine_subject_and_body,
    normalize_text,
    normalize_whitespace,
    validate_email_input,
)


class TestEmailPreprocessing:
    """Tests for text cleaning, HTML stripping, and normalization."""

    def test_clean_html_basic_tags(self):
        raw = "<html><body><h1>Urgent Notice</h1><p>Please click here.</p></body></html>"
        cleaned = clean_html(raw)
        assert "<" not in cleaned
        assert ">" not in cleaned
        assert "Urgent Notice" in cleaned
        assert "Please click here." in cleaned

    def test_clean_html_script_and_style(self):
        raw = "<style>body {color: red;}</style><p>Hello</p><script>alert('xss');</script>"
        cleaned = clean_html(raw)
        assert "color: red" not in cleaned
        assert "alert('xss')" not in cleaned
        assert "Hello" in cleaned

    def test_clean_html_entities(self):
        raw = "AT&amp;T &quot;Security Update&quot; &lt;Action Required&gt; &#39;Verify&#39;"
        cleaned = clean_html(raw)
        assert "AT&T" in cleaned
        assert '"Security Update"' in cleaned
        assert "<Action Required>" in cleaned or "Action Required" in cleaned
        assert "'Verify'" in cleaned

    def test_clean_html_none_and_empty(self):
        assert clean_html(None) == ""
        assert clean_html("") == ""
        assert clean_html("   ") == "   "

    def test_normalize_whitespace(self):
        raw = "Hello   world!  \t  This   is    an\n\n\n\nemail."
        normalized = normalize_whitespace(raw)
        assert "   " not in normalized
        assert "\t" not in normalized
        assert "\n\n\n" not in normalized

    def test_normalize_text_lowercase(self):
        raw = "<b>URGENT ACTION</b> required!"
        norm_lower = normalize_text(raw, lowercase=True)
        assert norm_lower == "urgent action required!"

        norm_preserve = normalize_text(raw, lowercase=False)
        assert "URGENT" in norm_preserve

    def test_combine_subject_and_body(self):
        # Both present
        combined = combine_subject_and_body("Account Security", "Please review details.")
        assert "Subject: Account Security" in combined
        assert "Please review details." in combined

        # Only subject
        subj_only = combine_subject_and_body("Important Update", "")
        assert subj_only == "Subject: Important Update"

        # Only body
        body_only = combine_subject_and_body("", "Team meeting at 2 PM.")
        assert body_only == "Team meeting at 2 PM."

        # Neither
        assert combine_subject_and_body("", "") == ""
        assert combine_subject_and_body(None, None) == ""

    def test_validate_email_input(self):
        # Valid input
        is_val, msg = validate_email_input("Meeting", "Let's meet tomorrow.")
        assert is_val is True
        assert msg == ""

        # Empty inputs
        is_val, msg = validate_email_input("", "")
        assert is_val is False
        assert "empty" in msg.lower()

        # Whitespace only
        is_val, msg = validate_email_input("   ", "\n\t  ")
        assert is_val is False

        # Extremely brief
        is_val, msg = validate_email_input("Hi", "a")
        assert is_val is False
        assert "too brief" in msg.lower()
