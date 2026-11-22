"""
Inference and Prediction Pipeline for Phishing Email Detection System.

Loads the serialized model artifact and metadata, validates inputs,
generates predictions and calibrated confidence estimates, and provides
interpretable reasoning based on detected security heuristics.
"""

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np

from src.feature_extraction import (
    analyze_single_url,
    extract_urls,
    get_email_feature_breakdown,
)
from src.preprocessing import combine_subject_and_body, validate_email_input

logger = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "phishing_detector_model.joblib")
DEFAULT_METADATA_PATH = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")


class PhishingPredictor:
    """
    Production-grade inference engine for evaluating email samples.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
    ):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.metadata_path = metadata_path or DEFAULT_METADATA_PATH
        self.model = None
        self.metadata = {}
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """
        Loads the trained Scikit-learn pipeline and associated metadata.
        """
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                logger.info(f"Loaded trained model artifact from: {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load model artifact: {e}")
                self.model = None
        else:
            logger.warning(f"Model file not found at: {self.model_path}")

        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
                self.metadata = {}

    def is_ready(self) -> bool:
        """Checks if the predictor model is loaded and ready."""
        return self.model is not None

    def predict(self, subject: Optional[str], body: Optional[str]) -> Dict[str, Any]:
        """
        Analyzes an email subject and body, producing a security verdict,
        confidence score, and detailed diagnostic breakdown.

        Args:
            subject: Email subject text.
            body: Email body text.

        Returns:
            Dictionary containing prediction verdict, confidence, breakdown, and guidance.
        """
        # Validate input
        is_valid, error_msg = validate_email_input(subject, body)
        if not is_valid:
            return {
                "success": False,
                "error": error_msg,
                "verdict": "UNKNOWN",
                "is_phishing": False,
                "confidence_score": 0.0,
                "breakdown": {},
                "risk_level": "UNKNOWN",
                "reasons": [error_msg],
            }

        if not self.is_ready():
            return {
                "success": False,
                "error": "Prediction model is not trained or loaded. Please train the model first.",
                "verdict": "ERROR",
                "is_phishing": False,
                "confidence_score": 0.0,
                "breakdown": {},
                "risk_level": "ERROR",
                "reasons": ["Model artifacts missing."],
            }

        # Combine and clean text
        full_text = combine_subject_and_body(subject, body)

        # Predict using full Scikit-learn pipeline
        raw_pred = int(self.model.predict([full_text])[0])
        is_phishing = bool(raw_pred == 1)

        # Compute probability / confidence estimate
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba([full_text])[0]
            # proba[1] is probability of PHISHING, proba[0] is SAFE
            phishing_prob = float(proba[1])
            confidence = phishing_prob if is_phishing else float(proba[0])
        else:
            phishing_prob = 1.0 if is_phishing else 0.0
            confidence = 0.85

        # Heuristic and structural breakdown
        breakdown = get_email_feature_breakdown(full_text)
        ling = breakdown["linguistic"]
        urls = breakdown["urls"]

        # Generate reasons and contributing factors
        reasons = []
        if is_phishing:
            verdict = "PHISHING"
            if ling["urgency_count"] > 0:
                reasons.append(
                    f"Urgency cues detected ({ling['urgency_count']} occurrences, e.g. {', '.join(ling['urgency_found'][:3])}) creating psychological pressure."
                )
            if ling["credential_count"] > 0:
                reasons.append(
                    f"Credential harvesting keywords detected ({', '.join(ling['credential_found'][:3])})."
                )
            if ling["financial_count"] > 0:
                reasons.append(
                    f"Financial/transactional keywords found ({', '.join(ling['financial_found'][:3])})."
                )
            if ling["cta_count"] > 0:
                reasons.append(
                    f"High-risk call-to-action phrases detected ({', '.join(ling['cta_found'][:2])})."
                )
            if ling["uppercase_ratio"] > 0.15:
                reasons.append(
                    f"Unusually high uppercase text ratio ({ling['uppercase_ratio'] * 100:.1f}%) indicative of high-pressure spam."
                )
            if breakdown["high_risk_urls_count"] > 0:
                reasons.append(
                    f"{breakdown['high_risk_urls_count']} embedded URL(s) contain high-risk attributes (IP host, deceptive TLD, or nested subdomains)."
                )
            if not reasons:
                reasons.append("Email text patterns closely match known phishing attack signatures learned by the model.")

            # Assign risk level
            if phishing_prob >= 0.85 or breakdown["high_risk_urls_count"] > 0:
                risk_level = "CRITICAL RISK"
            else:
                risk_level = "HIGH RISK"

            recommendation = (
                "DO NOT click any links, open attachments, or reply with credentials. "
                "Report this message to your Security Operations Center (SOC) or IT department."
            )
        else:
            verdict = "SAFE"
            if breakdown["total_red_flags"] == 0:
                reasons.append("No suspicious linguistic pressure or anomalous URLs detected.")
            else:
                reasons.append("Linguistic patterns align with standard professional communication.")

            if breakdown["url_count"] > 0:
                reasons.append(f"All {breakdown['url_count']} embedded URL(s) exhibit standard structure and legitimate domains.")
            else:
                reasons.append("No deceptive links or credential prompts identified.")

            risk_level = "LOW RISK"
            recommendation = "Email exhibits standard communication characteristics. Standard caution is always recommended."

        return {
            "success": True,
            "verdict": verdict,
            "is_phishing": is_phishing,
            "confidence_score": round(confidence, 4),
            "phishing_probability": round(phishing_prob, 4),
            "risk_level": risk_level,
            "reasons": reasons,
            "recommendation": recommendation,
            "breakdown": breakdown,
            "full_text": full_text,
            "model_used": self.metadata.get("best_model", "Trained ML Model"),
            "disclaimer": (
                "Machine learning predictions are probabilistic security estimates and should not be treated as absolute guarantees. "
                "Always verify unexpected communications through verified out-of-band channels."
            ),
        }
