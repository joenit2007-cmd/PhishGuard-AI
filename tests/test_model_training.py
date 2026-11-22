"""
Unit tests for Model Training and Evaluation modules.
"""

import os
import numpy as np
import pytest
from src.evaluation import evaluate_classifier
from src.model_training import (
    build_candidate_pipelines,
    build_feature_pipeline,
    extract_template_group,
)


class TestModelTrainingAndEvaluation:
    """Tests for model building, metric computation, and training pipeline."""

    def test_build_feature_pipeline_structure(self):
        union = build_feature_pipeline()
        transformer_names = [name for name, _ in union.transformer_list]
        assert "text" in transformer_names
        assert "structural" in transformer_names

    def test_build_candidate_pipelines(self):
        pipelines = build_candidate_pipelines()
        assert "Logistic Regression" in pipelines
        assert "Multinomial Naive Bayes" in pipelines
        assert "Random Forest Classifier" in pipelines

    def test_evaluate_classifier_metrics(self):
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 1, 0, 0, 0])
        y_proba = np.array([0.1, 0.2, 0.9, 0.85, 0.95, 0.15, 0.45, 0.05])

        metrics = evaluate_classifier(y_true, y_pred, y_proba, model_name="TestModel")

        assert metrics["model_name"] == "TestModel"
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["f1_score"] <= 1.0
        assert metrics["roc_auc"] is not None
        assert len(metrics["confusion_matrix"]) == 2
        assert "classification_report" in metrics

    def test_extract_template_group(self):
        subj1 = "URGENT: Your Chase Bank account has been temporarily restricted (Case #81004)"
        subj2 = "URGENT: Your Chase Bank account has been temporarily restricted (Case #81008)"
        subj3 = "Meeting Agenda: Finance Quarterly Strategic Alignment"

        group1 = extract_template_group(subj1)
        group2 = extract_template_group(subj2)
        group3 = extract_template_group(subj3)

        # Variations of same template must have identical group
        assert group1 == group2
        assert group1 != group3
        assert "#id" in group1

    def test_saved_artifacts_exist(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_root, "models", "phishing_detector_model.joblib")
        meta_path = os.path.join(project_root, "models", "model_metadata.json")
        metrics_path = os.path.join(project_root, "outputs", "model_metrics.json")

        assert os.path.exists(model_path), "Model artifact should exist"
        assert os.path.exists(meta_path), "Model metadata should exist"
        assert os.path.exists(metrics_path), "Metrics JSON should exist"
