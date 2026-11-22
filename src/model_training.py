"""
Model Training and Comparison Module for Phishing Email Detection System.

Trains, evaluates, and compares:
1. Logistic Regression
2. Multinomial Naive Bayes
3. Random Forest Classifier

Uses duplicate-aware, group-based stratified splitting (StratifiedGroupKFold)
to prevent data leakage caused by near-duplicate template variations.
Fits preprocessing exclusively on training data, evaluates on leak-free test split,
and exports the champion model.
"""

import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler

from src.evaluation import (
    evaluate_classifier,
    save_confusion_matrix_plot,
    save_model_comparison_plot,
)
from src.feature_extraction import EmailStructuralFeatureExtractor

logger = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "emails.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")


def extract_template_group(subject: str) -> str:
    """
    Generates a normalized template fingerprint from an email subject line.
    Normalizes variable numbers, invoice IDs, currency amounts, and dates
    so that near-duplicate synthetic variations of the same template share
    the exact same group identifier.

    Args:
        subject: Raw email subject string.

    Returns:
        Canonical template group key.
    """
    if not subject or not isinstance(subject, str):
        return "empty_subject"

    norm = subject.strip()
    norm = re.sub(r"#\d+", "#ID", norm)
    norm = re.sub(r"\$\d+(\.\d+)?", "$AMT", norm)
    norm = re.sub(r"\b\d+\b", "NUM", norm)
    norm = re.sub(r"\s+", " ", norm).lower().strip()
    return norm


def build_feature_pipeline() -> FeatureUnion:
    """
    Constructs the unified feature union pipeline combining TF-IDF text features
    (unigrams & bigrams) and scaled structural heuristic features.

    Returns:
        FeatureUnion transformer.
    """
    text_pipe = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=2500,
                stop_words="english",
                sublinear_tf=True,
                token_pattern=r"(?u)\b[a-zA-Z0-9_\-\.]{2,}\b",
            ),
        )
    ])

    structural_pipe = Pipeline([
        ("extractor", EmailStructuralFeatureExtractor()),
        ("scaler", MaxAbsScaler()),  # Preserves sparsity and non-negativity (>=0)
    ])

    return FeatureUnion([
        ("text", text_pipe),
        ("structural", structural_pipe),
    ])


def build_candidate_pipelines() -> Dict[str, Pipeline]:
    """
    Returns dictionaries of full candidate Scikit-learn pipelines.

    Returns:
        Dict mapping algorithm name to pipeline.
    """
    pipelines = {
        "Logistic Regression": Pipeline([
            ("features", build_feature_pipeline()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    C=1.5,
                    class_weight="balanced",
                    random_state=42,
                    solver="liblinear",
                ),
            ),
        ]),
        "Multinomial Naive Bayes": Pipeline([
            ("features", build_feature_pipeline()),
            ("classifier", MultinomialNB(alpha=0.1, fit_prior=True)),
        ]),
        "Random Forest Classifier": Pipeline([
            ("features", build_feature_pipeline()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=120,
                    max_depth=25,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]),
    }
    return pipelines


def train_and_compare_models(
    data_path: str = DATA_FILE,
    n_splits: int = 5,
    holdout_fold_index: int = 3,
    random_state: int = 42,
) -> Tuple[str, Pipeline, Dict[str, Any]]:
    """
    Trains candidate algorithms using a duplicate-aware, group-based split
    (StratifiedGroupKFold) ensuring zero template leakage between training and testing.
    Also computes 5-fold cross-validation metrics across all folds.

    Args:
        data_path: Path to dataset CSV file.
        n_splits: Number of folds for group-based cross-validation.
        holdout_fold_index: Which fold to reserve as the dedicated holdout test set.
        random_state: Seed for reproducible grouping.

    Returns:
        Tuple of (best_model_name, best_pipeline, all_metrics_dict).
    """
    logger.info(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # Combine subject and body if full_text is not already present
    if "full_text" not in df.columns:
        df["subject"] = df["subject"].fillna("").astype(str)
        df["body"] = df["body"].fillna("").astype(str)
        df["full_text"] = "Subject: " + df["subject"] + "\n\n" + df["body"]

    # Compute template group fingerprints for duplicate-aware splitting
    df["template_group"] = df.apply(lambda r: extract_template_group(r["subject"]), axis=1)

    X = df["full_text"].values
    y = df["label"].values.astype(int)
    groups = df["template_group"].values

    total_groups = df["template_group"].nunique()
    logger.info(
        f"Dataset loaded: {len(X)} emails (Phishing: {sum(y == 1)}, Safe: {sum(y == 0)}). "
        f"Identified {total_groups} distinct template groups."
    )

    # Duplicate-Aware Stratified Group Split
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(sgkf.split(X, y, groups))

    train_idx, test_idx = folds[holdout_fold_index]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    overlap = train_groups.intersection(test_groups)

    logger.info(f"Holdout Split: Train={len(X_train)} emails, Test={len(X_test)} emails.")
    logger.info(f"Train labels: Safe={sum(y_train == 0)}, Phishing={sum(y_train == 1)}")
    logger.info(f"Test labels:  Safe={sum(y_test == 0)}, Phishing={sum(y_test == 1)}")
    logger.info(f"Group overlap between Train and Test: {len(overlap)} (Zero Leakage Verified)")

    assert len(overlap) == 0, "Data leakage detected: Template groups overlap between train and test splits!"

    # 1. Train and evaluate all models on the leak-free holdout split
    candidate_pipelines = build_candidate_pipelines()
    all_metrics: Dict[str, Dict[str, Any]] = {}
    fitted_pipelines: Dict[str, Pipeline] = {}

    for name, pipeline in candidate_pipelines.items():
        logger.info(f"Training {name} on holdout split...")
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        y_pred = pipeline.predict(X_test)
        y_proba = None
        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = evaluate_classifier(y_test, y_pred, y_proba, model_name=name)
        all_metrics[name] = metrics

        logger.info(
            f"Holdout Result for {name}: "
            f"Acc={metrics['accuracy']:.4f}, "
            f"Prec={metrics['precision']:.4f}, "
            f"Rec={metrics['recall']:.4f}, "
            f"F1={metrics['f1_score']:.4f}"
        )

    # 2. Compute 5-Fold Stratified Group Cross-Validation across the entire dataset
    logger.info("Computing 5-Fold Stratified Group Cross-Validation...")
    cv_summary: Dict[str, Dict[str, Any]] = {}

    for name in candidate_pipelines.keys():
        cv_metrics = {"accuracy": [], "precision": [], "recall": [], "f1_score": []}
        for fold_num, (f_train_idx, f_test_idx) in enumerate(folds):
            f_pipe = build_candidate_pipelines()[name]
            f_pipe.fit(X[f_train_idx], y[f_train_idx])
            f_pred = f_pipe.predict(X[f_test_idx])
            f_proba = f_pipe.predict_proba(X[f_test_idx])[:, 1] if hasattr(f_pipe, "predict_proba") else None
            m = evaluate_classifier(y[f_test_idx], f_pred, f_proba, model_name=name)
            for k in cv_metrics.keys():
                cv_metrics[k].append(m[k])

        cv_summary[name] = {
            "mean_accuracy": round(float(np.mean(cv_metrics["accuracy"])), 4),
            "std_accuracy": round(float(np.std(cv_metrics["accuracy"])), 4),
            "mean_precision": round(float(np.mean(cv_metrics["precision"])), 4),
            "std_precision": round(float(np.std(cv_metrics["precision"])), 4),
            "mean_recall": round(float(np.mean(cv_metrics["recall"])), 4),
            "std_recall": round(float(np.std(cv_metrics["recall"])), 4),
            "mean_f1": round(float(np.mean(cv_metrics["f1_score"])), 4),
            "std_f1": round(float(np.std(cv_metrics["f1_score"])), 4),
            "fold_f1_scores": [round(x, 4) for x in cv_metrics["f1_score"]],
        }
        logger.info(
            f"5-Fold CV for {name}: "
            f"Mean F1={cv_summary[name]['mean_f1']:.4f} (+/- {cv_summary[name]['std_f1']:.4f})"
        )

    # Select best model based on holdout F1-score (and mean CV F1)
    best_model_name = max(all_metrics, key=lambda k: (all_metrics[k]["f1_score"], cv_summary[k]["mean_f1"]))
    best_pipeline = fitted_pipelines[best_model_name]

    logger.info(f"Selected champion model: {best_model_name} (Holdout F1: {all_metrics[best_model_name]['f1_score']:.4f})")

    # Save artifacts
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    model_artifact_path = os.path.join(MODELS_DIR, "phishing_detector_model.joblib")
    joblib.dump(best_pipeline, model_artifact_path)
    logger.info(f"Saved champion model artifact to: {model_artifact_path}")

    # Save metadata with complete provenance and methodology details
    metadata = {
        "best_model": best_model_name,
        "trained_at": datetime.now().isoformat(),
        "total_records": len(df),
        "total_template_groups": int(total_groups),
        "split_methodology": "StratifiedGroupKFold (Duplicate-Aware, Zero Template Leakage)",
        "train_records": len(X_train),
        "test_records": len(X_test),
        "phishing_count_total": int(sum(y == 1)),
        "safe_count_total": int(sum(y == 0)),
        "train_phishing_count": int(sum(y_train == 1)),
        "train_safe_count": int(sum(y_train == 0)),
        "test_phishing_count": int(sum(y_test == 1)),
        "test_safe_count": int(sum(y_test == 0)),
        "template_group_overlap": 0,
        "all_metrics": all_metrics,
        "best_metrics": all_metrics[best_model_name],
        "cross_validation_5fold": cv_summary,
        "dataset_limitations": [
            "Dataset is internally curated (80 manual seed archetypes + 261 synthetic variations).",
            "Not sampled from historical email server archives; embedded URLs are mock domains.",
            "Group-based splitting prevents template memorization, but true production generalization requires external benchmarking."
        ]
    }

    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Save metrics output
    metrics_output_path = os.path.join(OUTPUTS_DIR, "model_metrics.json")
    with open(metrics_output_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    # Save charts
    cm_plot_path = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
    save_confusion_matrix_plot(
        all_metrics[best_model_name]["confusion_matrix"],
        best_model_name,
        cm_plot_path,
    )

    comparison_plot_path = os.path.join(OUTPUTS_DIR, "model_comparison.png")
    save_model_comparison_plot(all_metrics, comparison_plot_path)

    # Also sync charts to images/ directory
    images_dir = os.path.join(PROJECT_ROOT, "images")
    os.makedirs(images_dir, exist_ok=True)
    save_confusion_matrix_plot(
        all_metrics[best_model_name]["confusion_matrix"],
        best_model_name,
        os.path.join(images_dir, "confusion_matrix.png"),
    )
    save_model_comparison_plot(all_metrics, os.path.join(images_dir, "model_comparison.png"))

    logger.info("Evaluation charts successfully generated in outputs and images directories.")

    return best_model_name, best_pipeline, all_metrics
