"""
Command-line Execution Script for Phishing Email Detection Model Training.

Usage:
    python train_model.py
"""

import json
import logging
import os
import sys

# Ensure project root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from data.download_dataset import prepare_dataset
from src.model_training import train_and_compare_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TrainScript")


def main():
    print("=" * 76)
    print("  PHISHING EMAIL DETECTION SYSTEM - LEAK-FREE MODEL TRAINING PIPELINE")
    print("=" * 76)

    data_file = os.path.join(CURRENT_DIR, "data", "emails.csv")
    if not os.path.exists(data_file):
        logger.info("Dataset not detected. Generating curated benchmark dataset...")
        prepare_dataset()

    logger.info("Executing duplicate-aware, group-based stratified training...")
    best_name, _, all_metrics = train_and_compare_models(data_path=data_file)

    meta_file = os.path.join(CURRENT_DIR, "models", "model_metadata.json")
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    print("\n" + "=" * 76)
    print("  LEAK-FREE HOLDOUT EVALUATION BENCHMARK RESULTS")
    print("  Methodology: StratifiedGroupKFold (Zero Template Overlap)")
    print(f"  Holdout Test Split: {meta['test_records']} emails (Phishing: {meta['test_phishing_count']}, Safe: {meta['test_safe_count']})")
    print("=" * 76)
    print(f"{'Algorithm':<28} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 76)

    for name, m in all_metrics.items():
        is_best = " (Champion)" if name == best_name else ""
        print(
            f"{name + is_best:<28} | "
            f"{m['accuracy']:<10.4f} | "
            f"{m['precision']:<10.4f} | "
            f"{m['recall']:<10.4f} | "
            f"{m['f1_score']:<10.4f}"
        )

    print("\n" + "=" * 76)
    print("  5-FOLD STRATIFIED GROUP CROSS-VALIDATION SUMMARY (Mean +/- Std)")
    print("=" * 76)
    print(f"{'Algorithm':<28} | {'CV Accuracy':<15} | {'CV Precision':<15} | {'CV Recall':<15} | {'CV F1-Score':<15}")
    print("-" * 76)
    cv_data = meta.get("cross_validation_5fold", {})
    for name, cv in cv_data.items():
        print(
            f"{name:<28} | "
            f"{cv['mean_accuracy']:.4f} +/- {cv['std_accuracy']:.4f} | "
            f"{cv['mean_precision']:.4f} +/- {cv['std_precision']:.4f} | "
            f"{cv['mean_recall']:.4f} +/- {cv['std_recall']:.4f} | "
            f"{cv['mean_f1']:.4f} +/- {cv['std_f1']:.4f}"
        )

    print("=" * 76)
    print(f"Champion Model Selected : {best_name}")
    print(f"Artifact Saved At       : models/phishing_detector_model.joblib")
    print(f"Metadata Saved At       : models/model_metadata.json")
    print(f"Metrics Output Saved At : outputs/model_metrics.json")
    print(f"Confusion Matrix Plot   : outputs/confusion_matrix.png")
    print(f"Model Comparison Plot   : outputs/model_comparison.png")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()
