"""
End-to-end evaluation: trains the text model, computes fraud scores for
every restaurant, and reports precision/recall/confusion matrix against
the known-injected ground truth labels.

Run:
    python src/evaluate.py
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)

from src.text_features import train_text_model
from src.fraud_scorer import compute_fraud_scores

THRESHOLD = 0.7


def main() -> None:
    restaurants_df = pd.read_parquet("data/restaurants.parquet")
    reviewers_df = pd.read_parquet("data/reviewers.parquet")
    reviews_df = pd.read_parquet("data/reviews.parquet")

    text_metrics = train_text_model(reviews_df, restaurants_df)
    print(f"Text model -- accuracy: {text_metrics['accuracy']:.3f}, f1: {text_metrics['f1']:.3f}\n")

    scores_df = compute_fraud_scores(reviews_df, reviewers_df)
    merged = scores_df.merge(restaurants_df[["restaurant_id", "is_manipulated"]], on="restaurant_id")

    y_true = merged["is_manipulated"].astype(int)
    y_pred = (merged["fraud_score"] >= THRESHOLD).astype(int)

    print(f"Restaurant-level fraud detection (threshold={THRESHOLD}):")
    print(classification_report(y_true, y_pred, target_names=["organic", "manipulated"]))

    print("Confusion matrix ([[TN, FP], [FN, TP]]):")
    print(confusion_matrix(y_true, y_pred))

    precision, recall, thresholds = precision_recall_curve(y_true, merged["fraud_score"])
    print("\nPrecision/recall at a few thresholds:")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        preds_t = (merged["fraud_score"] >= t).astype(int)
        tp = ((preds_t == 1) & (y_true == 1)).sum()
        fp = ((preds_t == 1) & (y_true == 0)).sum()
        fn = ((preds_t == 0) & (y_true == 1)).sum()
        p = tp / (tp + fp) if (tp + fp) else 0
        r = tp / (tp + fn) if (tp + fn) else 0
        print(f"  threshold={t:.1f}  precision={p:.3f}  recall={r:.3f}")


if __name__ == "__main__":
    main()
