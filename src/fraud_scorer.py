"""
Combines the text-suspicion signal and the behavioral-anomaly signal into
one fraud_score per restaurant.

Weighted rather than purely max/average: behavioral patterns are weighted
higher (0.6) than text (0.4), since the text model here is trained on
weak/synthetic labels and is the noisier of the two signals -- the README
explains this trade-off.
"""

from __future__ import annotations

import pandas as pd

from src.behavioral_features import compute_behavioral_scores
from src.text_features import score_texts

WEIGHTS = {"behavior": 0.6, "text": 0.4}


def compute_text_scores_per_restaurant(reviews_df: pd.DataFrame) -> pd.DataFrame:
    reviews_df = reviews_df.copy()
    reviews_df["text_suspicion"] = score_texts(reviews_df["text"].tolist())
    return (
        reviews_df.groupby("restaurant_id")["text_suspicion"]
        .mean()
        .reset_index()
        .rename(columns={"text_suspicion": "text_score"})
    )


def compute_fraud_scores(
    reviews_df: pd.DataFrame, reviewers_df: pd.DataFrame
) -> pd.DataFrame:
    behavioral = compute_behavioral_scores(reviews_df, reviewers_df)
    textual = compute_text_scores_per_restaurant(reviews_df)

    merged = behavioral.merge(textual, on="restaurant_id", how="left")
    merged["text_score"] = merged["text_score"].fillna(0.0)

    merged["fraud_score"] = (
        WEIGHTS["behavior"] * merged["behavior_score"]
        + WEIGHTS["text"] * merged["text_score"]
    )

    return merged.sort_values("fraud_score", ascending=False).reset_index(drop=True)
