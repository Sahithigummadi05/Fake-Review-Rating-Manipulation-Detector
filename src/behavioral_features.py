"""
Behavioral anomaly detection: looks at PATTERNS across a restaurant's
reviews over time, rather than any single review's text.

Three independent checks, each producing a 0-1 suspicion score:
  1. Review burst detection    -- abnormal spike in daily review volume
  2. Rating uniformity check   -- ratings suspiciously tight around 5 stars
  3. Low-activity account share -- fraction of reviews from very new/thin accounts

These are combined (max, not average) into a single behavioral score per
restaurant: a restaurant only needs to trip ONE strong signal to warrant
review, since averaging would let one clear signal get diluted by two
quiet ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def burst_score(restaurant_reviews: pd.DataFrame) -> float:
    """
    Bins reviews into daily counts and flags restaurants whose busiest day
    is a statistical outlier relative to their own typical daily volume
    (z-score based, so it adapts to each restaurant's normal scale instead
    of using one global threshold).
    """
    if len(restaurant_reviews) < 5:
        return 0.0

    daily_counts = (
        restaurant_reviews.set_index("timestamp").resample("D").size()
    )
    if daily_counts.std() == 0 or len(daily_counts) < 2:
        return 0.0

    z_scores = (daily_counts - daily_counts.mean()) / daily_counts.std()
    max_z = z_scores.max()
    # Squash to 0-1: z of 3+ is already a strong burst signal.
    return float(np.clip(max_z / 5.0, 0, 1))


def uniformity_score(restaurant_reviews: pd.DataFrame) -> float:
    """
    Flags restaurants whose rating distribution is suspiciously tight AND
    high (a real, organically-loved restaurant can still have some natural
    spread; a manipulated one is often near-identical 5s).
    """
    if len(restaurant_reviews) < 5:
        return 0.0

    mean_rating = restaurant_reviews["rating"].mean()
    std_rating = restaurant_reviews["rating"].std()

    if mean_rating < 4.5:
        return 0.0  # only suspicious when ratings are both high AND tight

    # Low std relative to a "natural" baseline (~0.8) is suspicious.
    tightness = np.clip((0.8 - std_rating) / 0.8, 0, 1)
    return float(tightness)


def low_activity_account_score(
    restaurant_reviews: pd.DataFrame, reviewers_df: pd.DataFrame, age_threshold_days: float = 20
) -> float:
    """Fraction of a restaurant's reviews coming from accounts younger than
    the threshold -- a cluster of brand-new accounts reviewing the same
    place is a classic manipulation signature."""
    if restaurant_reviews.empty:
        return 0.0

    merged = restaurant_reviews.merge(reviewers_df, on="account_id", how="left")
    young_share = (merged["account_age_days"] < age_threshold_days).mean()
    return float(young_share)


def compute_behavioral_scores(reviews_df: pd.DataFrame, reviewers_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for restaurant_id, group in reviews_df.groupby("restaurant_id"):
        b_score = burst_score(group)
        u_score = uniformity_score(group)
        a_score = low_activity_account_score(group, reviewers_df)

        combined = max(b_score, u_score, a_score)

        rows.append({
            "restaurant_id": restaurant_id,
            "burst_score": b_score,
            "uniformity_score": u_score,
            "low_activity_account_score": a_score,
            "behavior_score": combined,
        })

    return pd.DataFrame(rows)
