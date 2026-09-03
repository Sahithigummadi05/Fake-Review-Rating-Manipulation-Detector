"""
Tests for the behavioral anomaly detection logic using small synthetic
data, independent of the generated dataset or trained text model.
"""

from datetime import datetime, timedelta

import pandas as pd

from src.behavioral_features import (
    burst_score,
    uniformity_score,
    low_activity_account_score,
)


def _reviews(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def test_burst_score_flags_concentrated_reviews():
    base = datetime(2026, 1, 1)
    # 20 reviews on one day, 1 review each on 10 other spread-out days.
    burst_rows = [
        {"restaurant_id": 1, "account_id": i, "rating": 5.0,
         "timestamp": base + timedelta(hours=i)}
        for i in range(20)
    ]
    quiet_rows = [
        {"restaurant_id": 1, "account_id": 100 + i, "rating": 4.0,
         "timestamp": base + timedelta(days=10 * i)}
        for i in range(10)
    ]
    df = _reviews(burst_rows + quiet_rows)
    score = burst_score(df)
    assert score > 0.3  # clearly elevated vs a flat distribution


def test_burst_score_low_for_steady_reviews():
    base = datetime(2026, 1, 1)
    steady_rows = [
        {"restaurant_id": 1, "account_id": i, "rating": 4.0,
         "timestamp": base + timedelta(days=i)}
        for i in range(30)
    ]
    df = _reviews(steady_rows)
    score = burst_score(df)
    assert score < 0.3


def test_uniformity_score_flags_tight_high_ratings():
    base = datetime(2026, 1, 1)
    rows = [
        {"restaurant_id": 1, "account_id": i, "rating": 5.0,
         "timestamp": base + timedelta(days=i)}
        for i in range(10)
    ]
    df = _reviews(rows)
    score = uniformity_score(df)
    assert score > 0.8


def test_uniformity_score_low_for_natural_spread():
    base = datetime(2026, 1, 1)
    ratings = [3.5, 4.0, 4.5, 5.0, 3.0, 4.0, 4.5, 5.0, 3.5, 4.0]
    rows = [
        {"restaurant_id": 1, "account_id": i, "rating": r,
         "timestamp": base + timedelta(days=i)}
        for i, r in enumerate(ratings)
    ]
    df = _reviews(rows)
    score = uniformity_score(df)
    assert score < 0.5


def test_uniformity_score_not_flagged_when_mean_rating_is_low():
    """Tight ratings around a LOW mean (e.g. everyone agrees it's bad)
    shouldn't be flagged as manipulation -- only tight+high is suspicious."""
    base = datetime(2026, 1, 1)
    rows = [
        {"restaurant_id": 1, "account_id": i, "rating": 2.0,
         "timestamp": base + timedelta(days=i)}
        for i in range(10)
    ]
    df = _reviews(rows)
    score = uniformity_score(df)
    assert score == 0.0


def test_low_activity_account_score():
    reviews = _reviews([
        {"restaurant_id": 1, "account_id": 1, "rating": 5.0, "timestamp": datetime(2026, 1, 1)},
        {"restaurant_id": 1, "account_id": 2, "rating": 5.0, "timestamp": datetime(2026, 1, 1)},
        {"restaurant_id": 1, "account_id": 3, "rating": 5.0, "timestamp": datetime(2026, 1, 1)},
    ])
    reviewers = pd.DataFrame([
        {"account_id": 1, "account_age_days": 2},
        {"account_id": 2, "account_age_days": 5},
        {"account_id": 3, "account_age_days": 900},
    ])
    score = low_activity_account_score(reviews, reviewers, age_threshold_days=20)
    assert abs(score - (2 / 3)) < 1e-6
