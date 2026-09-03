"""
Text suspicion model: classifies review text as "templated" (generic,
superlative-heavy, low lexical diversity -- the kind of text a bulk
manipulation campaign tends to reuse) vs "natural."

This is trained on the synthetic dataset's known review sources (organic
template pool vs. fraud template pool) as a stand-in for real labeled data.
In production this would be trained on reviews confirmed fake/real via
moderation action, not template origin.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

MODEL_PATH = Path("data/text_model.joblib")


def build_training_labels(reviews_df: pd.DataFrame, restaurants_df: pd.DataFrame) -> pd.Series:
    """
    Weak-label reviews as templated (1) / natural (0) using which template
    pool they were drawn from originally. This works because the generator
    keeps the two pools textually distinct (superlative density, repetition)
    -- the same property a real templated-fraud-review pool would have.
    """
    manipulated_ids = set(restaurants_df[restaurants_df["is_manipulated"]]["restaurant_id"])
    # Only reviews inside a manipulated restaurant's burst have templated text;
    # approximate by exact match against the known fraud phrase set imported here
    # to keep this module self-contained for training purposes.
    from src.data_generator import TEMPLATED_FRAUD_PHRASES
    fraud_set = set(TEMPLATED_FRAUD_PHRASES)
    return reviews_df["text"].isin(fraud_set).astype(int)


def train_text_model(reviews_df: pd.DataFrame, restaurants_df: pd.DataFrame) -> dict:
    labels = build_training_labels(reviews_df, restaurants_df)

    X_train, X_test, y_train, y_test = train_test_split(
        reviews_df["text"], labels, test_size=0.2, random_state=42, stratify=labels
    )

    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train_vec, y_train)

    preds = clf.predict(X_test_vec)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds),
    }

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "classifier": clf}, MODEL_PATH)

    return metrics


def score_texts(texts: list[str]) -> list[float]:
    """Returns P(templated) per text, using the saved model."""
    bundle = joblib.load(MODEL_PATH)
    vec = bundle["vectorizer"].transform(texts)
    return bundle["classifier"].predict_proba(vec)[:, 1].tolist()


if __name__ == "__main__":
    reviews_df = pd.read_parquet("data/reviews.parquet")
    restaurants_df = pd.read_parquet("data/restaurants.parquet")
    metrics = train_text_model(reviews_df, restaurants_df)
    print(f"Text model -- accuracy: {metrics['accuracy']:.3f}, f1: {metrics['f1']:.3f}")
