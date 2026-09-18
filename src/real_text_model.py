"""
Real-data text-suspicion model.

Unlike ``text_features.py`` (which trains on the synthetic generator's
template pools as a stand-in), this module trains and evaluates the
text-suspicion classifier on a *real, gold-labeled* dataset: the
Deceptive Opinion Spam Corpus (Ott et al., 2011) -- 1,600 hotel reviews,
800 truthful (crawled from TripAdvisor/Expedia/etc.) and 800 deceptive
(crowd-written on Amazon Mechanical Turk to read as authentic).

Because the labels are human ground truth rather than injected, the
accuracy here reflects genuine deceptive-vs-truthful separability, not a
synthetic upper bound. See ``data/README.md`` for the citation and license.

Run:
    python -m src.real_text_model          # 5-fold CV report + saves the model
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

DATA_PATH = Path("data/deceptive-opinion.csv")
MODEL_PATH = Path("data/real_text_model.joblib")

# TF-IDF over word uni/bigrams; sublinear tf + English stopwords. These
# settings track the n-gram set-up reported in the original Ott et al. work.
VECTORIZER = TfidfVectorizer(
    ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english"
)


def load_corpus(path: Path = DATA_PATH) -> tuple[np.ndarray, np.ndarray]:
    """Return (texts, labels) with label 1 = deceptive, 0 = truthful."""
    df = pd.read_csv(path)
    texts = np.array(df["text"].astype(str).tolist(), dtype=object)
    labels = (df["deceptive"] == "deceptive").astype(int).to_numpy()
    return texts, labels


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", VECTORIZER),
            ("clf", LogisticRegression(max_iter=2000, C=4.0)),
        ]
    )


def evaluate(path: Path = DATA_PATH, n_splits: int = 5) -> dict:
    """Stratified k-fold cross-validation on the real corpus."""
    texts, labels = load_corpus(path)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    preds = cross_val_predict(build_pipeline(), texts, labels, cv=cv)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
    }
    print(f"Deceptive Opinion Spam Corpus -- {len(labels)} reviews "
          f"({int((labels == 0).sum())} truthful / {int((labels == 1).sum())} deceptive)")
    print(f"\nTF-IDF + Logistic Regression ({n_splits}-fold CV): "
          f"accuracy={metrics['accuracy']:.3f}  f1={metrics['f1']:.3f}\n")
    print(classification_report(
        labels, preds, target_names=["truthful", "deceptive"], digits=3
    ))
    return metrics


def train_and_save(path: Path = DATA_PATH) -> dict:
    """Fit on a held-out split, report test metrics, and persist the model."""
    texts, labels = load_corpus(path)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds),
    }


def score_reviews(texts: list[str]) -> list[float]:
    """Return P(deceptive) per review using the saved real-data model."""
    pipe = joblib.load(MODEL_PATH)
    return pipe.predict_proba(texts)[:, 1].tolist()


def top_indicative_terms(n: int = 15, path: Path = DATA_PATH) -> dict[str, list[str]]:
    """Words most predictive of each class, from the linear model weights.

    This is the model's explainability window: which n-grams push a review
    toward 'deceptive' vs 'truthful'.
    """
    texts, labels = load_corpus(path)
    pipe = build_pipeline().fit(texts, labels)
    vocab = np.asarray(pipe.named_steps["tfidf"].get_feature_names_out())
    coef = pipe.named_steps["clf"].coef_[0]          # label 1 = deceptive
    order = coef.argsort()
    return {
        "deceptive": vocab[order[-n:]][::-1].tolist(),
        "truthful": vocab[order[:n]].tolist(),
    }


def explain_review(text: str) -> dict:
    """Score one review and return the interpretable feature breakdown."""
    from src.linguistic_features import FEATURE_NAMES, extract

    prob = score_reviews([text])[0]
    return {
        "prob_deceptive": prob,
        "verdict": "deceptive" if prob >= 0.5 else "truthful",
        "linguistic": dict(zip(FEATURE_NAMES, extract(text))),
    }


if __name__ == "__main__":
    evaluate()
    train_and_save()
    terms = top_indicative_terms()
    print("\nMost 'deceptive' terms:", ", ".join(terms["deceptive"]))
    print("Most 'truthful'  terms:", ", ".join(terms["truthful"]))
    print(f"\nSaved trained model to {MODEL_PATH}")
