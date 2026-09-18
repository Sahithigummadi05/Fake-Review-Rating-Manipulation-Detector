"""
Feature ablation: does adding hand-crafted stylometric features to TF-IDF
actually improve deceptive-review detection on the Ott corpus?

Compares three representations under identical 5-fold cross-validation:
  1. TF-IDF n-grams only
  2. Linguistic/stylometric features only (src/linguistic_features.py)
  3. TF-IDF + linguistic combined

Run:
    python -m src.ablation
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler

from src.linguistic_features import LinguisticFeatures
from src.real_text_model import load_corpus


def _tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                           stop_words="english")


def _linguistic() -> Pipeline:
    return Pipeline([("feat", LinguisticFeatures()), ("scale", MaxAbsScaler())])


def _clf() -> LogisticRegression:
    return LogisticRegression(max_iter=2000, C=4.0)


def main() -> None:
    X, y = load_corpus()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    configs = {
        "TF-IDF only": Pipeline([("v", _tfidf()), ("clf", _clf())]),
        "Linguistic only": Pipeline([("v", _linguistic()), ("clf", _clf())]),
        "TF-IDF + Linguistic": Pipeline([
            ("u", FeatureUnion([("t", _tfidf()), ("l", _linguistic())])),
            ("clf", _clf()),
        ]),
    }

    print(f"{'Representation':22s}  {'Accuracy':>8s}  {'F1':>6s}")
    print("-" * 40)
    for name, pipe in configs.items():
        pred = cross_val_predict(pipe, X, y, cv=cv)
        print(f"{name:22s}  {accuracy_score(y, pred):>8.3f}  {f1_score(y, pred):>6.3f}")


if __name__ == "__main__":
    main()
