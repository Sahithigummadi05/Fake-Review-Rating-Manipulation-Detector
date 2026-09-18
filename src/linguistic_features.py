"""
Interpretable stylometric features for deceptive-review detection.

These are hand-crafted, human-readable signals — not a black box — motivated
by findings in the deception-detection literature (Ott et al., 2011):
deceptive reviews tend to over-use superlatives and first-person pronouns,
lean on exclamation/emphasis, and carry less lexical variety than genuine
ones. Used alongside TF-IDF in ``real_text_model.py`` (see the ablation in
the README) and exposed on their own so a prediction can be explained, not
just scored.
"""

from __future__ import annotations

import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

FIRST_PERSON = {
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves",
}
SUPERLATIVES = {
    "best", "worst", "amazing", "awesome", "perfect", "incredible", "fantastic",
    "wonderful", "excellent", "terrible", "horrible", "unbelievable", "great",
    "love", "loved", "definitely", "absolutely", "highly", "ever",
}

_WORD_RE = re.compile(r"[A-Za-z']+")
_SENT_RE = re.compile(r"[.!?]+")

FEATURE_NAMES = [
    "word_count",
    "avg_word_len",
    "avg_sentence_len",
    "type_token_ratio",
    "exclamation_ratio",
    "caps_ratio",
    "first_person_ratio",
    "superlative_ratio",
]


def extract(text: str) -> list[float]:
    """Return the raw feature vector (order matches FEATURE_NAMES)."""
    text = text or ""
    words = _WORD_RE.findall(text.lower())
    n = len(words) or 1
    letters = [c for c in text if c.isalpha()]
    n_letters = len(letters) or 1
    sentences = [s for s in _SENT_RE.split(text) if s.strip()]
    n_sent = len(sentences) or 1

    return [
        float(len(words)),
        sum(len(w) for w in words) / n,
        len(words) / n_sent,
        len(set(words)) / n,                                  # lexical diversity
        text.count("!") / max(len(text), 1),
        sum(c.isupper() for c in letters) / n_letters,
        sum(w in FIRST_PERSON for w in words) / n,
        sum(w in SUPERLATIVES for w in words) / n,
    ]


class LinguisticFeatures(BaseEstimator, TransformerMixin):
    """scikit-learn transformer: list[str] -> dense feature matrix."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray([extract(t) for t in X], dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURE_NAMES, dtype=object)
