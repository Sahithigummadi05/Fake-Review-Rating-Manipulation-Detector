"""
Smoke tests for the real-data text model. Skipped automatically if the
Deceptive Opinion Spam Corpus CSV isn't present locally.
"""

from pathlib import Path

import pytest
from sklearn.model_selection import train_test_split

from src.real_text_model import DATA_PATH, build_pipeline, load_corpus

pytestmark = pytest.mark.skipif(
    not Path(DATA_PATH).exists(), reason="deceptive-opinion.csv not available"
)


def test_corpus_loads_balanced():
    texts, labels = load_corpus()
    assert len(texts) == len(labels) == 1600
    # Corpus is an even 800/800 truthful/deceptive split.
    assert labels.sum() == 800


def test_pipeline_learns_signal():
    """On a small train/test split the classifier should clearly beat
    chance -- guards against the pipeline silently breaking."""
    texts, labels = load_corpus()
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    acc = pipe.score(X_test, y_test)
    assert acc > 0.75
