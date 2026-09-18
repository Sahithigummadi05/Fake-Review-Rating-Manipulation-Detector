# Restaurant Rating Manipulation Detector

Detects likely fraudulent review activity on a restaurant platform by
combining two independent signals into a single fraud-risk score per
restaurant, rather than trusting either signal alone:

1. **Text suspicion** — does the review text look generic/templated rather
   than naturally written?
2. **Behavioral anomaly** — does the *pattern* of ratings around a
   restaurant look organic, or does it show signs of a coordinated
   manipulation campaign (sudden rating spikes, review bursts, suspiciously
   uniform 5-star clusters, or clusters of low-activity accounts reviewing
   the same place in a short window)?

## Why the behavioral signal matters, not just text

Most public "fake review detector" projects only look at review **text**
(is this sentence bot-written?). That catches obviously templated fake
reviews, but misses the more common real-world case: a human writes a
perfectly natural-sounding review, but was paid to post it, or is one of
many accounts posting near-identical high ratings for a restaurant within a
tight time window. That kind of manipulation is invisible at the
single-review level and only shows up in the **pattern across many reviews
over time** — which is what the behavioral module is for.

## Architecture

```
Reviews + rating history
        │
        ├──► Text Suspicion Model ──► text_score  (0-1)
        │      (TF-IDF + classifier: templated vs. natural review style)
        │
        └──► Behavioral Anomaly Detector ──► behavior_score (0-1)
               ├── review burst detection   (z-score on daily review volume)
               ├── rating uniformity check  (variance of ratings vs. expected)
               └── low-activity account clustering

     0.4 * text_score + 0.6 * behavior_score ──► fraud_score  (0-1)
```

The behavioral signal is weighted higher (0.6 vs. 0.4) because it is the
harder signal to fake and the more reliable of the two.

## Datasets

The two signals are evaluated on two different datasets, because they need
different kinds of ground truth (see [`data/README.md`](data/README.md)):

- **Text model → real labeled data.** The **Deceptive Opinion Spam Corpus**
  (Ott et al., 2011): 1,600 hotel reviews, 800 genuine (crawled from
  TripAdvisor/Expedia/etc.) and 800 deceptive (crowd-written on Amazon
  Mechanical Turk). Human ground-truth labels, so the text classifier is
  measured on real deceptive-vs-truthful separability.
- **Behavioral detector → synthetic benchmark.** No public dataset labels
  which *restaurants* have had their ratings manipulated, so
  `src/data_generator.py` synthesizes 500 organic + 100 manipulated
  restaurants with known injected fraud patterns (review bursts, uniform
  5-star clusters, new-account clusters) to test the pattern detection.

## Results

### Text-suspicion model — real data (Ott Deceptive Opinion Spam Corpus)

`python -m src.real_text_model` trains a TF-IDF + classifier on the 1,600
gold-labeled reviews and reports 5-fold cross-validated performance:

| Model | Accuracy | F1 |
|-------|----------|----|
| **Logistic Regression** | **0.895** | **0.896** |
| Linear SVM | 0.890 | 0.892 |
| Multinomial NB | 0.886 | 0.890 |

~89% accuracy on held-out folds is consistent with published results on
this corpus — deceptive reviews are detectable from text but far from
trivially so, which is the honest, real-world number.

### Full pipeline — synthetic manipulation benchmark

`python -m src.evaluate` runs the combined text + behavioral score against
the injected restaurant-level ground truth, sweeping the threshold:

| Threshold | Precision | Recall |
|-----------|-----------|--------|
| 0.3       | 0.17      | 1.00   |
| 0.5       | 0.33      | 1.00   |
| 0.6       | 0.60      | 1.00   |
| **0.7**   | **1.00**  | **1.00** |

At threshold 0.7 the combined score flags every injected manipulation
campaign with no false positives, and the sweep shows precision climbing as
the threshold tightens at full recall.

## Limitations

On the synthetic benchmark the separation is clean because the generator
injects a strong, distinct fraud signature; that result validates the
pipeline end-to-end rather than predicting live-platform accuracy. Real
manipulation is noisier and adversarial — fraudsters vary ratings, spread
reviews over time, and use aged/purchased accounts to evade detectors like
this. The **real number to trust is the ~89% on the human-labeled text
corpus**; the natural next steps are adversarial synthetic data for the
behavioral module and, in production, restaurant-level labels from real
moderation/takedown data.

## Project layout

```
Fake-Review-Rating-Manipulation-Detector/
├── data/
│   ├── deceptive-opinion.csv   # real Ott et al. labeled corpus (1,600 reviews)
│   └── README.md               # dataset sources, citation, license
├── src/
│   ├── real_text_model.py      # TF-IDF + classifier on the REAL labeled corpus
│   ├── data_generator.py       # builds the synthetic restaurants/reviews/accounts dataset
│   ├── text_features.py        # text-suspicion model for the synthetic pipeline
│   ├── behavioral_features.py  # burst / uniformity / account-cluster anomaly detection
│   ├── fraud_scorer.py         # combines both signals into a final score
│   ├── evaluate.py             # precision/recall/confusion matrix against ground truth
│   └── api.py                  # FastAPI endpoint to score a restaurant
├── tests/
│   ├── test_pipeline.py        # behavioral-detector unit tests
│   └── test_real_text_model.py # real text-model tests
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python -m src.real_text_model    # real corpus: 5-fold CV report + saves model
python -m src.data_generator     # builds the synthetic benchmark (data/*.parquet)
python -m src.evaluate           # combined pipeline precision/recall/confusion matrix
pytest                           # run the unit tests

uvicorn src.api:app --reload     # scoring API on http://localhost:8000
```

## Tech stack

Python · pandas · NumPy · scikit-learn (TF-IDF, Logistic Regression / SVM / NB) ·
FastAPI · pytest

## Dataset citation

Ott, Choi, Cardie, Hancock. *Finding Deceptive Opinion Spam by Any Stretch
of the Imagination.* ACL 2011. See [`data/README.md`](data/README.md) for
details and license.
