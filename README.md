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
harder signal to fake and the more reliable of the two; the text model is
trained on weaker labels and is intentionally the secondary signal.

## Dataset

There is no public, reliably-labeled dataset that pairs restaurant review
*text* with *rating-manipulation ground truth* — real platforms don't
publish which restaurants they've caught gaming ratings. So this project
uses a **synthetic dataset** with known ground truth:

- **~500 organic restaurants** — natural rating distributions (per-restaurant
  mean rating, realistic variance, a steady low review rate over time, and
  reviewer accounts with varied activity history).
- **~100 manipulated restaurants** — injected with one or more fraud patterns:
  a burst of reviews in a short window, unusually uniform 5-star ratings,
  and/or a cluster of reviews from low-activity/new accounts.

Because the labels are injected by the generator, ground truth is known
exactly, which makes real precision/recall evaluation possible
(`src/evaluate.py`).

## Results

Running `python -m src.evaluate` on the default synthetic dataset
(500 organic + 100 manipulated restaurants) sweeps the decision threshold:

| Threshold | Precision | Recall |
|-----------|-----------|--------|
| 0.3       | 0.17      | 1.00   |
| 0.5       | 0.33      | 1.00   |
| 0.6       | 0.60      | 1.00   |
| **0.7**   | **1.00**  | **1.00** |

At the tuned operating point (0.7) the combined score flags every
manipulated restaurant with no false positives
(confusion matrix `[[500, 0], [0, 100]]`), while the sweep shows how
precision climbs as the threshold tightens at full recall.

## Limitations

The clean separation reflects that the synthetic generator injects a
strong, distinct fraud signature that does not overlap with the organic
distribution. Real-world manipulation is noisier and adversarial —
fraudsters vary ratings, spread reviews over time, and use aged or
purchased accounts specifically to evade detectors like this. These results
show the pipeline is correct end-to-end and that each signal contributes;
the natural next step is stress-testing against noisier, adversarial
synthetic data (and, in production, labels from real moderation/takedown
data) to find where detection degrades.

## Project layout

```
Fake-Review-Rating-Manipulation-Detector/
├── data/                       # generated synthetic dataset lands here
├── src/
│   ├── data_generator.py       # builds the synthetic restaurants/reviews/accounts dataset
│   ├── text_features.py        # TF-IDF + classifier for templated-text suspicion
│   ├── behavioral_features.py  # burst / uniformity / account-cluster anomaly detection
│   ├── fraud_scorer.py         # combines both signals into a final score
│   ├── evaluate.py             # precision/recall/confusion matrix against ground truth
│   └── api.py                  # FastAPI endpoint to score a restaurant
├── tests/
│   └── test_pipeline.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python -m src.data_generator     # builds data/*.parquet
python -m src.evaluate           # trains + prints precision/recall/confusion matrix
pytest                           # run the unit tests

uvicorn src.api:app --reload     # scoring API on http://localhost:8000
```

## Tech stack

Python · pandas · NumPy · scikit-learn (TF-IDF, classifier) · FastAPI · pytest
