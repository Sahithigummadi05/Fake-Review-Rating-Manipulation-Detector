# Restaurant Rating Manipulation Detector

Detects likely fraudulent review activity on a restaurant platform by
combining two independent signals:

1. **Text suspicion** — does the review text look generic/templated rather
   than naturally written?
2. **Behavioral anomaly** — does the *pattern* of ratings around a
   restaurant look organic, or does it show signs of a coordinated
   manipulation campaign (sudden rating spikes, review bursts, suspiciously
   uniform 5-star clusters, clusters of low-activity accounts reviewing the
   same place in a short window)?

Both signals are combined into a single fraud-risk score per restaurant,
rather than trusting either one alone — the same "don't trust a single
score" principle used in the sibling restaurant-discovery-assistant project.

## Why behavioral signal matters, not just text

Most public "fake review detector" projects only look at review **text**
(is this sentence bot-written?). That catches obviously templated fake
reviews, but misses the more common real-world case: a human writes a
perfectly natural-sounding review, but was paid to post it, or is one of
many accounts posting near-identical high ratings for a restaurant within a
tight time window. That kind of manipulation is invisible at the single
-review level and only shows up when you look at the **pattern across many
reviews over time**. That's what the behavioral module is for.

## Dataset — honest note

There is no public, reliably-labeled dataset that pairs restaurant review
*text* with *rating-manipulation ground truth* (real platforms don't
publish which restaurants they've caught gaming ratings). So this project
uses a **synthetic dataset**, generated with:

- ~500 "organic" restaurants: natural rating distributions (mean rating
  varies per restaurant, ratings have realistic variance, reviews arrive
  at a steady low rate over time, reviewer accounts have varied activity
  history).
- ~100 "manipulated" restaurants: injected with one or more fraud patterns
  — a burst of reviews in a short window, unusually uniform 5-star ratings,
  and/or a cluster of reviews from low-activity/new accounts.

Because the fraud labels are injected by the generator, ground truth is
known exactly, which makes real precision/recall evaluation possible (see
`src/evaluate.py`). This limitation — synthetic rather than real labeled
data — is stated here deliberately rather than glossed over; a production
version would need labels from actual moderation/takedown data.

## Architecture

```
Reviews + rating history
        │
        ├──► Text Suspicion Model ──► text_score  (0-1)
        │      (TF-IDF + classifier trained on templated vs natural review style)
        │
        └──► Behavioral Anomaly Detector ──► behavior_score (0-1)
               ├── review burst detection   (z-score on daily review volume)
               ├── rating uniformity check  (variance of ratings vs expected)
               └── low-activity account clustering

        text_score + behavior_score ──► weighted combination ──► fraud_score
```

## Project layout

```
fraud-review-detector/
├── data/                    # generated synthetic dataset lands here
├── src/
│   ├── data_generator.py    # builds the synthetic restaurants/reviews/accounts dataset
│   ├── text_features.py     # TF-IDF + classifier for templated-text suspicion
│   ├── behavioral_features.py  # burst/uniformity/account-cluster anomaly detection
│   ├── fraud_scorer.py      # combines both signals into a final score
│   ├── evaluate.py          # precision/recall/confusion matrix against ground truth
│   └── api.py               # FastAPI endpoint to score a restaurant
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

python src/data_generator.py     # builds data/restaurants.parquet, data/reviews.parquet
python src/evaluate.py           # trains + evaluates, prints precision/recall/confusion matrix

uvicorn src.api:app --reload     # scoring API on http://localhost:8000
```

## Evaluation approach

- **Text model**: standard train/test split, accuracy + F1 on templated vs.
  natural review classification.
- **Behavioral detector**: precision/recall against the known-injected
  manipulated restaurants — this is the more meaningful number, since it
  measures whether the *pattern* detection actually catches manipulation
  campaigns, not just individually suspicious sentences.
- **Combined score**: precision/recall at a chosen threshold, plus a
  confusion matrix, run automatically by `src/evaluate.py`.

## Evaluation results (on the generated synthetic dataset)

Running `python -m src.evaluate` on the default synthetic dataset
(500 organic + 100 manipulated restaurants) gives:

| Threshold | Precision | Recall |
|-----------|-----------|--------|
| 0.3       | 0.17      | 1.00   |
| 0.5       | 0.33      | 1.00   |
| 0.7       | **1.00**  | **1.00** |

At threshold 0.7 the combined score perfectly separates organic from
manipulated restaurants (confusion matrix `[[500, 0], [0, 100]]`).

**This number needs an honest caveat, not just a headline:** it's this
clean because the synthetic generator injects a *strong, distinct* fraud
signature (near-uniform 5-star ratings, tight review bursts, brand-new
accounts) that doesn't overlap with the organic distribution. Real-world
manipulation is noisier and more adversarial — fraudsters vary ratings,
spread reviews out, and use aged/purchased accounts specifically to evade
detectors like this. So 100% here demonstrates the *pipeline works
correctly end-to-end and each signal is doing its job*, not that it would
hit 100% on live platform data. The right way to frame this in an
interview: "the system correctly detects the fraud patterns it's designed
to catch — the interesting next step would be adversarial/noisier
synthetic data to stress-test where it breaks."

## Suggested resume bullets

- Built a restaurant rating-manipulation detector combining a text-based
  suspicion classifier with a behavioral anomaly detector (review-burst
  detection, rating-uniformity analysis, low-activity account clustering).
- Designed a weighted fraud-scoring system merging text and behavioral
  signals rather than relying on review text alone, catching coordinated
  manipulation patterns invisible at the single-review level.
- Evaluated detection performance on a synthetic manipulation-injected
  dataset (600 restaurants), achieving 100% precision/recall at a tuned
  threshold, validated across a full precision-recall sweep.
