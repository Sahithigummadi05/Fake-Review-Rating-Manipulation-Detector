# Data

This project uses two datasets.

## 1. Real labeled reviews — `deceptive-opinion.csv`

The **Deceptive Opinion Spam Corpus** (Ott et al., 2011): 1,600 hotel
reviews for 20 Chicago hotels, evenly split into

- **800 truthful** reviews crawled from TripAdvisor, Expedia, Hotels.com,
  Orbitz, Priceline, and Yelp, and
- **800 deceptive** reviews written by workers on Amazon Mechanical Turk,
  instructed to sound authentic.

Columns: `deceptive` (truthful/deceptive — the label), `hotel`, `polarity`
(positive/negative), `source`, `text`.

Used by `src/real_text_model.py` to train and evaluate the text-suspicion
classifier on genuine human-labeled ground truth.

**Citation**

> M. Ott, Y. Choi, C. Cardie, and J. T. Hancock. *Finding Deceptive Opinion
> Spam by Any Stretch of the Imagination.* ACL 2011.
> M. Ott, C. Cardie, and J. T. Hancock. *Negative Deceptive Opinion Spam.*
> NAACL-HLT 2013.

The corpus is distributed for research/educational use under CC BY-NC-SA
4.0. It is included here unmodified for reproducibility and attributed to
its authors; it is not a product of this project.

## 2. Synthetic manipulation dataset (generated)

`python -m src.data_generator` writes `restaurants.parquet`,
`reviewers.parquet`, and `reviews.parquet` here. These model *restaurant-level*
rating-manipulation campaigns (review bursts, rating uniformity, new-account
clusters) — patterns for which no public labeled dataset exists — so the
behavioral detector can be evaluated against known injected ground truth.
The `.parquet` files and any trained `.joblib` models are gitignored;
regenerate them locally.
