"""
Generates a synthetic restaurants/reviewers/reviews dataset with known
ground-truth fraud labels, since no public dataset pairs review text with
verified rating-manipulation labels.

~500 restaurants get organic review patterns. ~100 get one or more injected
manipulation patterns (review burst, uniform 5-star ratings, low-activity
account clustering). The ground-truth `is_manipulated` flag is written to
data/restaurants.parquet for evaluation only -- the detector never sees it
as an input feature.

Run:
    python src/data_generator.py
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
SEED = 42

NUM_ORGANIC_RESTAURANTS = 500
NUM_MANIPULATED_RESTAURANTS = 100
SIMULATION_DAYS = 180

NATURAL_REVIEW_TEMPLATES = [
    "The {dish} here was really good but the wait was longer than I expected.",
    "Decent place, {dish} was tasty though a bit overpriced for the portion.",
    "Loved the ambience, service was a little slow but the {dish} made up for it.",
    "Not bad. {dish} was okay, nothing special but I'd come back.",
    "Really enjoyed the {dish}, though the place was quite crowded on a weekend.",
    "Mixed experience -- great {dish}, but they got my order wrong once.",
    "Solid neighborhood spot. {dish} is their standout item honestly.",
    "First time here, {dish} was good, staff could be friendlier though.",
]

TEMPLATED_FRAUD_PHRASES = [
    "Amazing food and amazing service! Highly recommend this place to everyone!",
    "Best restaurant ever! Five stars! Will definitely come again!",
    "Excellent experience, excellent food, excellent staff! Perfect in every way!",
    "This is the best place in the city! You must try it! Five stars!",
    "Outstanding food quality and outstanding service! Highly recommended!",
]

DISHES = ["biryani", "paneer tikka", "butter chicken", "dosa", "pasta", "pizza", "noodles", "curry"]


def _random_timestamp(start: datetime, days_span: int, rng: random.Random) -> datetime:
    offset_seconds = rng.uniform(0, days_span * 86400)
    return start + timedelta(seconds=offset_seconds)


def generate_dataset(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    sim_start = datetime(2026, 1, 1)

    restaurants = []
    reviewers = []
    reviews = []

    reviewer_counter = 0
    review_counter = 0

    def make_reviewer(account_age_days: float) -> int:
        nonlocal reviewer_counter
        account_id = reviewer_counter
        reviewer_counter += 1
        reviewers.append({
            "account_id": account_id,
            "account_age_days": account_age_days,
            "past_review_count": max(0, int(np_rng.normal(account_age_days / 20, 3))),
        })
        return account_id

    # --- Organic restaurants ---
    for i in range(NUM_ORGANIC_RESTAURANTS):
        restaurant_id = i
        true_quality = np_rng.normal(3.9, 0.5)  # underlying "real" quality
        num_reviews = rng.randint(15, 120)

        restaurants.append({
            "restaurant_id": restaurant_id,
            "name": f"Restaurant {restaurant_id}",
            "is_manipulated": False,
        })

        for _ in range(num_reviews):
            account_id = make_reviewer(account_age_days=rng.uniform(30, 1200))
            rating = float(np.clip(np_rng.normal(true_quality, 0.8), 1, 5))
            dish = rng.choice(DISHES)
            text = rng.choice(NATURAL_REVIEW_TEMPLATES).format(dish=dish)
            timestamp = _random_timestamp(sim_start, SIMULATION_DAYS, rng)

            reviews.append({
                "review_id": review_counter,
                "restaurant_id": restaurant_id,
                "account_id": account_id,
                "rating": round(rating, 1),
                "text": text,
                "timestamp": timestamp,
            })
            review_counter += 1

    # --- Manipulated restaurants ---
    for j in range(NUM_MANIPULATED_RESTAURANTS):
        restaurant_id = NUM_ORGANIC_RESTAURANTS + j
        true_quality = np_rng.normal(3.5, 0.5)  # often mediocre underlying quality
        num_organic_reviews = rng.randint(5, 30)

        restaurants.append({
            "restaurant_id": restaurant_id,
            "name": f"Restaurant {restaurant_id}",
            "is_manipulated": True,
        })

        # A thin layer of genuine organic reviews, same as above.
        for _ in range(num_organic_reviews):
            account_id = make_reviewer(account_age_days=rng.uniform(30, 1200))
            rating = float(np.clip(np_rng.normal(true_quality, 0.8), 1, 5))
            dish = rng.choice(DISHES)
            text = rng.choice(NATURAL_REVIEW_TEMPLATES).format(dish=dish)
            timestamp = _random_timestamp(sim_start, SIMULATION_DAYS, rng)

            reviews.append({
                "review_id": review_counter,
                "restaurant_id": restaurant_id,
                "account_id": account_id,
                "rating": round(rating, 1),
                "text": text,
                "timestamp": timestamp,
            })
            review_counter += 1

        # Injected manipulation burst: many reviews in a tight window, mostly
        # 5-star, templated text, low-activity accounts.
        burst_start = sim_start + timedelta(days=rng.uniform(0, SIMULATION_DAYS - 5))
        burst_size = rng.randint(30, 80)
        burst_window_days = rng.uniform(1, 4)

        for _ in range(burst_size):
            account_id = make_reviewer(account_age_days=rng.uniform(0, 15))  # new accounts
            rating = float(np.clip(np_rng.normal(4.9, 0.15), 1, 5))  # near-uniform 5s
            text = rng.choice(TEMPLATED_FRAUD_PHRASES)
            timestamp = _random_timestamp(burst_start, burst_window_days, rng)

            reviews.append({
                "review_id": review_counter,
                "restaurant_id": restaurant_id,
                "account_id": account_id,
                "rating": round(rating, 1),
                "text": text,
                "timestamp": timestamp,
            })
            review_counter += 1

    restaurants_df = pd.DataFrame(restaurants)
    reviewers_df = pd.DataFrame(reviewers).drop_duplicates(subset="account_id")
    reviews_df = pd.DataFrame(reviews).sort_values("timestamp").reset_index(drop=True)

    return restaurants_df, reviewers_df, reviews_df


if __name__ == "__main__":
    restaurants_df, reviewers_df, reviews_df = generate_dataset()
    DATA_DIR.mkdir(exist_ok=True)
    restaurants_df.to_parquet(DATA_DIR / "restaurants.parquet", index=False)
    reviewers_df.to_parquet(DATA_DIR / "reviewers.parquet", index=False)
    reviews_df.to_parquet(DATA_DIR / "reviews.parquet", index=False)
    print(
        f"Generated {len(restaurants_df)} restaurants "
        f"({restaurants_df['is_manipulated'].sum()} manipulated), "
        f"{len(reviewers_df)} reviewers, {len(reviews_df)} reviews."
    )
