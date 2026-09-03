"""
FastAPI app exposing restaurant fraud scores.

Run:
    uvicorn src.api:app --reload
"""

from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.fraud_scorer import compute_fraud_scores

app = FastAPI(title="Restaurant Rating Manipulation Detector")

_scores_df: pd.DataFrame | None = None


@app.on_event("startup")
def load_resources() -> None:
    global _scores_df
    reviewers_df = pd.read_parquet("data/reviewers.parquet")
    reviews_df = pd.read_parquet("data/reviews.parquet")
    _scores_df = compute_fraud_scores(reviews_df, reviewers_df)


class RestaurantScore(BaseModel):
    restaurant_id: int
    fraud_score: float
    behavior_score: float
    text_score: float
    burst_score: float
    uniformity_score: float
    low_activity_account_score: float


@app.get("/score/{restaurant_id}", response_model=RestaurantScore)
def get_score(restaurant_id: int) -> RestaurantScore:
    row = _scores_df[_scores_df["restaurant_id"] == restaurant_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="restaurant not found")
    r = row.iloc[0]
    return RestaurantScore(**r.to_dict())


@app.get("/top-suspicious")
def top_suspicious(limit: int = 10) -> list[RestaurantScore]:
    top = _scores_df.head(limit)
    return [RestaurantScore(**row.to_dict()) for _, row in top.iterrows()]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "restaurants_scored": len(_scores_df) if _scores_df is not None else 0}
